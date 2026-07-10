"""Gold: site hourly utilization profile for the digital twin (Spark job).

Feeds the EV Charging Digital Twin (github.com/kjl1c20/ev-charging-digital-twin):
per-site 24-hour utilization curves, split weekday/weekend, with mean ("typical
day") and p90 ("busy day") across dates. Runs on Databricks after
build_charge_points.py, alongside site_pressure.py.

Site definition matches dashboard.build_site_view exactly: charge points sharing
the same coordinates (rounded to 6 dp) are one site, keyed by
site_key = "lat,lon". The dashboard joins this table on that key.

Method:
  1. Explode each session into hourly occupancy slices (a session 08:20-10:05
     contributes 40min@08, 60min@09, 5min@10). end_time is nullable in Silver;
     fall back to start_time + duration_minutes (same as site_pressure.py).
  2. Per site x date x hour: utilization = occupied connector-minutes
     / (site connectors * 60), capped at 1.0.
  3. Dense date grid per site (first to last observed session date) so
     zero-session hours count as 0 and don't inflate the mean.
  4. Aggregate across dates by day_type: mean + p90 per hour-of-day.
     p90 is computed across daily values (busy-day profile), not from means.

Idempotent: full overwrite of the target table.
"""

import os
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

SESSIONS_TABLE = os.getenv("SILVER_SESSIONS_TABLE", "chargepoint_analysis.silver.cps_sessions_clean")
CP_TABLE = os.getenv("SILVER_CP_TABLE", "chargepoint_analysis.silver.charge_points")
TARGET_TABLE = os.getenv("GOLD_SITE_HOURLY_PROFILE_TABLE", "chargepoint_analysis.gold.site_hourly_profile")

MAX_SESSION_HOURS = 24  # matches Silver's MAX_DURATION_MINUTES guard


def main():
    sessions = spark.table(SESSIONS_TABLE)
    cps = spark.table(CP_TABLE)

    # ---- site mapping: same-coordinate grouping, identical to build_site_view ----
    cp_ref = (
        cps.where(F.col("latitude").isNotNull() & F.col("longitude").isNotNull())
        .groupBy("cp_id")
        .agg(
            F.first("latitude", ignorenulls=True).alias("latitude"),
            F.first("longitude", ignorenulls=True).alias("longitude"),
            F.first("site_name", ignorenulls=True).alias("site_name"),
            F.first("n_connectors", ignorenulls=True).alias("n_connectors"),
        )
        .withColumn(
            "site_key",
            F.concat_ws(",", F.round("latitude", 6).cast("string"), F.round("longitude", 6).cast("string")),
        )
    )
    site_ref = cp_ref.groupBy("site_key").agg(
        F.sum("n_connectors").alias("connectors"),
        F.first("site_name", ignorenulls=True).alias("site_name"),
    )

    # ---- DQ: input validation at the boundary ----
    n_raw = sessions.count()
    s = (
        sessions
        .where(F.col("cp_id").isNotNull() & F.col("start_time").isNotNull())
        # end_time is nullable in Silver; same fallback as site_pressure.py
        .withColumn(
            "end_time_filled",
            F.coalesce(
                F.col("end_time"),
                (F.col("start_time").cast("long") + F.col("duration_minutes") * 60).cast("timestamp"),
            ),
        )
        .where(F.col("end_time_filled") > F.col("start_time"))
        .where(
            (F.col("end_time_filled").cast("long") - F.col("start_time").cast("long"))
            <= MAX_SESSION_HOURS * 3600
        )
        .join(F.broadcast(cp_ref.select("cp_id", "site_key")), "cp_id", "inner")
    )
    n_mapped = s.count()
    logger.info("Sessions: %d raw -> %d valid+mapped (unmapped cp_ids dropped, known geocoding gap)",
                n_raw, n_mapped)

    # ---- 1. explode sessions into hourly slices ----
    s = s.withColumn(
        "hour_ts",
        F.explode(
            F.sequence(
                F.date_trunc("hour", F.col("start_time")),
                F.date_trunc("hour", F.col("end_time_filled")),
                F.expr("interval 1 hour"),
            )
        ),
    )
    slice_start = F.greatest(F.col("start_time"), F.col("hour_ts"))
    slice_end = F.least(F.col("end_time_filled"), F.col("hour_ts") + F.expr("interval 1 hour"))
    s = s.withColumn(
        "occupied_min",
        (slice_end.cast("long") - slice_start.cast("long")) / 60.0,
    ).where(F.col("occupied_min") > 0)

    # ---- 2. utilization per site x date x hour ----
    hourly = (
        s.groupBy("site_key", F.to_date("hour_ts").alias("date"), F.hour("hour_ts").alias("hour"))
        .agg(F.sum("occupied_min").alias("occupied_min"))
    )

    # ---- 3. dense grid: every date x hour within each site's observed window ----
    # Zero-session hours are real zeros; without this, means are inflated.
    site_window = hourly.groupBy("site_key").agg(
        F.min("date").alias("d0"), F.max("date").alias("d1")
    )
    dates = (
        site_window
        .withColumn("date", F.explode(F.sequence("d0", "d1")))
        .select("site_key", "date")
    )
    hours = spark.range(24).select(F.col("id").cast("int").alias("hour"))
    dense = (
        dates.crossJoin(hours)
        .join(hourly, ["site_key", "date", "hour"], "left")
        .fillna(0.0, subset=["occupied_min"])
        .join(F.broadcast(site_ref), "site_key", "inner")
        .withColumn(
            "utilization",
            # clamp: overlapping raw sessions can exceed capacity; cap at 1.0
            F.when(F.col("connectors") > 0,
                   F.least(F.col("occupied_min") / (F.col("connectors") * 60.0), F.lit(1.0)))
            .otherwise(F.lit(None)),
        )
        .where(F.col("utilization").isNotNull())
        .withColumn(
            "day_type",
            F.when(F.dayofweek("date").isin(1, 7), "weekend").otherwise("weekday"),
        )
    )

    # ---- 4. aggregate mean + p90 across dates by day_type ----
    profile = (
        dense.groupBy("site_key", "day_type", "hour")
        .agg(
            F.avg("utilization").alias("mean_utilization"),
            F.expr("percentile_approx(utilization, 0.9)").alias("p90_utilization"),
            F.countDistinct("date").alias("n_days"),
        )
        .join(F.broadcast(site_ref), "site_key", "inner")
        .withColumn("ingested_at", F.current_timestamp())
    )

    # ---- DQ: output validation ----
    bad = profile.where(
        (F.col("mean_utilization") < 0) | (F.col("mean_utilization") > 1)
        | (F.col("p90_utilization") < 0) | (F.col("p90_utilization") > 1)
    ).count()
    if bad:
        raise ValueError(f"[DQ] {bad} rows with utilization outside [0,1] — aborting write")
    n_sites = profile.select("site_key").distinct().count()
    n_rows = profile.count()
    logger.info("[DQ] %d rows across %d sites (dense expectation <= %d = sites x 2 day-types x 24h)",
                n_rows, n_sites, n_sites * 48)
    flat = profile.groupBy("site_key").agg(
        (F.max("p90_utilization") - F.max("mean_utilization")).alias("p90_lift")
    ).where(F.col("p90_lift") < 0.01).count()
    logger.info("[DQ] %d sites where p90 barely exceeds mean (low day-to-day variance or sparse data)", flat)

    profile.select(
        "site_key", "site_name", "day_type", "hour",
        "mean_utilization", "p90_utilization", "n_days", "connectors", "ingested_at",
    ).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TARGET_TABLE)
    logger.info("Written %d rows to %s", n_rows, TARGET_TABLE)


if __name__ == "__main__":
    main()
