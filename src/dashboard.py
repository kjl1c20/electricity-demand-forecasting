"""
Scotland EV Charging — Infrastructure Planning Dashboard.

Single interactive page: a demand-pressure map of every charge point; click a point to
drill into that site's metrics and demand over time. Reads the Gold demand-pressure table
and Silver sessions from Databricks via the SQL connector (aggregations pushed to SQL).

Run:  poetry run streamlit run src/dashboard.py
Needs in .env:  DATABRICKS_SERVER_HOSTNAME (or DATABRICKS_HOST), DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
Map: tokenless MapLibre basemap (no Mapbox token required).
"""

import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from databricks import sql as dbsql
from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(page_title="Scotland EV Charging — Planning", layout="wide")

GOLD_TABLE = os.getenv("GOLD_SITE_PRESSURE_TABLE", "chargepoint_analysis.gold.site_pressure")
SESSIONS_TABLE = os.getenv("SILVER_SESSIONS_TABLE", "chargepoint_analysis.silver.cps_sessions_clean")

# Map rendering with pydeck (deck.gl). Tokenless Carto basemap; the selected site is marked
# with a real teardrop pin via IconLayer using deck.gl's stock marker atlas (mask=True lets
# us tint it). Sites are a ScatterplotLayer coloured by pressure (OrRd ramp), pickable so a
# click selects a site.
MAP_STYLE = os.getenv("MAP_STYLE", "light")  # pydeck/Carto style: light | dark | road | ...
PIN_ATLAS = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png"
PIN_MAPPING = {"marker": {"x": 0, "y": 0, "width": 128, "height": 128, "anchorY": 128, "mask": True}}
PIN_COLOR = [31, 120, 180]  # blue teardrop for the selected site

# OrRd colour ramp (matches the old plotly scale) — pressure_score (0–1) → [r, g, b].
_ORRD = [(255, 247, 236), (253, 212, 158), (253, 141, 60), (227, 74, 51), (179, 0, 0)]


def _orrd_rgb(t: float) -> list:
    """Interpolate the OrRd ramp at t∈[0,1] → [r, g, b]."""
    if t is None or pd.isna(t):
        return [180, 180, 180]
    t = min(max(float(t), 0.0), 1.0)
    pos = t * (len(_ORRD) - 1)
    i = int(pos)
    if i >= len(_ORRD) - 1:
        return list(_ORRD[-1])
    f = pos - i
    a, b = _ORRD[i], _ORRD[i + 1]
    return [round(a[c] + (b[c] - a[c]) * f) for c in range(3)]

# Scottish postcode areas → place name, for the region filter
AREA_NAMES = {
    "AB": "Aberdeen", "DD": "Dundee", "DG": "Dumfries", "EH": "Edinburgh",
    "FK": "Falkirk", "G": "Glasgow", "HS": "Outer Hebrides", "IV": "Inverness",
    "KA": "Kilmarnock", "KW": "Kirkwall", "KY": "Kirkcaldy", "ML": "Motherwell",
    "PA": "Paisley", "PH": "Perth", "TD": "Borders", "ZE": "Shetland",
}
ALL_REGIONS = "All Scotland"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# Session-length bands, ordered long → short so the heatmap shows long stays at the top.
DUR_BANDS = ["8h+", "4-8h", "2-4h", "1-2h", "30-60m", "<30m"]


# ============================================================
# Databricks SQL connection + cached query helper
# ============================================================

def _conn_params():
    host = (os.getenv("DATABRICKS_SERVER_HOSTNAME") or os.getenv("DATABRICKS_HOST") or "")
    host = host.replace("https://", "").replace("http://", "").rstrip("/")
    return host, os.getenv("DATABRICKS_HTTP_PATH"), os.getenv("DATABRICKS_TOKEN")


HOST, HTTP_PATH, TOKEN = _conn_params()
if not (HOST and HTTP_PATH and TOKEN):
    st.error(
        "Missing Databricks connection settings. Set DATABRICKS_SERVER_HOSTNAME "
        "(or DATABRICKS_HOST), DATABRICKS_HTTP_PATH and DATABRICKS_TOKEN in .env."
    )
    st.stop()


@st.cache_data(show_spinner="Querying Databricks…")
def run_query(query: str) -> pd.DataFrame:
    """Run a SQL query against the Databricks warehouse, return a pandas DataFrame."""
    with dbsql.connect(server_hostname=HOST, http_path=HTTP_PATH, access_token=TOKEN) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall_arrow().to_pandas()


@st.cache_data
def load_sites():
    return run_query(f"SELECT * FROM {GOLD_TABLE}")


@st.cache_data
def load_period():
    r = run_query(
        f"SELECT min(start_time) AS date_min, max(start_time) AS date_max FROM {SESSIONS_TABLE}"
    ).iloc[0]
    return pd.to_datetime(r["date_min"]).date(), pd.to_datetime(r["date_max"]).date()


def _in_clause(cp_ids: tuple) -> str:
    return ", ".join("'" + str(c).replace("'", "''") + "'" for c in cp_ids)


@st.cache_data
def load_site_profile(cp_ids: tuple):
    """Sessions by weekday × hour for a site's charge points — drives both daily charts."""
    return run_query(f"""
        SELECT weekday(start_time) AS dow, hour(start_time) AS hour, count(*) AS sessions
        FROM {SESSIONS_TABLE}
        WHERE cp_id IN ({_in_clause(cp_ids)})
        GROUP BY weekday(start_time), hour(start_time)
    """)


@st.cache_data
def load_site_daycounts(cp_ids: tuple):
    """Distinct weekday vs weekend dates for a site — to normalise the average profile."""
    return run_query(f"""
        SELECT CASE WHEN weekday(start_time) >= 5 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
               count(DISTINCT to_date(start_time)) AS days
        FROM {SESSIONS_TABLE}
        WHERE cp_id IN ({_in_clause(cp_ids)})
        GROUP BY CASE WHEN weekday(start_time) >= 5 THEN 'Weekend' ELSE 'Weekday' END
    """)


@st.cache_data
def load_site_hour_duration(cp_ids: tuple):
    """Session counts by hour of day × session-length band — the combined timing/length heatmap."""
    band = """CASE
                WHEN duration_minutes < 30  THEN '<30m'
                WHEN duration_minutes < 60  THEN '30-60m'
                WHEN duration_minutes < 120 THEN '1-2h'
                WHEN duration_minutes < 240 THEN '2-4h'
                WHEN duration_minutes < 480 THEN '4-8h'
                ELSE '8h+'
              END"""
    return run_query(f"""
        SELECT hour(start_time) AS hour, {band} AS dur_band, count(*) AS sessions
        FROM {SESSIONS_TABLE}
        WHERE cp_id IN ({_in_clause(cp_ids)})
        GROUP BY hour(start_time), {band}
    """)


@st.cache_data
def load_site_trend(cp_ids: tuple):
    """Monthly sessions across all charge points at a site — queried when a site is clicked."""
    return run_query(f"""
        SELECT date_format(start_time, 'yyyy-MM') AS month, count(*) AS sessions
        FROM {SESSIONS_TABLE}
        WHERE cp_id IN ({_in_clause(cp_ids)})
        GROUP BY date_format(start_time, 'yyyy-MM')
        ORDER BY month
    """)


def load_site_bundle(cp_ids: tuple) -> dict:
    """Fetch all four per-site datasets concurrently — each is an independent Databricks
    round-trip (its own connection), so firing them in parallel cuts first-click latency
    from the sum of 4 round-trips to roughly the slowest one. Each loader still has its own
    @st.cache_data, so a repeat selection skips the network entirely regardless."""
    loaders = {
        "trend": load_site_trend,
        "profile": load_site_profile,
        "hour_duration": load_site_hour_duration,
        "daycounts": load_site_daycounts,
    }
    with ThreadPoolExecutor(max_workers=len(loaders)) as ex:
        futures = {name: ex.submit(fn, cp_ids) for name, fn in loaders.items()}
        return {name: f.result() for name, f in futures.items()}


@st.cache_data
def build_site_view(cp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-cp_id gold table to one row per physical site (location).

    Charge points at the same coordinates are one site. Rates are recomputed from summed
    hours (not averaged), and pressure is re-ranked across sites.
    """
    g = (
        cp.groupby(["latitude", "longitude"], dropna=False)
        .agg(
            site_name=("site_name", "first"),
            postcode=("postcode", "first"),
            postcode_area=("postcode_area", "first"),
            n_charge_points=("cp_id", "nunique"),
            n_connectors=("n_connectors", "sum"),
            total_sessions=("total_sessions", "sum"),
            total_energy_kwh=("total_energy_kwh", "sum"),
            occupied_hours=("occupied_hours", "sum"),
            available_connector_hours=("available_connector_hours", "sum"),
            saturated_hours=("saturated_hours", "sum"),
            cp_available_hours=("cp_available_hours", "sum"),
            cp_ids=("cp_id", lambda s: tuple(s)),
        )
        .reset_index()
    )
    denom_u = g["available_connector_hours"].where(g["available_connector_hours"] > 0)
    g["utilisation"] = (g["occupied_hours"] / denom_u).fillna(0).clip(upper=1.0)
    denom_s = g["cp_available_hours"].where(g["cp_available_hours"] > 0)
    g["saturation_rate"] = (g["saturated_hours"] / denom_s).fillna(0)
    g["pressure_score"] = (
        0.6 * g["saturation_rate"].rank(pct=True) + 0.4 * g["utilisation"].rank(pct=True)
    )
    g["pressure_rank"] = g["pressure_score"].rank(ascending=False, method="min").astype(int)
    g["site_key"] = g["latitude"].round(6).astype(str) + "," + g["longitude"].round(6).astype(str)
    return g


# ============================================================
# Page
# ============================================================

def render_detail(row, bundle=None):
    """The clicked-site card: site-level metrics + demand-over-time."""
    if row is None:
        st.info("Click a site on the map to inspect its sessions.")
        return

    name = row["site_name"] if pd.notna(row["site_name"]) else "Unnamed site"
    st.markdown(f"### {name}")
    loc = row["postcode"] if pd.notna(row["postcode"]) else "—"
    st.caption(f"{loc} · "
               f"score {row['pressure_score']:.3f} · "
               f"{int(row['n_charge_points'])} charge point(s) · {int(row['n_connectors'])} connectors")

    a1, a2 = st.columns(2)
    a1.metric("Total sessions", f"{int(row['total_sessions']):,}")
    a2.metric("Total energy", f"{row['total_energy_kwh'] / 1000:,.1f} MWh")
    b1, b2 = st.columns(2)
    b1.metric("Utilisation", f"{row['utilisation']:.1%}")
    b2.metric("Saturation rate", f"{row['saturation_rate']:.1%}")

    trend = bundle["trend"]
    fig_trend = px.area(trend, x="month", y="sessions", title="Demand over time")
    fig_trend.update_traces(line_color="#d73027", fillcolor="rgba(215,48,39,0.15)")
    fig_trend.update_layout(height=290, margin=dict(t=40, b=0, l=0, r=0),
                            xaxis_title="", yaxis_title="Sessions / month")
    st.plotly_chart(fig_trend, use_container_width=True)


def render_site_profiles(row, bundle):
    """Per-site charging profile: hour×day heatmap, hour×duration heatmap, weekday vs weekend."""
    if row is None:
        return
    name = row["site_name"] if pd.notna(row["site_name"]) else "this site"
    st.markdown(f"#### When does **{name}** get used?")

    prof = bundle["profile"]
    p1, p2 = st.columns(2, gap="large")

    with p1:
        st.caption("Sessions by hour and day of week")
        mat = (prof.pivot(index="dow", columns="hour", values="sessions")
               .reindex(range(7)).reindex(columns=range(24)).fillna(0))
        fig_h = px.imshow(
            mat.values, x=list(range(24)), y=DOW, aspect="auto",
            color_continuous_scale="OrRd", labels=dict(x="Hour of day", y="", color="Sessions"),
        )
        fig_h.update_layout(height=300, margin=dict(t=10, b=0, l=0, r=0))
        st.plotly_chart(fig_h, use_container_width=True)

    with p2:
        st.caption("When do long vs short sessions happen?")
        hd = bundle["hour_duration"]
        mat2 = (hd.pivot(index="dur_band", columns="hour", values="sessions")
                .reindex(DUR_BANDS).reindex(columns=range(24)).fillna(0))
        fig_hd = px.imshow(
            mat2.values, x=list(range(24)), y=DUR_BANDS, aspect="auto",
            color_continuous_scale="OrRd",
            labels=dict(x="Hour of day", y="Session length", color="Sessions"),
        )
        fig_hd.update_layout(height=300, margin=dict(t=10, b=0, l=0, r=0))
        st.plotly_chart(fig_hd, use_container_width=True)

    st.caption("Average charging profile — weekday vs weekend")
    agg = prof.copy()
    agg["day_type"] = agg["dow"].apply(lambda d: "Weekend" if d >= 5 else "Weekday")
    agg = agg.groupby(["day_type", "hour"], as_index=False)["sessions"].sum()
    days = bundle["daycounts"].set_index("day_type")["days"].to_dict()
    agg["avg_sessions"] = agg.apply(lambda r: r["sessions"] / days.get(r["day_type"], 1), axis=1)
    fig_w = px.line(
        agg, x="hour", y="avg_sessions", color="day_type", markers=True,
        color_discrete_map={"Weekday": "#4575b4", "Weekend": "#d73027"},
    )
    fig_w.update_layout(
        height=280, margin=dict(t=10, b=0, l=0, r=0),
        xaxis_title="Hour of day", yaxis_title="Avg sessions / day", legend_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    st.plotly_chart(fig_w, use_container_width=True)


@st.fragment
def map_and_detail():
    """Map + detail. Pick a site by name or by clicking the map; one site is preselected.
    Defaults to the first postcode area (showing all of Scotland at once isn't useful)."""
    map_col, detail_col = st.columns([3, 2], gap="large")

    with map_col:
        areas = sorted(site_view["postcode_area"].dropna().unique())
        region = st.selectbox(
            "Postcode area", areas,
            format_func=lambda a: f"{a} — {AREA_NAMES.get(a, a)}",
        )
        st.caption("Click a site on the map to inspect its sessions.")

        mp = (site_view[(site_view["postcode_area"] == region)
                        & site_view["latitude"].notna() & site_view["longitude"].notna()]
              .sort_values("pressure_rank"))
        option_keys = mp["site_key"].tolist()
        name_by_key = dict(zip(mp["site_key"], mp["site_name"]))

        if mp.empty:
            st.info(f"No sites with coordinates in {region}.")
            return

        # A map click (stored in session_state by on_select) preselects that site in the box.
        # pydeck returns clicked rows under selection.objects[<layer id>]; gated by _applied_click
        # so a stale click can't fight a manual dropdown change.
        sitebox_key = f"sitebox::{region}"
        sel_state = st.session_state.get("pressure_map") or {}
        hits = ((sel_state.get("selection") or {}).get("objects") or {}).get("sites") or []
        clicked = hits[0].get("site_key") if hits else None
        if clicked in option_keys and clicked != st.session_state.get("_applied_click"):
            st.session_state[sitebox_key] = clicked
            st.session_state["_applied_click"] = clicked

        selected_key = st.selectbox(
            "Site", option_keys, format_func=lambda k: name_by_key.get(k, k), key=sitebox_key,
        )
        sel = mp[mp["site_key"] == selected_key]
        show_all = st.toggle("Show all Scotland", value=False)

        # Render frame: per-point OrRd colour + pre-formatted tooltip fields (deck.gl tooltips
        # can't format numbers, so we format them here).
        # show_all uses two layers: a background of all Scotland at reduced size/alpha (OrRd
        # colour preserved so pressure is readable everywhere), and a foreground of in-region
        # sites at full size/alpha so the selected area stands out clearly.
        def _with_disp(df):
            df = df.copy()
            df["score_disp"] = df["pressure_score"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
            df["sat_disp"] = df["saturation_rate"].map(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "—")
            df["util_disp"] = df["utilisation"].map(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "—")
            return df

        if show_all:
            bg_r = _with_disp(
                site_view[site_view["latitude"].notna() & site_view["longitude"].notna()]
            )
            bg_r["fill_color"] = bg_r["pressure_score"].apply(lambda v: _orrd_rgb(v) + [80])
            map_r = _with_disp(mp)
            map_r["fill_color"] = map_r["pressure_score"].apply(lambda v: _orrd_rgb(v) + [220])
            view_lat, view_lon, view_zoom = 57.0, -4.0, 6.0
        else:
            bg_r = None
            map_r = _with_disp(mp)
            map_r["fill_color"] = map_r["pressure_score"].apply(lambda v: _orrd_rgb(v) + [200])
            view_lat = float(mp["latitude"].mean())
            view_lon = float(mp["longitude"].mean())
            view_zoom = 8.5

        layers = []
        if show_all:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=bg_r, id="sites_bg", pickable=True, auto_highlight=False,
                get_position=["longitude", "latitude"], get_fill_color="fill_color",
                get_radius=80, radius_min_pixels=3, radius_max_pixels=10,
                stroked=False,
            ))
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=map_r, id="sites", pickable=True, auto_highlight=True,
            get_position=["longitude", "latitude"], get_fill_color="fill_color",
            get_radius=120, radius_min_pixels=6, radius_max_pixels=16,
            stroked=True, get_line_color=[0, 0, 0], line_width_min_pixels=1,
        ))
        if not sel.empty:  # real teardrop pin on the selected site (deck.gl marker atlas)
            layers.append(pdk.Layer(
                "IconLayer", data=sel.assign(icon="marker"), get_icon="icon",
                get_position=["longitude", "latitude"], get_size=4, size_scale=8,
                get_color=PIN_COLOR, icon_atlas=PIN_ATLAS, icon_mapping=PIN_MAPPING,
                pickable=False,
            ))

        deck = pdk.Deck(
            layers=layers, map_provider="carto", map_style=MAP_STYLE,
            initial_view_state=pdk.ViewState(
                latitude=view_lat, longitude=view_lon, zoom=view_zoom),
            tooltip={
                "html": (
                    "<b>{site_name}</b> (Rank {pressure_rank})<br/>"
                    "{postcode}<br/>"
                    "Saturation rate: {sat_disp}<br/>"
                    "Utilisation: {util_disp}<br/>"
                    "{n_charge_points} charge point(s) · {n_connectors} connectors"
                ),
                "style": {"backgroundColor": "#262730", "color": "white", "fontSize": "0.8rem"},
            },
        )
        st.pydeck_chart(deck, use_container_width=True, height=640,  # native trackpad/scroll zoom
                        on_select="rerun", selection_mode="single-object", key="pressure_map")
        if show_all:
            st.caption(
                f"{len(map_r):,} sites across Scotland · "
                f"{len(mp):,} highlighted in {region} — {AREA_NAMES.get(region, region)}."
            )
        else:
            st.caption(f"{len(mp):,} sites in {region} — {AREA_NAMES.get(region, region)}.")

    site_row = sel.iloc[0] if not sel.empty else None
    # One concurrent fetch shared by both render functions below — avoids firing the same
    # 4 queries twice and means render_detail/render_site_profiles never block each other.
    bundle = load_site_bundle(site_row["cp_ids"]) if site_row is not None else None

    with detail_col:
        with st.container(border=True):
            render_detail(site_row, bundle)

    if site_row is not None:
        st.markdown("")
        render_site_profiles(site_row, bundle)


sites = load_sites()
site_view = build_site_view(sites)
date_min, date_max = load_period()

st.title("Scotland EV Charging Profile Analysis")
st.caption(
    f"ChargePlace Scotland public network · demand pressure by site · "
    f"{date_min} → {date_max}"
)

st.subheader("Where is the network under pressure?")
st.caption(
    "Each point is a **site** (the charge points at one location), coloured by demand-pressure "
    "score (saturation 60% + utilisation 40%)."
)

map_and_detail()

st.divider()
st.caption("Data source: ChargePlace Scotland public session data · chargeplacescotland.org")
