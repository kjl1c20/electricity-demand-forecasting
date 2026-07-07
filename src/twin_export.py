"""Digital-twin integration for the dashboard site card.

Builds the twin's site-profile JSON (contract v1 — see
github.com/kjl1c20/ev-charging-digital-twin, docs/site-profile-contract.md)
from gold.site_hourly_profile + silver.charge_points, then offers it two ways:

  Phase 1: st.download_button — load the file in the twin manually
  Phase 2: iframe embed — the twin (GitHub Pages) receives the profile via
           postMessage and simulates the site with zero extra clicks

The Databricks token never reaches the browser: queries run through the
dashboard's cached run_query, and only the aggregated JSON is handed to the
iframe.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

TWIN_URL = os.getenv(
    "TWIN_URL",
    "https://kjl1c20.github.io/ev-charging-digital-twin/demo/demo.html",
)
PROFILE_TABLE = os.getenv(
    "GOLD_SITE_HOURLY_PROFILE_TABLE", "chargepoint_analysis.gold.site_hourly_profile"
)
ECONOMICS_TABLE = os.getenv(
    "GOLD_SITE_ECONOMICS_TABLE", "chargepoint_analysis.gold.site_economics"
)
CP_TABLE = os.getenv("SILVER_CP_TABLE", "chargepoint_analysis.silver.charge_points")

# max_charge_rate_kw is nullable in Silver — fall back by connector type
DEFAULT_KW_BY_TYPE = {"AC": 22.0, "DC": 50.0, "RAPID": 50.0, "RAPID DC": 50.0,
                      "FAST AC": 22.0, "SLOW AC": 7.0}
DEFAULT_KW = 22.0


def _esc(v) -> str:
    return str(v).replace("'", "''")


def _in_clause(cp_ids: tuple) -> str:
    return ", ".join(f"'{_esc(c)}'" for c in cp_ids)


def _clamp01(v) -> float:
    return round(min(max(float(v), 0.0), 1.0), 4)


def build_site_profile_json(run_query, site_row) -> dict | None:
    """Contract-v1 JSON for one site row from build_site_view.

    Returns None when the site has no profile rows (e.g. unmapped cp_ids —
    the known geocoding gap) or no connector detail.
    """
    has_kw = True
    try:
        prof = run_query(f"""
            SELECT day_type, hour, mean_utilization, p90_utilization, mean_kw, p90_kw
            FROM {PROFILE_TABLE}
            WHERE site_key = '{_esc(site_row["site_key"])}'
        """)
    except Exception:
        # Older Gold table without kW columns — retry without them
        has_kw = False
        try:
            prof = run_query(f"""
                SELECT day_type, hour, mean_utilization, p90_utilization
                FROM {PROFILE_TABLE}
                WHERE site_key = '{_esc(site_row["site_key"])}'
            """)
        except Exception:
            # Gold profile table not built yet — degrade to the info message
            return None
    if prof.empty:
        return None

    profiles = {}
    for day in ("weekday", "weekend"):
        d = prof[prof["day_type"] == day]
        if d.empty:
            continue
        mean, p90 = [0.0] * 24, [0.0] * 24
        mean_kw, p90_kw = [0.0] * 24, [0.0] * 24
        for _, r in d.iterrows():
            h = int(r["hour"])
            mean[h] = _clamp01(r["mean_utilization"])
            p90[h] = _clamp01(r["p90_utilization"])
            if has_kw:
                mean_kw[h] = round(max(float(r["mean_kw"] or 0.0), 0.0), 2)
                p90_kw[h] = round(max(float(r["p90_kw"] or 0.0), 0.0), 2)
        profiles[day] = {"mean": mean, "p90": p90}
        if has_kw:
            profiles[day]["mean_kw"] = mean_kw
            profiles[day]["p90_kw"] = p90_kw
    if not profiles:
        return None

    # economics (optional — omit block entirely if table missing)
    economics = None
    try:
        e = run_query(f"""
            SELECT avg_session_revenue, avg_session_energy_kwh, avg_session_minutes,
                   sessions_per_weekday, sessions_per_weekend
            FROM {ECONOMICS_TABLE}
            WHERE site_key = '{_esc(site_row["site_key"])}'
        """)
        if not e.empty:
            r = e.iloc[0]
            num = lambda v: round(float(v), 2) if pd.notna(v) and float(v) >= 0 else None
            economics = {
                "avg_session_revenue": num(r["avg_session_revenue"]),
                "avg_session_energy_kwh": num(r["avg_session_energy_kwh"]),
                "avg_session_minutes": num(r["avg_session_minutes"]),
                "sessions_per_day": {
                    "weekday": num(r["sessions_per_weekday"]),
                    "weekend": num(r["sessions_per_weekend"]),
                },
            }
    except Exception:
        pass

    ch = run_query(f"""
        SELECT cp_id, connector_id, connector_type, max_charge_rate_kw
        FROM {CP_TABLE}
        WHERE cp_id IN ({_in_clause(site_row["cp_ids"])})
    """)
    if ch.empty:
        return None
    chargers = []
    for _, r in ch.iterrows():
        kw = r["max_charge_rate_kw"]
        if pd.isna(kw) or float(kw) <= 0:
            ctype = str(r["connector_type"]).strip().upper() if pd.notna(r["connector_type"]) else ""
            kw = DEFAULT_KW_BY_TYPE.get(ctype, DEFAULT_KW)
        chargers.append({
            "connector_id": f"{r['cp_id']}-{r['connector_id']}",
            "rated_kw": float(kw),
        })

    name = site_row["site_name"]
    payload = {
        "version": 1,
        "site_id": str(site_row["site_key"]),
        "site_name": str(name) if pd.notna(name) else "Unnamed site",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": PROFILE_TABLE,
        "chargers": chargers,
        "profiles": profiles,
    }
    if economics is not None:
        payload["economics"] = economics
    return payload


def render_twin_section(run_query, site_row, height: int = 720) -> None:
    """The 'Simulate this site in 3D' expander on the site card."""
    with st.expander("⚡ Simulate this site in 3D (digital twin)"):
        payload = build_site_profile_json(run_query, site_row)
        if payload is None:
            st.info(
                "No twin profile for this site — its charge points aren't in the "
                "locations feed (known geocoding gap), or the Gold profile table "
                "hasn't been built yet (run src/site_hourly_profile.py)."
            )
            return

        st.caption(
            "Replay this site's real demand in 3D: typical (mean) vs busy (p90) day, "
            "then add chargers or grid capacity and watch saturation change."
        )
        data = json.dumps(payload)
        # Cache-buster: GitHub Pages serves demo.html with a ~10-minute cache
        # and iframes cling to it — a changing query param forces a fresh fetch
        # each dashboard session without affecting the twin itself.
        twin_url = f"{TWIN_URL}?v={int(datetime.now(timezone.utc).timestamp()) // 600}"
        components.html(
            f"""
            <iframe id="twin" src="{twin_url}"
                    style="width:100%;height:{height - 20}px;border:0;border-radius:8px"></iframe>
            <script>
              const twin = document.getElementById("twin");
              const site = {data};
              twin.addEventListener("load", () => {{
                // small delay: the twin registers its listener after module init
                setTimeout(() => {{
                  twin.contentWindow.postMessage(
                    {{ type: "cps-site-profile", payload: site }}, "*"
                  );
                }}, 800);
              }});
            </script>
            """,
            height=height,
        )
        st.download_button(
            label="Download twin profile (JSON)",
            data=json.dumps(payload, indent=2),
            file_name=f"cps-site-{site_row['site_key'].replace(',', '_')}.json",
            mime="application/json",
            help="Load this file in the twin manually (Load CPS site JSON)",
        )
