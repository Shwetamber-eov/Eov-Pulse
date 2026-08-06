"""
EOV-PULSE — Clinical lab report analysis
=========================================
Streamlit UI rebuilt to match the EOV-PULSE reference design (three tabs:
Analysis / Action plan / Nearby specialists), while keeping the original
data flow: upload -> extract_tables_and_text -> clinical_agent.invoke ->
extract_specialists -> fetch_local_doctors -> gmap.

----------------------------------------------------------------------------
EXPECTED BACKEND DATA CONTRACT
----------------------------------------------------------------------------
Every read from `result`, `specialists`, and `nearby_specialists` goes
through small get_*() adapters that use .get(...) with fallbacks, so the
app won't crash if a field is missing — that section just renders a
lighter version. For the full design to show up (status counts, colored
range bars, the priority-ordered action plan, specialist cards), shape
`clinical_agent.invoke(...)`'s return value roughly like this:

result = {
    "report_text": "...",
    "extracted_data": [
        {
            "summary": "Narrative paragraph describing the report...",
            "patient": {"age": 39, "sex": "male"},
            "report_date": "28 July 2026",
            "parameters": [
                {
                    "name": "Total cholesterol",
                    "category": "Lipid profile",          # groups table rows
                    "guideline_label": "NCEP ATP IV",      # small tag per group
                    "value": 235.6,
                    "unit": "mg/dL",
                    "reference_low": None,
                    "reference_high": 200,
                    "reference_label": "< 200",            # display string
                    "status": "elevated",                  # elevated | low | in_range
                },
                # ...
            ],
            "flagged_systems": [
                {"name": "Cardiovascular / lipids", "detail": "4 out of range", "status": "bad"},
                {"name": "Glycaemic control", "detail": "Normal", "status": "good"},
                # ...
            ],
            "next_retest": "In 2\u20134 weeks",
            "next_retest_detail": "Repeat testosterone on a morning sample",
        }
    ],
    "guideline_context": "...",   # long text, OR list of {"title":..., "description":...}
    "final_plan": {
        "specialists": [""],
        "follow_up_plan": "...",  # kept for backward-compat / plain-text fallback
        "actions": [
            {
                "priority": "high",  # high | medium | monitor
                "parameter_label": "LDL 151.92 mg/dL, non-HDL 183.3 mg/dL, hs-CRP 2.60 mg/L",
                "title": "Bring LDL and non-HDL cholesterol down",
                "description": "LDL sits 52 mg/dL above the ATP IV target...",
                "steps": ["Reduce saturated fat to under 7% of daily calories...", "..."],
                "retest_window": "Lipid panel in 12 weeks",
                "refer_to": "Cardiology or internal medicine",
            },
        ],
        "retest_schedule": [
            {"when": "2\u20134 weeks", "detail": "Repeat total testosterone, morning fasting sample..."},
        ],
    },
}

If your backend doesn't return this shape yet, the adapters fall back to
whatever text/values *are* present. Update graph_multiagent.py whenever
convenient — the UI degrades gracefully either way.
"""

import streamlit as st
from streamlit_geolocation import streamlit_geolocation
from geopy.geocoders import Nominatim

from pathlib import Path
import sys

# from gmap_test import gmap
from gmap_test2 import gmap

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# from app.backend.graph_multiagent import clinical_agent
from app.backend.graph_multiagent_heavy import clinical_agent
from app.backend.map_scraper import fetch_local_doctors, extract_specialists
from testing.clinical import extract_tables_and_text

st.set_page_config(page_title="EOV Pulse", layout="wide", page_icon="\u26a1", initial_sidebar_state="collapsed")

# ============================================================================
# STYLE
# ============================================================================

RED = "#DC2626"
RED_BG = "#FEE2E2"
GREEN = "#15803D"
GREEN_BG = "#DCFCE7"
AMBER = "#92400E"
AMBER_BG = "#FEF3C7"
GRAY = "#374151"
GRAY_BG = "#F3F4F6"
BLUE = "#2563EB"
BLUE_DARK = "#1D4ED8"
BORDER = "#E5E7EB"
TEXT_MUTED = "#6B7280"
TEXT_MAIN = "#111827"

STATUS_STYLES = {
    "elevated": ("Elevated", RED, RED_BG),
    "low": ("Low", RED, RED_BG),
    "in_range": ("In range", GREEN, GREEN_BG),
    "high": ("Elevated", RED, RED_BG),
    "normal": ("In range", GREEN, GREEN_BG),
}

PRIORITY_STYLES = {
    "high": ("High priority", "#B91C1C", RED_BG),
    "medium": ("Medium priority", AMBER, AMBER_BG),
    "monitor": ("Monitor", GRAY, GRAY_BG),
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: {TEXT_MAIN};
}}

.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1180px; }}

.eov-serif {{ font-family: 'Source Serif 4', Georgia, serif; }}

/* ---- top header ---- */
.eov-header {{
    display: flex; justify-content: space-between; align-items: center;
    padding-bottom: .9rem; border-bottom: 1px solid {BORDER}; margin-bottom: 1.4rem;
}}
.eov-logo {{ font-weight: 700; font-size: 1.05rem; letter-spacing: .01em; }}
.eov-logo-caption {{ color: {TEXT_MUTED}; font-size: .85rem; margin-left: .6rem; }}
.eov-header-right {{ color: {TEXT_MUTED}; font-size: .85rem; display:flex; gap:.9rem; align-items:center; }}
.eov-header-right b {{ color: {TEXT_MAIN}; font-weight: 500; }}

/* ---- page title block ---- */
.eov-meta {{ color: {TEXT_MUTED}; font-size: .82rem; margin-bottom: .15rem; }}
.eov-title {{ font-size: 2.0rem; font-weight: 600; margin: 0 0 .3rem 0; }}
.eov-subtitle {{ color: {TEXT_MUTED}; font-size: .9rem; margin-bottom: 1.1rem; }}

/* ---- metric / stat cards ---- */
.eov-card {{
    border: 1px solid {BORDER}; border-radius: 10px; padding: 1rem 1.1rem;
    background: #fff; height: 100%;
}}
.eov-card-label {{ color: {TEXT_MUTED}; font-size: .78rem; margin-bottom: .45rem; }}
.eov-card-value {{ font-size: 1.6rem; font-weight: 600; line-height: 1.1; }}
.eov-card-sub {{ color: {TEXT_MUTED}; font-size: .78rem; margin-top: .35rem; }}
.eov-status-dot {{
    display:inline-block; width:8px; height:8px; border-radius:2px; background:{RED}; margin-right:.4rem;
}}

/* ---- tab bar ---- */
div[data-testid="stHorizontalBlock"] .eov-tabbtn button {{
    border: none !important; background: transparent !important; box-shadow: none !important;
    border-radius: 0 !important; padding-bottom: .6rem !important; font-weight: 500 !important;
    color: {TEXT_MUTED} !important;
}}
.eov-tab-active button {{ color: {TEXT_MAIN} !important; border-bottom: 2px solid {TEXT_MAIN} !important; }}

/* ---- generic content card ---- */
.eov-panel {{
    border: 1px solid {BORDER}; border-radius: 10px; padding: 1.3rem 1.4rem; background: #fff;
}}
.eov-panel-title {{ font-size: .78rem; color: {TEXT_MUTED}; text-transform: none; margin-bottom: .6rem; }}
.eov-narrative {{ font-size: .95rem; line-height: 1.6; }}

.eov-flag-row {{ display:flex; justify-content: space-between; padding: .5rem 0; border-bottom: 1px solid {BORDER}; font-size: .85rem; }}
.eov-flag-row:last-child {{ border-bottom: none; }}

/* ---- badges / chips ---- */
.eov-badge {{
    display:inline-block; padding: .18rem .55rem; border-radius: 5px; font-size: .74rem; font-weight: 500;
    white-space: nowrap;
}}

/* ---- lab value table ---- */
.eov-group-header {{
    font-size: .78rem; font-weight: 600; color: {TEXT_MAIN}; text-transform: none;
    padding: .8rem 0 .35rem 0; display:flex; justify-content: space-between; border-bottom: 1px solid {BORDER};
}}
.eov-group-header span.tag {{ color: {TEXT_MUTED}; font-weight: 400; }}
.eov-row {{ display:flex; align-items:center; padding: .55rem 0; border-bottom: 1px solid #F3F4F6; font-size: .87rem; }}
.eov-row-name {{ flex: 1.6; }}
.eov-row-value {{ flex: 1.1; }}
.eov-row-value b {{ font-weight: 600; }}
.eov-row-ref {{ flex: 1.1; color: {TEXT_MUTED}; }}
.eov-row-bar {{ flex: 1.6; padding-right: 1rem; }}
.eov-row-status {{ flex: 0.9; text-align: right; }}

.eov-bar-track {{ position: relative; height: 6px; border-radius: 3px; background: #F3F4F6; }}
.eov-bar-range {{ position:absolute; top:0; bottom:0; background: #BBF7D0; border-radius: 3px; }}
.eov-bar-marker {{ position:absolute; top:-3px; width:2px; height:12px; border-radius:1px; }}

/* ---- action plan ---- */
.eov-action-card {{ border-bottom: 1px solid {BORDER}; padding: 1.3rem 0; }}
.eov-action-card:first-child {{ padding-top: 0; }}
.eov-action-tag {{ color: {TEXT_MUTED}; font-size: .78rem; margin-top: .2rem; }}
.eov-action-title {{ font-size: 1.05rem; font-weight: 600; margin: .3rem 0 .4rem 0; }}
.eov-action-desc {{ font-size: .88rem; line-height: 1.55; color: #374151; margin-bottom: .6rem; }}
.eov-action-step {{ font-size: .85rem; line-height: 1.6; color: #374151; margin-left: .1rem; }}
.eov-action-meta-label {{ color: {TEXT_MUTED}; font-size: .74rem; }}
.eov-action-meta-value {{ font-size: .85rem; margin-bottom: .8rem; }}

.eov-retest-card {{ border: 1px solid {BORDER}; border-radius: 10px; padding: .9rem 1rem; height: 100%; }}
.eov-retest-when {{ font-weight: 600; font-size: .92rem; margin-bottom: .25rem; }}
.eov-retest-detail {{ color: {TEXT_MUTED}; font-size: .8rem; line-height: 1.4; }}

.eov-info-box {{
    background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: .85rem 1rem; font-size: .85rem;
}}

/* ---- specialists ---- */
.eov-spec-card {{
    border: 1px solid {BORDER}; border-left: 3px solid transparent; border-radius: 8px;
    padding: .9rem 1rem; margin-bottom: .8rem;
}}
.eov-spec-card.selected {{ border-left-color: {BLUE}; }}
.eov-spec-top {{ display:flex; justify-content: space-between; }}
.eov-spec-name {{ font-weight: 600; font-size: .95rem; }}
.eov-spec-dist {{ color: {TEXT_MUTED}; font-size: .78rem; }}
.eov-spec-addr {{ color: {TEXT_MUTED}; font-size: .8rem; margin: .35rem 0; }}
.eov-spec-match {{ font-size: .8rem; margin-bottom: .3rem; }}
.eov-spec-slot {{ font-size: .8rem; color: {TEXT_MUTED}; }}

/* ---- upload empty state ---- */
.eov-upload-wrap {{
    border: 1px dashed {BORDER}; border-radius: 14px; padding: 1rem 0.66rem; text-align: center; background: #FAFAFA;
}}
.eov-upload-icon {{ font-size: 2rem; margin-bottom: .6rem; }}
.eov-upload-title {{ font-size: 1.15rem; font-weight: 600; margin-bottom: .3rem; }}
.eov-upload-sub {{ color: {TEXT_MUTED}; font-size: .87rem; margin-bottom: 1.2rem; }}

/* ---- footer ---- */
.eov-footer {{
    background: #111827; color: #D1D5DB; border-radius: 10px; padding: 1.6rem 1.8rem; margin-top: 2.5rem;
    display:flex; justify-content: space-between; align-items: flex-start; font-size: .82rem;
}}
.eov-footer b {{ color: #fff; font-size: .95rem; }}
.eov-footer .muted {{ color: #9CA3AF; margin-top: .3rem; max-width: 420px; line-height: 1.5; }}

div[data-testid="stMetricValue"] {{ font-size: 1.4rem; }}
.stExpander {{ border-radius: 10px; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

defaults = {
    "uploader_key": 0,
    "processed_file_id": None,
    "result": None,
    "specialists": None,
    "nearby_specialists": None,
    "addr": "",
    "coordinates": [],
    "location_resolved": False,
    "active_tab": "analysis",
    "specialist_filter": "All",
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


def clear_report():
    st.session_state["uploader_key"] += 1
    st.session_state["processed_file_id"] = None
    st.session_state["result"] = None
    st.session_state["specialists"] = None
    st.session_state["nearby_specialists"] = None
    st.session_state["active_tab"] = "analysis"
    st.cache_data.clear()


# ============================================================================
# CACHED BACKEND CALLS (unchanged from original)
# ============================================================================

@st.cache_data(show_spinner=False)
def run_clinical_analysis(text):
    inputs = {
        "report_text": text,
        "extracted_data": "",
        "guideline_context": "",
        "final_plan": "",
    }
    return clinical_agent.invoke(inputs)


@st.cache_data(show_spinner=False)
def get_nearby_specialists(specialists, addr, coordinates):
    return fetch_local_doctors(specialists, addr, list(coordinates))


@st.cache_data(show_spinner=False)
def reverse_geocode(lat, lng):
    geolocator = Nominatim(user_agent="clinical_referral_locator_app")
    geo_response = geolocator.reverse(f"{lat}, {lng}", timeout=10)
    return geo_response.address if geo_response else ""


# ============================================================================
# DATA ADAPTERS — normalize whatever the backend returns into the shapes
# the UI below expects. See the module docstring for the full contract.
# ============================================================================

def _extracted(result):
    ed = result.get("extracted_data") if isinstance(result, dict) else None
    if isinstance(ed, list) and ed and isinstance(ed[0], dict):
        return ed[0]
    if isinstance(ed, dict):
        return ed
    return {}


def get_narrative(result):
    return _extracted(result).get("summary", "No summary was returned for this report.")


def get_patient_line(result):
    ex = _extracted(result)
    patient = ex.get("patient", {}) if isinstance(ex.get("patient"), dict) else {}
    parts = []
    if patient.get("age"):
        sex = f"-year-old {patient['sex']}" if patient.get("sex") else "-year-old"
        parts.append(f"{patient['age']}{sex}")
    params = get_parameters(result)
    if params:
        parts.append(f"{len(params)} parameters extracted")
    parts.append("compared against WHO and guideline reference ranges")
    return " \u00b7 ".join(parts)


def get_report_date(result):
    return _extracted(result).get("report_date", "")


def get_parameters(result):
    params = _extracted(result).get("parameters")
    return params if isinstance(params, list) else []


def get_status_counts(result):
    params = get_parameters(result)
    total = len(params)
    elevated = sum(1 for p in params if p.get("status") in ("elevated", "high"))
    low = sum(1 for p in params if p.get("status") == "low")
    out_of_range = elevated + low
    in_range = total - out_of_range
    return {
        "total": total,
        "out_of_range": out_of_range,
        "in_range": in_range,
        "elevated": elevated,
        "low": low,
    }


def get_in_range_names(result, limit=4):
    params = get_parameters(result)
    names = [p["name"] for p in params if p.get("status") == "in_range" and p.get("name")]
    return ", ".join(names[:limit])


def get_next_retest(result):
    ex = _extracted(result)
    return ex.get("next_retest", "\u2014"), ex.get("next_retest_detail", "")


def get_flagged_systems(result):
    fs = _extracted(result).get("flagged_systems")
    return fs if isinstance(fs, list) else []


def get_guidelines(result):
    ctx = result.get("guideline_context") if isinstance(result, dict) else None
    if isinstance(ctx, list):
        return ctx
    if isinstance(ctx, str) and ctx.strip():
        return [{"title": "Referenced guideline context", "description": ctx[:600] + ("..." if len(ctx) > 600 else "")}]
    return []


def _final_plan(result):
    fp = result.get("final_plan") if isinstance(result, dict) else None
    return fp if isinstance(fp, dict) else {}


def get_actions(result):
    actions = _final_plan(result).get("actions")
    return actions if isinstance(actions, list) else []


def get_plan_fallback_text(result):
    return _final_plan(result).get("follow_up_plan", "No action plan was returned for this report.")


def get_retest_schedule(result):
    sched = _final_plan(result).get("retest_schedule")
    return sched if isinstance(sched, list) else []


def get_specialist_rows(nearby_specialists):
    """Best-effort normalization of whatever fetch_local_doctors() returns."""
    rows = []
    if isinstance(nearby_specialists, list):
        for i, s in enumerate(nearby_specialists, start=1):
            if not isinstance(s, dict):
                continue
            rows.append({
                "number": s.get("number", i),
                "name": s.get("name", "Specialist"),
                "specialty": s.get("specialty", ""),
                "distance": s.get("distance", ""),
                "address": s.get("address", ""),
                "matched_for": s.get("matched_for", ""),
                "next_slot": s.get("next_slot", ""),
            })
    return rows


# ============================================================================
# RENDER HELPERS
# ============================================================================

def badge_html(text, color, bg):
    return f'<span class="eov-badge" style="color:{color};background:{bg};">{text}</span>'


def status_badge(status):
    label, color, bg = STATUS_STYLES.get(status, ("\u2014", GRAY, GRAY_BG))
    return badge_html(label, color, bg)


def priority_badge(priority):
    label, color, bg = PRIORITY_STYLES.get(priority, ("Note", GRAY, GRAY_BG))
    return badge_html(label, color, bg)


def range_bar_html(value, ref_low, ref_high, status):
    vals = [v for v in [value, ref_low, ref_high] if isinstance(v, (int, float))]
    if not vals:
        return '<div class="eov-bar-track"></div>'
    hi_bound = max(vals) * 1.35 or 1
    lo_bound = 0

    def frac(x):
        if not isinstance(x, (int, float)):
            return None
        f = (x - lo_bound) / (hi_bound - lo_bound)
        return max(0.0, min(1.0, f)) * 100

    f_low = frac(ref_low)
    f_high = frac(ref_high)
    f_val = frac(value)

    if f_low is None:
        f_low = 0
    if f_high is None:
        f_high = 100

    range_style = f"left:{f_low}%; width:{max(f_high - f_low, 1)}%;"
    marker_color = GREEN if status == "in_range" else RED
    marker_style = f"left:calc({f_val}% - 1px); background:{marker_color};" if f_val is not None else "display:none;"

    return (
        '<div class="eov-bar-track">'
        f'<div class="eov-bar-range" style="{range_style}"></div>'
        f'<div class="eov-bar-marker" style="{marker_style}"></div>'
        '</div>'
    )


def render_header():
    file_name = st.session_state.get("uploaded_file_name", "")
    addr = st.session_state.get("addr", "")
    short_addr = addr.split(",")[0:2]
    short_addr = ", ".join(s.strip() for s in short_addr) if addr else ""

    right_bits = []
    if file_name:
        right_bits.append(f"\U0001F4C4 <b>{file_name}</b>")
    if short_addr:
        right_bits.append(f"<b>{short_addr}</b>")
    right_html = " &nbsp;|&nbsp; ".join(right_bits) if right_bits else ""

    st.markdown(
        f"""
        <div class="eov-header">
            <div>
                <span class="eov-logo"> EOV-PULSE</span>
                <span class="eov-logo-caption">Clinical lab report analysis</span>
            </div>
            <div class="eov-header-right">
                {right_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state["result"] is not None:
        _, col_new = st.columns([6, 1])
        with col_new:
            st.button("New report", key="new_report_btn", on_click=clear_report, use_container_width=True)


def render_title_and_actions(result):
    report_date = get_report_date(result)
    meta = f"Report analysis"
    if report_date:
        meta += f" \u00b7 {report_date}"
    st.markdown(f'<div class="eov-meta">{meta}</div>', unsafe_allow_html=True)

    col_title, col_btn1, col_btn2 = st.columns([5, 1.3, 1.3])
    with col_title:
        st.markdown('<div class="eov-title eov-serif">Health status summary</div>', unsafe_allow_html=True)
    with col_btn1:
        st.download_button(
            "\u2b07 Download summary",
            data=get_narrative(result) + "\n\n" + get_plan_fallback_text(result),
            file_name="eov_pulse_summary.txt",
            use_container_width=True,
        )
    with col_btn2:
        if st.button("Find a specialist \u2192", type="primary", use_container_width=True):
            st.session_state["active_tab"] = "specialists"
            st.rerun()

    st.markdown(f'<div class="eov-subtitle">{get_patient_line(result)}</div>', unsafe_allow_html=True)


def render_stat_cards(result):
    counts = get_status_counts(result)
    next_retest, next_retest_detail = get_next_retest(result)
    overall_ok = counts["out_of_range"] == 0
    overall_label = "Looks good" if overall_ok else "Needs attention"
    overall_dot = GREEN if overall_ok else RED

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="eov-card">
                <div class="eov-card-label">Overall status</div>
                <div class="eov-card-value">
                    <span class="eov-status-dot" style="background:{overall_dot};"></span>{overall_label}
                </div>
                <div class="eov-card-sub">{counts["out_of_range"]} of {counts["total"] or "\u2014"} parameters fall outside guideline range.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="eov-card">
                <div class="eov-card-label">Out of range</div>
                <div class="eov-card-value" style="color:{RED if counts['out_of_range'] else TEXT_MAIN};">{counts["out_of_range"]}</div>
                <div class="eov-card-sub">{counts["elevated"]} elevated \u00b7 {counts["low"]} low</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="eov-card">
                <div class="eov-card-label">Within range</div>
                <div class="eov-card-value">{counts["in_range"]}</div>
                <div class="eov-card-sub">{get_in_range_names(result) or "&mdash;"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="eov-card">
                <div class="eov-card-label">Next retest</div>
                <div class="eov-card-value">{next_retest}</div>
                <div class="eov-card-sub">{next_retest_detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")


def render_tab_bar():
    tabs = [("analysis", "Analysis"), ("action_plan", "Action plan"), ("specialists", "Nearby specialists")]
    cols = st.columns([1, 1, 1, 6])
    for (key, label), col in zip(tabs, cols):
        with col:
            btn_type = "primary" if st.session_state["active_tab"] == key else "secondary"
            if st.button(label, key=f"tab_{key}", type=btn_type, use_container_width=True):
                st.session_state["active_tab"] = key
                st.rerun()
    st.markdown(f"<div style='border-bottom:1px solid {BORDER}; margin-bottom:1.3rem;'></div>", unsafe_allow_html=True)


# ---- Analysis tab ----

def render_analysis_tab(result):
    col_summary, col_flags = st.columns([2.4, 1])
    with col_summary:
        st.markdown(
            f"""
            <div class="eov-panel">
                <div class="eov-panel-title">What the report says</div>
                <div class="eov-narrative">{get_narrative(result)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_flags:
        flagged = get_flagged_systems(result)
        rows_html = ""
        for f in flagged:
            color = GREEN if f.get("status") == "good" else RED
            rows_html += (
                f'<div class="eov-flag-row"><span>{f.get("name","")}</span>'
                f'<span style="color:{color}; font-weight:500;">{f.get("detail","")}</span></div>'
            )
        if not rows_html:
            rows_html = '<div class="eov-flag-row"><span style="color:#9CA3AF;">No flagged systems returned.</span></div>'
        st.markdown(
            f"""
            <div class="eov-panel" style="padding-top:1rem; padding-bottom: .4rem;">
                <div class="eov-panel-title">Flagged systems</div>
                {rows_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown('<div class="eov-title eov-serif" style="font-size:1.3rem;">Extracted lab values</div>', unsafe_allow_html=True)

    params = get_parameters(result)
    counts = get_status_counts(result)
    filter_choice = st.radio(
        "Filter",
        [f"All {counts['total']}", f"Out of range {counts['out_of_range']}", f"Within range {counts['in_range']}"],
        horizontal=True,
        label_visibility="collapsed",
        key="lab_filter",
    )

    if filter_choice.startswith("Out"):
        shown = [p for p in params if p.get("status") in ("elevated", "low")]
    elif filter_choice.startswith("Within"):
        shown = [p for p in params if p.get("status") == "in_range"]
    else:
        shown = params

    if not shown:
        st.info("No structured lab parameters were returned for this report yet — see `extracted_data[0]['parameters']` in the backend contract at the top of this file.")
    else:
        groups = {}
        for p in shown:
            groups.setdefault(p.get("category", "Other"), []).append(p)

        for category, rows in groups.items():
            guideline_label = rows[0].get("guideline_label", "")
            st.markdown(
                f'<div class="eov-group-header"><span>{category}</span><span class="tag">{guideline_label}</span></div>',
                unsafe_allow_html=True,
            )
            for r in rows:
                value = r.get("value")
                unit = r.get("unit", "")
                value_str = f"{value} <span style='color:{TEXT_MUTED};font-weight:400;'>{unit}</span>" if value is not None else "\u2014"
                st.markdown(
                    f"""
                    <div class="eov-row">
                        <div class="eov-row-name">{r.get("name","")}</div>
                        <div class="eov-row-value"><b>{value_str}</b></div>
                        <div class="eov-row-ref">{r.get("reference_label","")}</div>
                        <div class="eov-row-bar">{range_bar_html(value, r.get("reference_low"), r.get("reference_high"), r.get("status"))}</div>
                        <div class="eov-row-status">{status_badge(r.get("status"))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")
    guidelines = get_guidelines(result)
    with st.expander(f"Referenced guidelines ({len(guidelines)} source{'s' if len(guidelines) != 1 else ''})" if guidelines else "Referenced guidelines"):
        if guidelines:
            for g in guidelines:
                st.markdown(f"**{g.get('title','')}**")
                st.caption(g.get("description", ""))
        else:
            st.caption("No guideline context was returned for this report.")


# ---- Action plan tab ----

def render_action_plan_tab(result):
    st.markdown('<div class="eov-title eov-serif" style="font-size:1.3rem;">Personalised follow-up plan</div>', unsafe_allow_html=True)

    actions = get_actions(result)
    if not actions:
        st.markdown(f'<div class="eov-panel eov-narrative">{get_plan_fallback_text(result)}</div>', unsafe_allow_html=True)
    else:
        for a in actions:
            col_main, col_meta = st.columns([3, 1])
            
            with col_main:
                # 1. Start the single HTML string block
                card_html = f"""
                <div class="eov-action-card">
                    {priority_badge(a.get("priority"))}
                    <div class="eov-action-tag">{a.get("parameter_label","")}</div>
                    <div class="eov-action-title eov-serif">{a.get("title","")}</div>
                    <div class="eov-action-desc">{a.get("description","")}</div>
                """
                
                # 2. Loop and append the steps inside the string
                for step in a.get("steps", []) or []:
                    card_html += f'<div class="eov-action-step">\u2192 {step}</div>'
                
                # 3. Close the container string
                card_html += "</div>"
                
                # 4. Render the entire combined HTML snippet once
                st.markdown(card_html, unsafe_allow_html=True)
                
            with col_meta:
                st.markdown(
                    f"""
                    <div style="padding-top:1.3rem;">
                        <div class="eov-action-meta-label">Retest window</div>
                        <div class="eov-action-meta-value">{a.get("retest_window","\u2014")}</div>
                        <div class="eov-action-meta-label">Refer to</div>
                        <div class="eov-action-meta-value">{a.get("refer_to","\u2014")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    schedule = get_retest_schedule(result)
    if schedule:
        st.write("")
        st.markdown('<div class="eov-title eov-serif" style="font-size:1.15rem;">Retest schedule</div>', unsafe_allow_html=True)
        cols = st.columns(len(schedule))
        for col, item in zip(cols, schedule):
            with col:
                st.markdown(
                    f"""
                    <div class="eov-retest-card">
                        <div class="eov-retest-when">\U0001F535 {item.get("when","")}</div>
                        <div class="eov-retest-detail">{item.get("detail","")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")
    st.markdown(
        """
        <div class="eov-info-box">
            <b>\u2139\ufe0f Guidance, not a diagnosis</b><br>
            EOV-PULSE compares extracted values against published reference ranges.
            Interpretation and treatment decisions belong to a qualified clinician.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---- Nearby specialists tab ----

def render_specialists_tab(nearby_specialists):
    st.markdown('<div class="eov-title eov-serif" style="font-size:1.3rem;">Specialists near you</div>', unsafe_allow_html=True)
    addr_line = st.session_state.get("addr", "")
    st.markdown(
        f'<div class="eov-subtitle">Matched to your flagged parameters'
        f'{" \u00b7 near " + addr_line.split(",")[0] if addr_line else ""}</div>',
        unsafe_allow_html=True,
    )

    rows = get_specialist_rows(nearby_specialists)
    specialties = sorted({r["specialty"] for r in rows if r.get("specialty")})
    filter_options = [f"All {len(rows)}"] + specialties
    choice = st.radio("Specialty filter", filter_options, horizontal=True, label_visibility="collapsed", key="spec_filter")

    if choice.startswith("All"):
        filtered = rows
    else:
        filtered = [r for r in rows if r.get("specialty") == choice]

    # col_list, col_map = st.columns([1.1, 1.6])
    # with col_list:
    if not filtered:
        st.info("No nearby specialists were returned yet. Once `fetch_local_doctors()` / `extract_specialists()` return results, they'll be listed here.")
        # for r in filtered:
        #     st.markdown(
        #         f"""
        #         <div class="eov-spec-card">
        #             <div class="eov-spec-top">
        #                 <span class="eov-spec-name">{r["number"]}. {r["name"]}</span>
        #                 <span class="eov-spec-dist">{r["distance"]}</span>
        #             </div>
        #             {badge_html(r["specialty"], BLUE, "#DBEAFE") if r["specialty"] else ""}
        #             <div class="eov-spec-addr">{r["address"]}</div>
        #             <div class="eov-spec-match"><b>Matched for:</b> {r["matched_for"]}</div>
        #             <div class="eov-spec-slot">Next slot: {r["next_slot"]}</div>
        #         </div>
        #         """,
        #         unsafe_allow_html=True,
        #     )
    # with col_map:
    gmap(nearby_specialists)


# ---- Footer ----

def render_footer():
    st.markdown(
        """
        <div class="eov-footer">
            <div>
                <b>EOV-PULSE</b>
                <div class="muted">Clinical lab report analysis, guideline referencing, and local specialist
                routing. Reference ranges follow WHO and the guideline bodies cited in each panel.</div>
            </div>
            <div class="muted">Not a substitute for professional medical advice.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# SIDEBAR (kept from the original app, collapsed by default to match the
# mockup, which has no visible sidebar)
# ============================================================================

# with st.sidebar:
#     st.header("System Status")
#     st.success("Ollama: Connected")
#     st.success("ChromaDB: Connected")
#     st.info("Model: Llama 3 (Reasoning)")

# ============================================================================
# MAIN FLOW
# ============================================================================

render_header()

if st.session_state["result"] is None:
    # ---- Empty state: location + upload ----
    st.markdown('<div class="eov-title eov-serif">Health status summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="eov-subtitle">Upload a lab report to get a guideline-referenced summary, '
        'a follow-up plan, and nearby specialists matched to your results.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state["location_resolved"]:
        location = streamlit_geolocation()
        if location and location.get("latitude") and location.get("longitude"):
            lat, lng = location["latitude"], location["longitude"]
            st.session_state["coordinates"] = [lat, lng]
            try:
                st.session_state["addr"] = reverse_geocode(lat, lng)
                st.session_state["location_resolved"] = True
            except Exception as e:
                st.error(f"Geocoding service unavailable: {e}")
        elif location =={}:
            st.info("\U0001F4A1 Click the location button above to fetch coordinates for nearby specialist matching.")
        else:
            st.warning("\u26a0\ufe0f Location access denied or unavailable. Enable browser location permissions to see nearby specialists.")
    else:
        st.caption(f"\U0001F4CD Location set: {st.session_state['addr'][:80]}...")

    st.write("")
    st.markdown(
        """
        <div class="eov-upload-wrap">
            <div class="eov-upload-title">\U0001F4C4 Upload patient lab report</div>
            <div class="eov-upload-sub">PDF format \u00b7 analyzed against WHO and specialty-society guidelines</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload Patient Lab Report (PDF)",
        type="pdf",
        key=f"pdf_uploader_{st.session_state['uploader_key']}",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        st.session_state["uploaded_file_name"] = uploaded_file.name
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        if st.session_state["processed_file_id"] != file_id:
            with st.spinner("Processing report and consulting WHO guidelines..."):
                raw_text = extract_tables_and_text(uploaded_file)
                result = run_clinical_analysis(raw_text)
                specialists = extract_specialists(result.get("final_plan", {}))
                print("specialist is ::::::::::", specialists)
                nearby_specialists = get_nearby_specialists(
                    specialists, st.session_state["addr"], st.session_state["coordinates"]
                )

            st.session_state["result"] = result
            st.session_state["specialists"] = specialists
            st.session_state["nearby_specialists"] = nearby_specialists
            st.session_state["processed_file_id"] = file_id
            st.rerun()

    render_footer()

else:
    # ---- Results state ----
    result = st.session_state["result"]
    nearby_specialists = st.session_state["nearby_specialists"]

    render_title_and_actions(result)
    render_stat_cards(result)
    render_tab_bar()

    if st.session_state["active_tab"] == "analysis":
        render_analysis_tab(result)
    elif st.session_state["active_tab"] == "action_plan":
        render_action_plan_tab(result)
    else:
        render_specialists_tab(nearby_specialists)

    render_footer()