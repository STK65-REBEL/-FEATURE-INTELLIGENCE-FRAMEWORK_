"""
Feature Intelligence Framework — Python / Streamlit Edition
=============================================================
A fully customizable vehicle feature & cost intelligence tool.

Run it with:
    pip install streamlit pandas openpyxl plotly
    streamlit run feature_intelligence_app.py

Everything is customizable at runtime — no fixed constants:
  - Categories: add, remove, rename, re-weight, re-multiply, re-color
  - Currency symbol / price unit label
  - Feature types (Binary / Tier / Numeric / Categorical) — same 4-type
    scoring model as the JS version, all flowing through one formula
  - Data: upload the standard Excel format, add a vehicle by form, or
    paste a "Feature: value" quick-add block

It reads the same Excel schema already established (Feature, Type,
Levels/Ceiling, Notes, Example Vehicle, then one column per vehicle;
Price/Class special rows; Category/Subgroup header rows with blank
vehicle cells) — so `indian-market-feature-database.xlsx`, if present
in the same folder, loads automatically as the starting dataset.

Local persistence: session state auto-saves to fif_data.json next to
this script, so closing and reopening the app picks up where you left
off (this is real cross-session persistence — unlike the browser
artifact versions, which reset because window.storage isn't available
outside Claude's own environment).
"""

import json
import io
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ============================================================================
# CONSTANTS THAT ARE **NOT** CUSTOMIZABLE (structural, not business logic)
# ============================================================================
FEATURE_TYPES = ["Binary", "Tier", "Numeric", "Categorical"]
POSITIVE, NEGATIVE, ACCENT = "#16A34A", "#DC2626", "#1D4ED8"
CLASS_COLORS = ["#1D4ED8", "#DC2626", "#16A34A", "#F59E0B", "#7C3AED", "#0EA5E9", "#DB2777"]
TYPE_COLOR = {"Binary": "#667085", "Tier": "#7C3AED", "Numeric": "#0891B2", "Categorical": "#B45309"}

DATA_FILE = Path(__file__).parent / "fif_data.json"
EXCEL_DEFAULT = Path(__file__).parent / "indian-market-feature-database.xlsx"

DEFAULT_CATEGORY_CONFIG = [
    {"id": 0, "name": "Safety", "weight": 0.40, "multiplier": 1.5, "color": "#DC2626"},
    {"id": 1, "name": "Comfort", "weight": 0.25, "multiplier": 1.2, "color": "#F59E0B"},
    {"id": 2, "name": "Technology", "weight": 0.25, "multiplier": 2.0, "color": "#0EA5E9"},
    {"id": 3, "name": "Utility", "weight": 0.10, "multiplier": 1.0, "color": "#16A34A"},
]


FALLBACK_ROWS = [  # used only if no Excel and no saved session exist
    {"category": "Safety", "subgroup": "Active Safety", "feature": "ABS with EBD", "type": "Binary", "levels": None, "ceiling": None,
     "values": {"Model A": 1, "Model B": 1}},
    {"category": "Safety", "subgroup": "Active Safety", "feature": "ADAS Level", "type": "Tier",
     "levels": ["No ADAS", "Level 1", "Level 2", "Level 2+"], "ceiling": None,
     "values": {"Model A": 0, "Model B": 2}},
    {"category": "Technology", "subgroup": "Core Infotainment", "feature": "Boot Space (L)", "type": "Numeric", "levels": None, "ceiling": None,
     "values": {"Model A": 265, "Model B": 373}},
]
FALLBACK_PRICE = {"Model A": 5.5, "Model B": 8.0}
FALLBACK_CLASS = {"Model A": "Hatchback", "Model B": "Midsize SUV"}


# ============================================================================
# SCORING ENGINE — identical formula to the JS versions:
#   contribution = fraction_present(0..1) x category_multiplier
#   Final Score  = sum( weight[c] x category_score[c] )
# every feature Type reduces to the same 0..multiplier contribution.
# ============================================================================
def contribution_for(row, vehicle, category_by_name, numeric_stats, numeric_mode):
    cfg = category_by_name.get(row["category"])
    mult = cfg["multiplier"] if cfg else 1.0
    val = row["values"].get(vehicle)
    if row["type"] == "Categorical":
        return 0.0
    if row["type"] == "Binary":
        return (1.0 if val else 0.0) * mult
    if row["type"] == "Tier":
        levels = row["levels"] or ["Absent", "Present"]
        max_idx = len(levels) - 1
        idx = val if isinstance(val, (int, float)) else 0
        frac = (idx / max_idx) if max_idx > 0 else 0.0
        return frac * mult
    if row["type"] == "Numeric":
        is_missing = val in (None, "", "nan")
        if is_missing:
            return 0.0  # unresearched — no credit, and never compared against a scale it isn't part of
        num = float(val)
        if numeric_mode == "fixed" and row.get("ceiling"):
            frac = max(0.0, min(1.0, num / row["ceiling"]))
        else:
            stat = numeric_stats.get(row["feature"], {"min": 0, "max": 0})
            if stat["max"] > stat["min"]:
                frac = (num - stat["min"]) / (stat["max"] - stat["min"])
            else:
                frac = 1.0 if stat["max"] > 0 else 0.0
            frac = max(0.0, min(1.0, frac))  # defensive clamp — a contribution can never be negative or exceed the multiplier
        return frac * mult
    return 0.0


def compute_numeric_stats(vehicles, rows):
    """Single source of truth for Numeric-feature min/max across the current vehicle set.
    Every place that scores a Numeric feature outside the main compute_scores() loop
    MUST use this — passing {} silently makes every Numeric feature contribute 0
    (this was bug #1 in the Aug 2026 QA pass: 9 call sites had this exact mistake).

    A genuinely unresearched cell (None/""/"nan") is EXCLUDED from the min/max range,
    not silently counted as a real 0 — otherwise one unresearched vehicle drags the
    whole scale down and distorts every other vehicle's normalized score too. The
    unresearched vehicle itself still scores 0 for that feature (we can't credit what
    we don't know), but it no longer corrupts the scale for everyone else."""
    numeric_stats = {}
    for r in rows:
        if r["type"] == "Numeric":
            vals = [float(r["values"].get(v)) for v in vehicles if r["values"].get(v) not in (None, "", "nan")]
            numeric_stats[r["feature"]] = {"min": min(vals), "max": max(vals)} if vals else {"min": 0, "max": 0}
    return numeric_stats


def compute_scores(vehicles, rows, category_config, relevance, numeric_mode, numeric_stats=None):
    category_by_name = {c["name"]: c for c in category_config}
    cat_names = [c["name"] for c in category_config]

    if numeric_stats is None:
        numeric_stats = compute_numeric_stats(vehicles, rows)

    scorable = [r for r in rows if r["type"] != "Categorical" and r["category"] in category_by_name]
    max_category_raw = {
        c: sum(1 for r in scorable if r["category"] == c) * category_by_name[c]["multiplier"]
        for c in cat_names
    }

    per_vehicle = {}
    for v in vehicles:
        category_scores = {c: 0.0 for c in cat_names}
        for r in scorable:
            category_scores[r["category"]] += contribution_for(r, v, category_by_name, numeric_stats, numeric_mode)

        customer_category_scores = {c: category_scores[c] * relevance.get(c, 1.0) for c in cat_names}
        weighted = {c: category_by_name[c]["weight"] * category_scores[c] for c in cat_names}
        weighted_customer = {c: category_by_name[c]["weight"] * customer_category_scores[c] for c in cat_names}

        engineering_final = sum(weighted.values())
        customer_final = sum(weighted_customer.values())
        feature_count = sum(1 for r in scorable if contribution_for(r, v, category_by_name, numeric_stats, numeric_mode) > 0)

        normalized = {c: (category_scores[c] / max_category_raw[c] * 10 if max_category_raw[c] > 0 else 0.0) for c in cat_names}
        feature_rating = (sum(normalized.values()) / len(cat_names)) if cat_names else 0.0

        per_vehicle[v] = {
            "category_scores": category_scores, "customer_category_scores": customer_category_scores,
            "engineering_final": engineering_final, "customer_final": customer_final,
            "feature_count": feature_count, "normalized_category": normalized, "feature_rating": feature_rating,
        }
    return per_vehicle


def display_value(row, vehicle):
    val = row["values"].get(vehicle)
    if row["type"] == "Tier":
        levels = row["levels"] or ["Absent", "Present"]
        idx = val if isinstance(val, (int, float)) else 0
        return levels[int(idx)] if 0 <= int(idx) < len(levels) else "—"
    if row["type"] == "Binary":
        return "Yes" if val else "No"
    if row["type"] == "Numeric":
        return "" if val in (None, "") else str(val)
    return val or "—"


def percentile_rank(value, pool):
    if len(pool) <= 1:
        return 100
    below = sum(1 for x in pool if x < value)
    return round(below / (len(pool) - 1) * 100)


def tier_label(score):
    if score >= 8: return "Strong", "#16A34A", "#E7F6EC"
    if score >= 6: return "Moderate", "#0EA5E9", "#E8F6FD"
    if score >= 4: return "Below Avg", "#F59E0B", "#FFF6E5"
    return "Weak", "#DC2626", "#FDECEC"


def tier_label_relative(score, distribution):
    """Classifies a score against the CURRENT dataset's actual spread (quartiles),
    not a fixed absolute scale. With real-world research still being built out,
    fixed 8/6/4 cutoffs made every single vehicle read as "Weak" — even the
    best-researched one — because nothing in the dataset yet reaches that absolute
    bar. Quartile-based tiers stay meaningful today and keep recalibrating
    automatically as more verified data gets added, instead of needing a manual
    threshold change later."""
    dist = [d for d in distribution if d is not None]
    if len(set(dist)) <= 1:
        return "Moderate", "#0EA5E9", "#E8F6FD"
    sorted_d = sorted(dist)
    n = len(sorted_d)

    def pct(p):
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return sorted_d[idx]

    if score >= pct(0.75): return "Strong", "#16A34A", "#E7F6EC"
    if score >= pct(0.50): return "Moderate", "#0EA5E9", "#E8F6FD"
    if score >= pct(0.25): return "Below Avg", "#F59E0B", "#FFF6E5"
    return "Weak", "#DC2626", "#FDECEC"


def build_feature_gap_pdf(base_name, competitor_names, category_rows, feature_gap_rows, money_fn):
    """Builds a clean, presentable Feature Gap PDF for one base vehicle vs one or more
    competitors. Returns bytes, or None if reportlab isn't installed."""
    if not REPORTLAB_AVAILABLE:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm, bottomMargin=16 * mm, leftMargin=16 * mm, rightMargin=16 * mm)
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#0B2559")
    accent = colors.HexColor("#1D4ED8")
    muted = colors.HexColor("#667085")
    title_style = ParagraphStyle("TitleNavy", parent=styles["Title"], textColor=navy, fontSize=20, spaceAfter=2)
    sub_style = ParagraphStyle("SubMuted", parent=styles["Normal"], textColor=muted, fontSize=9.5, spaceAfter=14)
    h2_style = ParagraphStyle("H2Navy", parent=styles["Heading2"], textColor=navy, fontSize=13, spaceBefore=14, spaceAfter=6)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"], textColor=muted, fontSize=8, spaceBefore=6)

    story = [
        Paragraph("Feature Gap Report", title_style),
        Paragraph(f"Base: <b>{base_name}</b> &nbsp;|&nbsp; vs: {', '.join(competitor_names)} &nbsp;|&nbsp; "
                  f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}", sub_style),
        Paragraph("Category-Level Gap", h2_style),
    ]

    cat_header = ["Category", "Gap vs Base"]
    cat_data = [cat_header] + [[c, f"{g:+.2f}"] for c, g in category_rows]
    cat_table = Table(cat_data, colWidths=[100 * mm, 40 * mm])
    cat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE3EC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F9")]),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(cat_table)

    story.append(Paragraph("Feature-Level Gap — Ranked by Priority", h2_style))
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7.8, leading=9.5)
    feat_header = ["Category", "Feature", base_name, "Penetration", "Verdict"]
    feat_data = [feat_header]
    for r in feature_gap_rows:
        feat_data.append([
            Paragraph(str(r["Category"]), cell_style),
            Paragraph(str(r["Feature"]), cell_style),
            Paragraph(str(r[base_name]), cell_style),
            f"{r['Penetration %']}%",
            r["Verdict"],
        ])
    feat_table = Table(feat_data, colWidths=[24 * mm, 54 * mm, 38 * mm, 22 * mm, 24 * mm], repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE3EC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    for i, r in enumerate(feature_gap_rows, start=1):
        if r["Verdict"] == "Base behind":
            style_cmds.append(("TEXTCOLOR", (4, i), (4, i), colors.HexColor("#DC2626")))
        elif r["Verdict"] == "Base ahead":
            style_cmds.append(("TEXTCOLOR", (4, i), (4, i), colors.HexColor("#16A34A")))
    feat_table.setStyle(TableStyle(style_cmds))
    story.append(feat_table)
    story.append(Paragraph("Priority Score = (competitor avg − base) × competitor penetration × category weight. "
                            "Table sorted highest priority first — the top rows are the features most worth adding.", caption_style))

    doc.build(story)
    return buf.getvalue()


def _pdf_table(headers, rows, col_widths, navy, header_bg=None):
    """Shared table styling for the report PDFs — consistent look across sections.
    Text cells are wrapped in Paragraph objects, not plain strings — plain strings
    in reportlab Table cells don't wrap and silently overflow into the next
    column (the exact bug found and fixed in build_feature_gap_pdf earlier;
    this helper was written without that fix and needed the same one)."""
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle("PdfTableCell", parent=styles["Normal"], fontSize=8, leading=9.5)
    data = [headers]
    for r in rows:
        data.append([Paragraph(str(c), cell_style) if isinstance(c, str) and len(c) > 12 else str(c) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg or navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE3EC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def build_summary_pdf(vehicles, rep_table, top_priority_rows, best_v, worst_v, base_name, money_fn):
    """A condensed, one-to-two-page executive summary — headline KPIs, ranked scores,
    and the top priority features, nothing else."""
    if not REPORTLAB_AVAILABLE:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm, bottomMargin=16 * mm, leftMargin=16 * mm, rightMargin=16 * mm)
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#0B2559")
    muted = colors.HexColor("#667085")
    title_style = ParagraphStyle("T", parent=styles["Title"], textColor=navy, fontSize=20, spaceAfter=2)
    sub_style = ParagraphStyle("S", parent=styles["Normal"], textColor=muted, fontSize=9.5, spaceAfter=14)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=navy, fontSize=13, spaceBefore=12, spaceAfter=6)

    story = [
        Paragraph("Feature Intelligence Framework — Executive Summary", title_style),
        Paragraph(f"{len(vehicles)} vehicles in scope &nbsp;|&nbsp; Base: <b>{base_name}</b> &nbsp;|&nbsp; Generated {datetime.now().strftime('%d %b %Y, %H:%M')}", sub_style),
        Paragraph("Headline", h2),
    ]
    kpi_rows = [["Best in Scope", best_v or "—"], ["Worst in Scope", worst_v or "—"], ["Base Vehicle", base_name or "—"]]
    story.append(_pdf_table(["Metric", "Vehicle"], kpi_rows, [90 * mm, 90 * mm], navy))
    story.append(Paragraph("Final Score Ranking", h2))
    score_headers = list(rep_table[0].keys()) if rep_table else []
    score_rows = [[r[h] for h in score_headers] for r in rep_table]
    col_w = [180 * mm / max(1, len(score_headers))] * len(score_headers)
    story.append(_pdf_table(score_headers, score_rows, col_w, navy))
    if top_priority_rows:
        story.append(Paragraph(f"Top Priority Features to Add to {base_name}", h2))
        ph = ["Feature", "Category", "Priority", "Penetration %"]
        prows = [[r["Feature"], r["Category"], r["Priority"], r["Penetration %"]] for r in top_priority_rows]
        story.append(_pdf_table(ph, prows, [70 * mm, 40 * mm, 35 * mm, 35 * mm], navy))
    doc.build(story)
    return buf.getvalue()


def build_full_pdf(rep_table, matrix_table, vehicles, rows, base_name, money_fn):
    """The comprehensive version — final scores, the full score matrix, and the
    complete feature-by-feature audit, one page-friendly table PER VEHICLE.

    A single wide table with 15-20+ vehicle columns is unreadable on a portrait
    page — headers get compressed into a few unreadable characters each,
    regardless of font size. A per-vehicle table (Feature / Type / Value, 3
    columns) is the only version that stays legible at any vehicle count; it
    runs longer (more pages) by design, which is the right tradeoff for a
    document meant to be read, not just generated. The Excel export remains
    the right tool for a wide, all-vehicles-as-columns view."""
    if not REPORTLAB_AVAILABLE:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=16 * mm, bottomMargin=14 * mm, leftMargin=14 * mm, rightMargin=14 * mm)
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#0B2559")
    muted = colors.HexColor("#667085")
    title_style = ParagraphStyle("T", parent=styles["Title"], textColor=navy, fontSize=20, spaceAfter=2)
    sub_style = ParagraphStyle("S", parent=styles["Normal"], textColor=muted, fontSize=9.5, spaceAfter=14)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=navy, fontSize=13, spaceBefore=14, spaceAfter=6)
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=8, leading=9.5)

    story = [
        Paragraph("Feature Intelligence Framework — Full Report", title_style),
        Paragraph(f"Base: <b>{base_name}</b> &nbsp;|&nbsp; Generated {datetime.now().strftime('%d %b %Y, %H:%M')} &nbsp;|&nbsp; "
                  f"{len(rows)} features audited across {len(vehicles)} vehicles", sub_style),
        Paragraph("Final Score & Category Breakdown", h2),
    ]
    if rep_table:
        headers = list(rep_table[0].keys())
        trows = [[r[h] for h in headers] for r in rep_table]
        story.append(_pdf_table(headers, trows, [190 * mm / max(1, len(headers))] * len(headers), navy))

    story.append(Paragraph("Score Matrix", h2))
    if matrix_table:
        headers = list(matrix_table[0].keys())
        trows = [[r[h] for h in headers] for r in matrix_table]
        story.append(_pdf_table(headers, trows, [190 * mm / max(1, len(headers))] * len(headers), navy))

    for v in vehicles:
        story.append(PageBreak())
        story.append(Paragraph(f"Full Feature Audit — {v}", h2))
        vdata = [["Category", "Feature", "Type", "Value"]]
        for r in rows:
            vdata.append([
                Paragraph(r["category"], cell_style), Paragraph(r["feature"], cell_style),
                r["type"], Paragraph(str(display_value(r, v)), cell_style),
            ])
        vtable = Table(vdata, colWidths=[32 * mm, 78 * mm, 20 * mm, 60 * mm], repeatRows=1)
        vtable.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 8), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE3EC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F9")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(vtable)

    doc.build(story)
    return buf.getvalue()


# ============================================================================
# EXCEL / JSON LOADING & SAVING — same schema as the standardized workbook
# ============================================================================
KNOWN_TAXONOMY_CATEGORIES = {
    "Safety", "Comfort", "Technology", "Utility",
    "Powertrain & Performance", "Dimensions & Weight", "Environmental & Compliance", "Ownership & Cost",
}


def parse_excel_bytes(file_bytes, known_category_names=None):
    """Parses the standard Feature Matrix sheet layout into rows/vehicles/price/class."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet_name = "Feature Matrix" if "Feature Matrix" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
    cols = list(df.columns)
    # Expect: Feature, Type, Levels/Ceiling, [Notes], [Example Vehicle], then vehicle columns
    meta_cols = 3
    for extra in ("Notes", "Example Vehicle"):
        if extra in cols[:6]:
            meta_cols += 1
    vehicle_cols = cols[meta_cols:]
    vehicles = [str(v).strip() for v in vehicle_cols if str(v).strip() and "unnamed" not in str(v).lower()]
    vehicle_cols = vehicle_cols[: len(vehicles)]

    rows, price_row, class_row = [], {}, {}
    cur_cat, cur_sub = None, None
    for _, r in df.iterrows():
        feature = str(r[cols[0]]).strip() if pd.notna(r[cols[0]]) else ""
        if not feature or feature.lower() == "nan":
            continue
        lower = feature.lower()
        vals_raw = [r[c] for c in vehicle_cols]
        if lower == "price":
            for v, val in zip(vehicles, vals_raw):
                if pd.notna(val):
                    try: price_row[v] = float(val)
                    except Exception: pass
            continue
        if lower in ("class", "segment"):
            for v, val in zip(vehicles, vals_raw):
                if pd.notna(val):
                    class_row[v] = str(val).strip()
            continue
        type_raw = str(r[cols[1]]).strip() if pd.notna(r[cols[1]]) else ""
        meta_raw = str(r[cols[2]]).strip() if pd.notna(r[cols[2]]) else ""
        is_header = (not type_raw) and (not meta_raw) and all(pd.isna(v) or str(v).strip() == "" for v in vals_raw)
        if is_header:
            all_known = KNOWN_TAXONOMY_CATEGORIES | (set(known_category_names) if known_category_names else set())
            if feature in all_known or feature.lower() in ("safety", "comfort", "technology", "utility"):
                cur_cat, cur_sub = feature, None
            else:
                cur_sub = feature
            continue
        if cur_cat is None:
            continue
        ftype = type_raw if type_raw in FEATURE_TYPES else "Binary"
        levels, ceiling = None, None
        if ftype == "Tier":
            levels = [s.strip() for s in meta_raw.split("|") if s.strip()] or ["Absent", "Present"]
        if ftype == "Numeric" and meta_raw:
            try: ceiling = float(meta_raw)
            except Exception: ceiling = None

        values = {}
        for v, val in zip(vehicles, vals_raw):
            if ftype == "Binary":
                values[v] = 1 if (pd.notna(val) and float(val) >= 1) else 0
            elif ftype == "Tier":
                if pd.isna(val) or str(val).strip() == "":
                    values[v] = 0
                else:
                    try:
                        values[v] = max(0, min(len(levels) - 1, round(float(val))))
                    except Exception:
                        s = str(val).strip().lower()
                        idx = next((i for i, l in enumerate(levels) if l.lower() == s), 0)
                        values[v] = idx
            elif ftype == "Numeric":
                if pd.isna(val) or str(val).strip() == "":
                    values[v] = ""  # genuinely unresearched — NOT the same as a real 0
                else:
                    try: values[v] = float(val)
                    except Exception: values[v] = ""
            else:
                values[v] = str(val).strip() if pd.notna(val) else ""

        rows.append({"category": cur_cat, "subgroup": cur_sub or cur_cat, "feature": feature,
                      "type": ftype, "levels": levels, "ceiling": ceiling, "values": values})

    return vehicles, rows, price_row, class_row


def rows_to_dataframe(vehicles, rows):
    """Flat editable view for st.data_editor — one row per feature, one column per vehicle."""
    records = []
    for r in rows:
        rec = {"Category": r["category"], "Subgroup": r["subgroup"], "Feature": r["feature"],
               "Type": r["type"], "Levels / Ceiling": ("|".join(r["levels"]) if r["levels"] else (r["ceiling"] or ""))}
        for v in vehicles:
            rec[v] = r["values"].get(v, "")
        records.append(rec)
    return pd.DataFrame(records)


def dataframe_to_rows(df, vehicles):
    rows = []
    for _, r in df.iterrows():
        feature = str(r.get("Feature", "")).strip()
        if not feature:
            continue
        ftype = str(r.get("Type", "Binary")).strip() or "Binary"
        if ftype not in FEATURE_TYPES:
            ftype = "Binary"
        meta = str(r.get("Levels / Ceiling", "") or "")
        levels, ceiling = None, None
        if ftype == "Tier":
            levels = [s.strip() for s in meta.split("|") if s.strip()] or ["Absent", "Present"]
        if ftype == "Numeric" and meta:
            try: ceiling = float(meta)
            except Exception: ceiling = None
        values = {}
        for v in vehicles:
            raw = r.get(v, "")
            if ftype == "Binary":
                values[v] = 1 if str(raw) in ("1", "1.0", "True", "true") else 0
            elif ftype == "Tier":
                try: values[v] = int(float(raw)) if str(raw).strip() != "" else 0
                except Exception: values[v] = 0
            elif ftype == "Numeric":
                try: values[v] = float(raw) if str(raw).strip() != "" else ""
                except Exception: values[v] = ""
            else:
                values[v] = str(raw) if raw is not None else ""
        rows.append({"category": str(r.get("Category", "")).strip(), "subgroup": str(r.get("Subgroup", "")).strip() or str(r.get("Category", "")).strip(),
                      "feature": feature, "type": ftype, "levels": levels, "ceiling": ceiling, "values": values})
    return rows


def save_session_to_disk():
    payload = {
        "vehicles": st.session_state.vehicles, "rows": st.session_state.rows,
        "price": st.session_state.price, "class_": st.session_state.class_,
        "category_config": st.session_state.category_config,
        "currency_symbol": st.session_state.currency_symbol, "price_unit": st.session_state.price_unit,
        "variants": st.session_state.get("variants", []), "next_variant_id": st.session_state.get("next_variant_id", 0),
    }
    try:
        DATA_FILE.write_text(json.dumps(payload, indent=1))
        return True
    except Exception:
        return False


def load_session_from_disk():
    if not DATA_FILE.exists():
        return None
    try:
        return json.loads(DATA_FILE.read_text())
    except Exception:
        return None


# ============================================================================
# APP STATE INITIALIZATION
# ============================================================================
st.set_page_config(page_title="Feature Intelligence Framework", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+Pro:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

/* ---- App background & top padding ---- */
.stApp { background: #F4F6F9; }
.main .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }

/* ---- Headline treatment: bold condensed sans, consulting-deck style ---- */
h1 { font-weight: 800 !important; letter-spacing: -0.02em; color: #0B2559 !important; font-size: 2.1rem !important; }
h2, h3 { font-weight: 700 !important; color: #0B2559 !important; }

/* ---- Sidebar: dark navy, distinct from content ---- */
[data-testid="stSidebar"] {
  background: #0B2559;
}
/* Text that sits directly on the dark background: headings, labels, captions, plain markdown */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #FFFFFF !important; font-weight: 800 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #AFC2E6 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #E7ECF5; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color: #AFC2E6 !important; font-weight: 600 !important; font-size: 0.78rem !important;
  text-transform: uppercase; letter-spacing: 0.03em;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p { color: #E7ECF5 !important; }

/* Widgets with their own white/light background need DARK text — do not force light color here */
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #1F2A37 !important; }
[data-testid="stSidebar"] input { color: #1F2A37 !important; }
[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: #FFFFFF; border-radius: 8px; padding: 10px 14px; border: 1px solid rgba(255,255,255,0.12);
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #0B2559 !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #667085 !important; }

[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }
[data-testid="stSidebar"] .stButton button { background: #1D4ED8; color: #FFFFFF; border: none; font-weight: 700; }
[data-testid="stSidebar"] .stButton button:hover { background: #2563EB; }

/* ---- KPI / metric cards: give them the card treatment McKinsey decks use ---- */
[data-testid="stMetric"] {
  background: #FFFFFF;
  border: 1px solid #DCE3EC;
  border-radius: 10px;
  padding: 16px 18px 12px 18px;
  box-shadow: 0 1px 3px rgba(11,37,89,0.06);
}
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700 !important; color: #667085 !important; }
[data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 800 !important; color: #0B2559 !important; }

/* ---- Tabs: clean underline, no boxy default look ---- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #DCE3EC; background: transparent; }
.stTabs [data-baseweb="tab"] {
  background: transparent; border: none; color: #667085; font-weight: 600; font-size: 0.85rem;
  padding: 10px 14px; border-radius: 0;
}
.stTabs [aria-selected="true"] { color: #1D4ED8 !important; border-bottom: 3px solid #1D4ED8 !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }

/* ---- Panels: give bordered sections a card feel ---- */
[data-testid="stExpander"], .stDataFrame, [data-testid="stForm"] {
  border-radius: 10px !important;
  border: 1px solid #DCE3EC !important;
}
[data-testid="stForm"] { background: #FFFFFF; padding: 1.2rem; box-shadow: 0 1px 3px rgba(11,37,89,0.05); }

/* ---- Buttons in main area ---- */
.main .stButton button {
  border-radius: 6px; font-weight: 700; border: 1px solid #DCE3EC; color: #0B2559;
}
.main .stButton button[kind="primary"], .main .stButton button:has(p) { }
.main .stButton button:hover { border-color: #1D4ED8; color: #1D4ED8; }

/* ---- Section captions ---- */
.stCaption, [data-testid="stCaptionContainer"] { color: #667085 !important; }

/* ---- Dataframe header row ---- */
.stDataFrame thead tr th { background: #EEF3FA !important; color: #667085 !important; font-weight: 700 !important; text-transform: uppercase; font-size: 0.72rem !important; letter-spacing: 0.03em; }

/* ---- Divider lines lighter ---- */
hr { border-color: #DCE3EC !important; }

/* ================= HIGH-END POLISH PASS ================= */

/* Subtle branded title bar — gradient wash, not a flat block */
.main h1 {
  background: linear-gradient(135deg, #0B2559 0%, #13316B 60%, #1D4ED8 140%);
  color: #FFFFFF !important;
  padding: 22px 28px 18px 28px;
  border-radius: 14px;
  margin-bottom: 4px;
  box-shadow: 0 8px 24px rgba(11,37,89,0.18);
}

/* Layered, softer shadows on every card surface — reads as elevation, not a flat border */
[data-testid="stMetric"] {
  box-shadow: 0 1px 2px rgba(11,37,89,0.04), 0 6px 16px rgba(11,37,89,0.07);
  transition: box-shadow 0.15s ease, transform 0.15s ease;
}
[data-testid="stMetric"]:hover { box-shadow: 0 2px 4px rgba(11,37,89,0.06), 0 10px 24px rgba(11,37,89,0.11); transform: translateY(-1px); }

/* Subheaders get a touch of hierarchy without a forbidden accent stripe */
h3 { letter-spacing: -0.01em; margin-top: 0.4rem !important; }

/* Tabs: subtle icon-style bullet + smoother hover */
.stTabs [data-baseweb="tab"] { transition: color 0.12s ease; }
.stTabs [data-baseweb="tab"]:hover { color: #1D4ED8; }

/* Buttons: soft depth + smoother press feel */
.main .stButton button {
  box-shadow: 0 1px 2px rgba(11,37,89,0.05);
  transition: all 0.12s ease;
}
.main .stButton button:hover { box-shadow: 0 3px 10px rgba(29,78,216,0.15); transform: translateY(-1px); }
.main .stButton button:active { transform: translateY(0); box-shadow: none; }

/* Primary-looking buttons (Add / Apply / Download) get real weight */
.main .stButton button p { font-weight: 700; }
.main .stDownloadButton button {
  background: #1D4ED8; color: #FFFFFF; border: none; font-weight: 700;
  box-shadow: 0 2px 8px rgba(29,78,216,0.22);
}
.main .stDownloadButton button:hover { background: #2563EB; box-shadow: 0 4px 14px rgba(29,78,216,0.3); }

/* Dataframes: crisper edges, subtle elevation */
.stDataFrame { box-shadow: 0 1px 2px rgba(11,37,89,0.04), 0 4px 12px rgba(11,37,89,0.05); }

/* Plotly charts: give them the same card frame as everything else */
[data-testid="stPlotlyChart"] {
  background: #FFFFFF; border: 1px solid #DCE3EC; border-radius: 10px; padding: 8px;
  box-shadow: 0 1px 2px rgba(11,37,89,0.04), 0 4px 12px rgba(11,37,89,0.05);
}

/* Tighter, more deliberate vertical rhythm between blocks */
.main .block-container [data-testid="stVerticalBlock"] > div { margin-bottom: 2px; }

/* Selectbox/input focus states in a brand-consistent blue, not browser default */
[data-baseweb="select"]:focus-within, input:focus { outline: 2px solid #1D4ED8 !important; outline-offset: 1px; }

/* Form containers: crisper, more deliberate card */
[data-testid="stForm"] { border-radius: 12px !important; box-shadow: 0 1px 2px rgba(11,37,89,0.04), 0 6px 16px rgba(11,37,89,0.06); }
</style>
""", unsafe_allow_html=True)

def ensure_category_ids(category_config):
    """Migrates older saved configs (no 'id' field) and returns the next free id."""
    next_id = 0
    for c in category_config:
        if "id" not in c:
            c["id"] = next_id
        next_id = max(next_id, c["id"] + 1)
    return category_config, next_id


if "initialized" not in st.session_state:
    saved = load_session_from_disk()
    if saved:
        st.session_state.vehicles = saved["vehicles"]
        st.session_state.rows = saved["rows"]
        st.session_state.price = saved["price"]
        st.session_state.class_ = saved["class_"]
        st.session_state.category_config, st.session_state.next_cat_id = ensure_category_ids(saved["category_config"])
        st.session_state.currency_symbol = saved.get("currency_symbol", "₹")
        st.session_state.price_unit = saved.get("price_unit", "Lakh")
        st.session_state.variants = saved.get("variants", [])
        st.session_state.next_variant_id = saved.get("next_variant_id", 0)
        st.session_state.data_source = "Saved session (fif_data.json)"
    elif EXCEL_DEFAULT.exists():
        vehicles, rows, price_row, class_row = parse_excel_bytes(EXCEL_DEFAULT.read_bytes())
        st.session_state.vehicles = vehicles
        st.session_state.rows = rows
        st.session_state.price = price_row
        st.session_state.class_ = class_row
        st.session_state.category_config, st.session_state.next_cat_id = ensure_category_ids([dict(c) for c in DEFAULT_CATEGORY_CONFIG])
        st.session_state.currency_symbol = "₹"
        st.session_state.price_unit = "Lakh"
        st.session_state.variants = []
        st.session_state.next_variant_id = 0
        st.session_state.data_source = f"Loaded from {EXCEL_DEFAULT.name}"
    else:
        st.session_state.vehicles = list(FALLBACK_PRICE.keys())
        st.session_state.rows = FALLBACK_ROWS
        st.session_state.price = FALLBACK_PRICE
        st.session_state.class_ = FALLBACK_CLASS
        st.session_state.category_config, st.session_state.next_cat_id = ensure_category_ids([dict(c) for c in DEFAULT_CATEGORY_CONFIG])
        st.session_state.currency_symbol = "₹"
        st.session_state.price_unit = "Lakh"
        st.session_state.variants = []
        st.session_state.next_variant_id = 0
        st.session_state.data_source = "Built-in fallback sample (no Excel or saved session found)"
    st.session_state.base_vehicle = st.session_state.vehicles[0] if st.session_state.vehicles else None
    st.session_state.relevance = {}
    st.session_state.score_mode = "Engineering"
    st.session_state.numeric_mode = "Relative"
    st.session_state.initialized = True



# ============================================================================
# SIDEBAR — global controls (mirrors the header + Framework Configuration
# panel from the React versions, but every screen reads live from here)
# ============================================================================
with st.sidebar:
    st.markdown("## Feature Intelligence Framework")
    st.caption(st.session_state.data_source)

    _all_classes = sorted(set((st.session_state.class_.get(v) or "Unclassified") for v in st.session_state.vehicles))
    _scope_options = ["Overall Market"] + _all_classes
    if "market_scope" not in st.session_state or st.session_state.market_scope not in _scope_options:
        st.session_state.market_scope = "Overall Market"
    st.session_state.market_scope = st.selectbox("Market Scope", _scope_options,
        index=_scope_options.index(st.session_state.market_scope),
        help="Restricts every comparison view (charts, tables, rankings) to vehicles in this segment. Data Input still shows every vehicle regardless of scope.")

    _scoped_vehicles = st.session_state.vehicles if st.session_state.market_scope == "Overall Market" else [
        v for v in st.session_state.vehicles if (st.session_state.class_.get(v) or "Unclassified") == st.session_state.market_scope]
    st.markdown(
        f"""<div style="background:#FFFFFF; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; font-weight:700; color:#667085;">Vehicles in Scope</div>
        <div style="font-size:1.55rem; font-weight:800; color:#0B2559;">{len(_scoped_vehicles)} / {len(st.session_state.vehicles)}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if st.session_state.base_vehicle not in _scoped_vehicles and _scoped_vehicles:
        st.session_state.base_vehicle = _scoped_vehicles[0]
    st.session_state.base_vehicle = st.selectbox(
        "Base vehicle", _scoped_vehicles,
        index=_scoped_vehicles.index(st.session_state.base_vehicle) if st.session_state.base_vehicle in _scoped_vehicles else 0,
        help="Only shows vehicles within the Market Scope above. Switch scope to \"Overall Market\" to pick any vehicle as base.",
    )
    st.session_state.score_mode = st.radio("Score mode", ["Engineering", "Customer-Weighted"], horizontal=True)
    st.session_state.numeric_mode = st.radio("Numeric normalization", ["Relative", "Fixed Ceiling"], horizontal=True)

    st.markdown("---")
    st.markdown("### Framework Configuration")
    cat_df = pd.DataFrame(st.session_state.category_config).set_index("id")
    cat_df["weight_pct"] = (cat_df["weight"] * 100).round(0)
    edited_cats = st.data_editor(
        cat_df[["name", "weight_pct", "multiplier", "color"]],
        column_config={
            "name": "Category", "weight_pct": st.column_config.NumberColumn("Weight %", min_value=0, max_value=100, step=1),
            "multiplier": st.column_config.NumberColumn("Multiplier", min_value=0.0, step=0.1),
            "color": "Color (hex)",
        },
        num_rows="dynamic", key="category_editor", use_container_width=True,
    )
    if st.button("Apply Framework Configuration", use_container_width=True):
        old_by_id = {c["id"]: c for c in st.session_state.category_config}
        new_config = []
        rename_map = {}
        seen_names = set()
        for idx, r in edited_cats.iterrows():
            name = str(r["name"]).strip()
            if not name or name in seen_names:
                continue  # blank or duplicate name — skip rather than silently merge two categories
            seen_names.add(name)
            entry = {"weight": (r["weight_pct"] or 0) / 100.0, "multiplier": float(r["multiplier"] or 0), "color": r["color"] or "#667085"}
            if idx in old_by_id:
                # existing category, matched by its stable id — safe even if reordered
                entry["id"] = idx
                entry["name"] = name
                if old_by_id[idx]["name"] != name:
                    rename_map[old_by_id[idx]["name"]] = name
            else:
                # a brand-new row added via the editor's "+" — assign a fresh id
                entry["id"] = st.session_state.next_cat_id
                st.session_state.next_cat_id += 1
                entry["name"] = name
            new_config.append(entry)

        surviving_ids = {c["id"] for c in new_config}
        removed_names = [c["name"] for c in st.session_state.category_config if c["id"] not in surviving_ids]

        for r in st.session_state.rows:
            if r["category"] in rename_map:
                r["category"] = rename_map[r["category"]]
        st.session_state.rows = [r for r in st.session_state.rows if r["category"] not in removed_names]
        st.session_state.category_config = new_config
        if removed_names:
            st.info(f"Removed categories: {', '.join(removed_names)} — their feature rows were removed too.")
        st.success("Framework updated.")
        st.rerun()

    total_weight = sum(c["weight"] for c in st.session_state.category_config)
    if abs(total_weight - 1) < 0.005:
        st.success(f"Total weight: {total_weight*100:.0f}%")
    else:
        st.warning(f"Total weight: {total_weight*100:.0f}% (doesn't sum to 100%)")

    st.markdown("### Cost Intelligence")
    st.session_state.currency_symbol = st.text_input("Currency symbol", st.session_state.currency_symbol)
    st.session_state.price_unit = st.text_input("Price unit label", st.session_state.price_unit)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save", use_container_width=True):
            ok = save_session_to_disk()
            st.success("Saved.") if ok else st.error("Save failed.")
    with c2:
        if st.button("↺ Reset", use_container_width=True):
            for k in ["initialized"]:
                del st.session_state[k]
            st.rerun()


# ============================================================================
# DERIVED VALUES (recomputed every rerun — cheap, pure functions)
# ============================================================================
vehicles = st.session_state.vehicles
rows = st.session_state.rows
category_config = st.session_state.category_config
cat_names = [c["name"] for c in category_config]
cat_by_name = {c["name"]: c for c in category_config}
base = st.session_state.base_vehicle if st.session_state.base_vehicle in vehicles else (vehicles[0] if vehicles else None)
numeric_stats = compute_numeric_stats(vehicles, rows)
scores = compute_scores(vehicles, rows, category_config, st.session_state.relevance,
                         "fixed" if st.session_state.numeric_mode == "Fixed Ceiling" else "relative", numeric_stats)
score_key = "customer_final" if st.session_state.score_mode == "Customer-Weighted" else "engineering_final"
cat_key = "customer_category_scores" if st.session_state.score_mode == "Customer-Weighted" else "category_scores"


def final_of(v): return scores[v][score_key] if v in scores else 0.0
def cat_score_of(v, c): return scores[v][cat_key].get(c, 0.0) if v in scores else 0.0
def price_of(v): return st.session_state.price.get(v, 1.0)
def class_of(v): return st.session_state.class_.get(v, "Unclassified")
def money(n): return f"{st.session_state.currency_symbol}{n:,.2f} {st.session_state.price_unit}"


_scope = st.session_state.get("market_scope", "Overall Market")
if _scope == "Overall Market":
    active_vehicles = list(vehicles)
else:
    active_vehicles = [v for v in vehicles if class_of(v) == _scope]
    if base and base not in active_vehicles:
        active_vehicles = [base] + active_vehicles  # base always visible even outside the selected segment

ranked = sorted(active_vehicles, key=final_of, reverse=True)
best_vehicle = ranked[0] if ranked else None
worst_vehicle = ranked[-1] if ranked else None
non_base = [v for v in active_vehicles if v != base]
gaps = {v: final_of(v) - final_of(base) for v in active_vehicles} if base else {}
# Best-in-scope can legitimately BE the base vehicle (it means base is winning) — but a
# "Recommendation Engine: X vs X" chart is a real bug, not a legitimate self-comparison.
# The Recommendation Engine specifically needs the best NON-base competitor.
best_competitor = max(non_base, key=final_of) if non_base else None


def fmt_signed(x, decimals=2):
    """+1.28 / -1.28 / 0.00 — never the '+-1.28' bug from string-concatenating a
    hardcoded '+' onto a value that might already be negative."""
    return f"{x:+.{decimals}f}"


# ============================================================================
# SCREENS
# ============================================================================
st.title("Feature Intelligence Framework")
st.caption("Every category, weight, multiplier, and color is editable in the sidebar. Every feature Type — Binary, Tier, Numeric, Categorical — scores through the same formula.")

TAB_NAMES = ["Overview", "Head-to-Head", "Segment & Position", "Score Comparison", "Category / Subgroup",
             "Feature Gap", "Radar", "Score Matrix", "Baseline vs VFM", "Variant Analysis", "Audit Trail", "Data Input", "Full Report"]
tabs = st.tabs(TAB_NAMES)

# ---------------- OVERVIEW ----------------
with tabs[0]:
    scorable_count = sum(1 for r in rows if r["type"] != "Categorical" and r["category"] in cat_by_name)
    st.subheader("Market Snapshot")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Vehicles Tracked", len(active_vehicles))
    d2.metric("Features Tracked", scorable_count)
    d3.metric("OEMs in Scope", len(set(v.split()[0] for v in active_vehicles)))
    avg_score = sum(final_of(v) for v in active_vehicles) / len(active_vehicles) if active_vehicles else 0
    d4.metric("Average Final Score", f"{avg_score:.2f}")

    all_ranked_df = pd.DataFrame({"Vehicle": ranked, "Final Score": [round(final_of(v), 2) for v in ranked],
                                   "Class": [class_of(v) for v in ranked]})
    fig_market = px.bar(all_ranked_df, x="Vehicle", y="Final Score", color="Class", color_discrete_sequence=CLASS_COLORS)
    fig_market.update_layout(height=340, xaxis=dict(tickangle=-45, automargin=True), legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_market, use_container_width=True)
    st.caption("Ranked by Final Score across every vehicle currently in scope — this is the headline market view. Everything below explains a specific piece of it.")

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    if best_vehicle:
        best_tier = tier_label_relative(scores[best_vehicle]["feature_rating"], [scores[v]["feature_rating"] for v in active_vehicles])[0]
        c1.metric("Market Leader", best_vehicle, f"{final_of(best_vehicle):.2f}  \u00b7  {best_tier}")
    else:
        c1.metric("Market Leader", "—")
    c2.metric("Needs the Most Work", worst_vehicle, f"{final_of(worst_vehicle):.2f}" if worst_vehicle else "—")
    if len(non_base) >= 2:
        hp = max(non_base, key=lambda v: gaps[v])
        hn = min(non_base, key=lambda v: gaps[v])
        hp_label = "Highest Positive Gap" if gaps[hp] > 0 else "Closest Competitor (still behind base)"
        hn_label = "Highest Negative Gap" if gaps[hn] < 0 else "Weakest Competitor (still ahead of base)"
        c3.metric(hp_label, hp, fmt_signed(gaps[hp]))
        c4.metric(hn_label, hn, fmt_signed(gaps[hn]))
    elif len(non_base) == 1:
        v = non_base[0]
        label = "Only Competitor in Scope" if gaps[v] != 0 else "Only Competitor (tied with base)"
        c3.metric(label, v, fmt_signed(gaps[v]))
        c4.metric(" ", "", "")  # intentionally blank — one competitor can't produce two distinct extremes
    else:
        c3.metric("Highest Positive Gap", "—", "No competitors in this scope")
        c4.metric("Highest Negative Gap", "—", "No competitors in this scope")
    with st.expander("How is Final Score calculated?"):
        st.markdown(
            "**Final Score** = weighted sum of 4 category scores (Safety, Comfort, Technology, Utility — weights editable in the sidebar). "
            "Each category score is the sum of every feature's contribution: `(how much of the feature the vehicle has, 0 to 1) \u00d7 category multiplier`. "
            "A vehicle only scores on features it's actually been researched for — an unresearched feature contributes 0, the same as a real absence. "
            f"That's why even the top vehicle here typically doesn't approach a theoretical maximum: it reflects **{scores[best_vehicle]['feature_count'] if best_vehicle else 0} of {scorable_count} scorable features researched**, not a ceiling in the math."
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if best_competitor is None:
            st.subheader("Recommendation Engine")
            st.info(f"No competitors in the current scope to compare {base} against — widen Market Scope in the sidebar.")
        elif best_competitor == base:
            st.subheader("Recommendation Engine")
            st.success(f"{base} is already the top scorer in this scope — no competitor to close a gap against.")
        else:
            st.subheader(f"Recommendation Engine: {best_competitor} vs {base}")
            rec = sorted([{"Category": c, "Gap": cat_score_of(best_competitor, c) - cat_score_of(base, c)} for c in cat_names],
                         key=lambda x: -x["Gap"])
            fig = px.bar(pd.DataFrame(rec), x="Gap", y="Category", orientation="h",
                         color="Gap", color_continuous_scale=["#DC2626", "#94A3B8", ACCENT])
            fig.update_layout(height=300, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Positive bars = categories where {best_competitor} leads {base}. Formula: {best_competitor}'s category score minus {base}'s, for each category.")
    with col2:
        st.subheader(f"Top Priority Features to add to {base}")
        priority_rows = []
        for r in rows:
            base_c = contribution_for(r, base, cat_by_name, numeric_stats, "relative") if base else 0
            others = [contribution_for(r, v, cat_by_name, numeric_stats, "relative") for v in non_base]
            avg_others = (sum(others) / len(others)) if others else 0
            pen = (sum(1 for o in others if o > 0) / len(others)) if others else 0
            weight = cat_by_name.get(r["category"], {"weight": 0})["weight"]
            priority = max(0, avg_others - base_c) * pen * weight
            priority_rows.append({"Feature": r["feature"], "Category": r["category"], "Priority": round(priority, 3), "Penetration %": round(pen * 100)})
        top5 = sorted(priority_rows, key=lambda x: -x["Priority"])[:5]
        if top5:
            fig_top5 = px.bar(pd.DataFrame(top5), x="Priority", y="Feature", orientation="h", color="Category",
                               color_discrete_map={catCfg["name"]: catCfg["color"] for catCfg in category_config})
            fig_top5.update_layout(height=280, yaxis=dict(autorange="reversed", automargin=True), bargap=0.4,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, title=None), margin=dict(l=10))
            st.plotly_chart(fig_top5, use_container_width=True)
        st.dataframe(pd.DataFrame(top5), use_container_width=True, hide_index=True)
        st.caption("Priority = (competitor avg − base) × competitor penetration × category weight")

# ---------------- HEAD TO HEAD ----------------
with tabs[1]:
    colA, colB = st.columns(2)
    a = colA.selectbox("Vehicle A", active_vehicles, index=0, key="cmp_a")
    b = colB.selectbox("Vehicle B", active_vehicles, index=min(1, len(active_vehicles)-1), key="cmp_b")
    fa, fb = final_of(a), final_of(b)
    winner = "Tied" if abs(fa - fb) < 0.005 else (a if fa > fb else b)
    st.metric("Overall Winner", winner, f"{fa:.2f} vs {fb:.2f}")

    cmp_df = pd.DataFrame([{"Category": c, a: cat_score_of(a, c), b: cat_score_of(b, c)} for c in cat_names])
    fig = go.Figure()
    fig.add_bar(name=a, x=cmp_df["Category"], y=cmp_df[a], marker_color=ACCENT)
    fig.add_bar(name=b, x=cmp_df["Category"], y=cmp_df[b], marker_color="#F59E0B")
    fig.update_layout(barmode="group", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature-Level Difference")
    diff_rows = []
    for r in rows:
        va, vb = display_value(r, a), display_value(r, b)
        ca = contribution_for(r, a, cat_by_name, numeric_stats, "relative")
        cb = contribution_for(r, b, cat_by_name, numeric_stats, "relative")
        diff_rows.append({"Category": r["category"], "Feature": r["feature"], a: va, b: vb, "Differs": abs(ca - cb) > 0.005})
    diff_df = pd.DataFrame(diff_rows)
    st.dataframe(diff_df.style.apply(lambda row: ["background-color: #FFFBEB" if row["Differs"] else "" for _ in row], axis=1),
                 use_container_width=True, hide_index=True, height=420)

# ---------------- SEGMENT & POSITION ----------------
with tabs[2]:
    classes = sorted(set(class_of(v) for v in vehicles))
    seg_filter = st.selectbox("Filter by class", ["All"] + classes)
    pos_data = [{"Vehicle": v, "Price": price_of(v), "Score": final_of(v), "Class": class_of(v), "Features": scores[v]["feature_count"]}
                for v in vehicles if seg_filter == "All" or class_of(v) == seg_filter]
    st.subheader("Market Map — Price vs. Score")
    fig = px.scatter(pd.DataFrame(pos_data), x="Price", y="Score", color="Class", size="Features",
                      hover_name="Vehicle", hover_data={"Class": True, "Price": ":.2f", "Score": ":.2f", "Features": True},
                      color_discrete_sequence=CLASS_COLORS)
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig.update_layout(height=420, xaxis=dict(automargin=True))
    st.plotly_chart(fig, use_container_width=True)
    if len(pos_data) > 10:
        st.caption("Hover a point for the vehicle name \u2014 with this many vehicles, permanent on-chart labels would overlap. Filter by class above for a labeled close-up.")

    st.markdown("---")
    st.subheader("Feature Penetration Lens")
    feature_names = sorted(set(r["feature"] for r in rows))
    chosen_feature = st.selectbox("Feature", feature_names)
    frow = next((r for r in rows if r["feature"] == chosen_feature), None)
    if frow:
        pool = [v for v in vehicles if seg_filter == "All" or class_of(v) == seg_filter]
        has_it = [v for v in pool if contribution_for(frow, v, cat_by_name, numeric_stats, "relative") > 0]
        pen_pct = round(len(has_it) / len(pool) * 100) if pool else 0
        pc1, pc2 = st.columns([1, 3])
        with pc1:
            st.metric("Penetration", f"{pen_pct}%")
        with pc2:
            st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
            st.progress(pen_pct / 100, text=f"{len(has_it)} of {len(pool)} vehicles have this feature")

        if frow["type"] != "Categorical" and pool:
            mult = cat_by_name.get(frow["category"], {"multiplier": 1})["multiplier"]
            scatter_rows = [{
                "Vehicle": v, "Price": price_of(v), "Class": class_of(v),
                "Feature Level": round(contribution_for(frow, v, cat_by_name, numeric_stats, "relative") / mult, 2) if mult else 0,
                "Value": display_value(frow, v),
            } for v in pool]
            fig_pen = px.scatter(pd.DataFrame(scatter_rows), x="Price", y="Feature Level", color="Class",
                                  hover_name="Vehicle", hover_data={"Class": True, "Value": True, "Price": ":.2f", "Feature Level": ":.2f"},
                                  color_discrete_sequence=CLASS_COLORS)
            fig_pen.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
            fig_pen.update_layout(
                height=380, yaxis=dict(range=[-0.1, 1.1]),
                yaxis_title="Feature Level (0 = absent, 1 = best available)",
                xaxis=dict(automargin=True),
            )
            st.plotly_chart(fig_pen, use_container_width=True)
            st.caption("Hover a point for the vehicle name and exact value — labels are hidden by default so they stay readable as more vehicles are added.")
        st.dataframe(pd.DataFrame([{"Vehicle": v, "Value": display_value(frow, v)} for v in pool]),
                     use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Percentile Rank within Class")
    pct_rows = []
    for v in vehicles:
        pool2 = [final_of(x) for x in vehicles if class_of(x) == class_of(v)]
        p = percentile_rank(final_of(v), pool2)
        pct_rows.append({"Vehicle": v, "Class": class_of(v), "Score": round(final_of(v), 2),
                          "PercentileNum": p, "Percentile": f"{p}th (n={len(pool2)})"})
    pct_df = pd.DataFrame(pct_rows).sort_values("PercentileNum", ascending=True)
    fig_pct = px.bar(pct_df, x="PercentileNum", y="Vehicle", orientation="h", color="Class",
                      color_discrete_sequence=CLASS_COLORS)
    fig_pct.update_layout(height=max(280, 26 * len(pct_df)), xaxis_title="Percentile within class",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, title=None), margin=dict(l=10), bargap=0.3)
    st.plotly_chart(fig_pct, use_container_width=True)
    st.dataframe(pd.DataFrame(pct_rows).drop(columns=["PercentileNum"]), use_container_width=True, hide_index=True)

# ---------------- SCORE COMPARISON ----------------
with tabs[3]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Final Vehicle Score")
        sdf = pd.DataFrame({"Vehicle": ranked, "Score": [round(final_of(v), 2) for v in ranked],
                             "IsBase": [v == base for v in ranked]})
        fig = px.bar(sdf, x="Vehicle", y="Score", color="IsBase", color_discrete_map={True: "#94A3B8", False: ACCENT})
        fig.update_layout(showlegend=False, height=380, xaxis=dict(tickangle=-45, automargin=True))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader(f"Gap vs {base}")
        gdf = pd.DataFrame({"Vehicle": non_base, "Gap": [round(gaps[v], 2) for v in non_base]})
        fig = px.bar(gdf, x="Vehicle", y="Gap", color=gdf["Gap"] >= 0, color_discrete_map={True: POSITIVE, False: NEGATIVE})
        fig.update_layout(showlegend=False, height=380, xaxis=dict(tickangle=-45, automargin=True))
        fig.add_hline(y=0, line_color="#94A3B8")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Final Score Table")
    table = []
    for v in ranked:
        row = {"Vehicle": v + (" (base)" if v == base else ""), "Class": class_of(v)}
        for c in cat_names:
            row[c] = round(cat_score_of(v, c), 2)
        row["Final Score"] = round(final_of(v), 2)
        row["Gap"] = "—" if v == base else f"{gaps[v]:+.2f}"
        table.append(row)
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

# ---------------- CATEGORY / SUBGROUP ----------------
with tabs[4]:
    st.subheader("Category Score by Vehicle")
    cat_chart = pd.DataFrame([{"Category": c, "Vehicle": v, "Score": round(cat_score_of(v, c), 2)} for c in cat_names for v in active_vehicles])
    fig = px.bar(cat_chart, x="Category", y="Score", color="Vehicle", barmode="group", text="Score")
    fig.update_traces(textposition="outside", textfont_size=9)
    fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Top & Bottom per Category — Gap vs {base}")
    tb_rows = []
    for c in cat_names:
        ranked_c = sorted(active_vehicles, key=lambda v: cat_score_of(v, c), reverse=True)
        top_v, bottom_v = ranked_c[0], ranked_c[-1]
        tb_rows.append({
            "Category": c,
            "Top Vehicle": top_v, "Top Score": round(cat_score_of(top_v, c), 2),
            "Bottom Vehicle": bottom_v, "Bottom Score": round(cat_score_of(bottom_v, c), 2),
            f"{base} Score": round(cat_score_of(base, c), 2) if base else "—",
            f"Gap vs {base}": (f"{cat_score_of(top_v, c) - cat_score_of(base, c):+.2f}" if base else "—"),
        })
    st.dataframe(pd.DataFrame(tb_rows), use_container_width=True, hide_index=True)
    st.caption(f"\"Gap vs {base}\" is the top vehicle's lead over your selected base in that category — the size of the ceiling still to close.")

    st.subheader("Subgroup Analysis")
    sc1, sc2 = st.columns([3, 1])
    sub_cat = sc1.selectbox("Category", cat_names, key="subgroup_cat")
    sub_orientation = sc2.radio("Orientation", ["Vertical", "Horizontal"], key="subgroup_orientation")
    subgroups = []
    seen = set()
    for r in rows:
        if r["category"] == sub_cat and r["subgroup"] not in seen:
            seen.add(r["subgroup"]); subgroups.append(r["subgroup"])
    sub_records = []
    for sg in subgroups:
        for v in active_vehicles:
            val = sum(contribution_for(r, v, cat_by_name, numeric_stats, "relative") for r in rows if r["category"] == sub_cat and r["subgroup"] == sg)
            sub_records.append({"Subgroup": sg, "Vehicle": v, "Score": round(val, 2)})
    if sub_records:
        sub_df = pd.DataFrame(sub_records)
        bar_px = min(22, max(10, 260 // max(1, len(active_vehicles))))  # thinner bars as vehicle count grows, never disappearing
        if sub_orientation == "Horizontal":
            chart_height = min(900, max(280, len(subgroups) * max(bar_px * len(active_vehicles), 70)))
            fig2 = px.bar(sub_df, x="Score", y="Subgroup", color="Vehicle", orientation="h", barmode="group", text="Score")
        else:
            chart_height = min(650, max(380, 320 + len(active_vehicles) * 8))
            fig2 = px.bar(sub_df, x="Subgroup", y="Score", color="Vehicle", barmode="group", text="Score")
            fig2.update_layout(xaxis=dict(tickangle=-20, automargin=True))
        fig2.update_traces(textposition="outside", textfont_size=9)
        fig2.update_layout(height=chart_height, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)
        if len(active_vehicles) > 8:
            st.caption(f"Comparing {len(active_vehicles)} vehicles at once gets crowded — narrow the Market Scope in the sidebar for a cleaner read.")

# ---------------- FEATURE GAP ----------------
with tabs[5]:
    perspective = st.radio("Perspective", ["Vehicle", "Feature"], horizontal=True)
    gc1, gc2 = st.columns(2)
    competitor = gc1.selectbox("Competitor(s)", ["All"] + non_base)
    gap_cat = gc2.selectbox("Category filter", ["All"] + cat_names)
    competitors = non_base if competitor == "All" else [competitor]
    filtered_rows = [r for r in rows if gap_cat == "All" or r["category"] == gap_cat]

    if perspective == "Vehicle":
        n_per_row = 3 if len(competitors) >= 3 else max(1, len(competitors))
        for i in range(0, len(competitors), n_per_row):
            row_vehicles = competitors[i:i + n_per_row]
            row_cols = st.columns(len(row_vehicles))
            for col, v in zip(row_cols, row_vehicles):
                with col:
                    st.markdown(f"**{v}**")
                    data = [{"Category": c, "Gap": cat_score_of(v, c) - cat_score_of(base, c)} for c in cat_names]
                    fig = px.bar(pd.DataFrame(data), x="Category", y="Gap", color=pd.DataFrame(data)["Gap"] >= 0,
                                 color_discrete_map={True: POSITIVE, False: NEGATIVE})
                    fig.update_layout(height=260, showlegend=False, margin=dict(l=0, r=0))
                    fig.add_hline(y=0, line_color="#94A3B8")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        gap_table = []
        for r in filtered_rows:
            base_c = contribution_for(r, base, cat_by_name, numeric_stats, "relative")
            others = [contribution_for(r, v, cat_by_name, numeric_stats, "relative") for v in competitors]
            avg_others = (sum(others) / len(others)) if others else 0
            verdict = "Parity"
            if base_c > avg_others + 0.001: verdict = "Base ahead"
            elif base_c < avg_others - 0.001: verdict = "Base behind"
            pen = round((sum(1 for o in others if o > 0) / len(others)) * 100) if others else 0
            weight = cat_by_name.get(r["category"], {"weight": 0})["weight"]
            priority = max(0, avg_others - base_c) * (pen / 100) * weight
            gap_table.append({"Category": r["category"], "Type": r["type"], "Feature": r["feature"],
                               base: display_value(r, base), "Penetration %": pen, "Verdict": verdict, "Priority": round(priority, 3)})
        gap_table.sort(key=lambda x: -x["Priority"])
        top10 = [g for g in gap_table if g["Priority"] > 0][:10]
        if top10:
            fig_gap = px.bar(pd.DataFrame(top10), x="Priority", y="Feature", orientation="h", color="Category",
                              color_discrete_map={catCfg["name"]: catCfg["color"] for catCfg in category_config})
            fig_gap.update_layout(height=max(300, 30 * len(top10)), yaxis=dict(autorange="reversed", automargin=True), bargap=0.35,
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, title=None), margin=dict(l=10))
            st.plotly_chart(fig_gap, use_container_width=True)
        st.dataframe(pd.DataFrame(gap_table), use_container_width=True, hide_index=True, height=460)
        st.caption("Priority Score = (competitor avg − base) × competitor penetration × category weight")

        st.markdown("---")
        if REPORTLAB_AVAILABLE:
            cat_gap_rows = [(c, (sum(cat_score_of(v, c) for v in competitors) / len(competitors)) - cat_score_of(base, c))
                             for c in cat_names] if competitors else []
            pdf_bytes = build_feature_gap_pdf(base, competitors, cat_gap_rows, gap_table, money)
            st.download_button("📄 Download Feature Gap Report (PDF)", pdf_bytes,
                                file_name=f"feature-gap-{base.replace(' ', '-')}.pdf", mime="application/pdf")
        else:
            st.info("PDF export needs the `reportlab` package — add it to requirements.txt and reinstall to enable this.")

# ---------------- RADAR ----------------
with tabs[6]:
    radar_mode = st.radio("Mode", ["Overlay All", "Base vs Each"], horizontal=True)
    if radar_mode == "Overlay All":
        radar_selection = st.multiselect("Vehicles to show", active_vehicles, default=active_vehicles[:6] if len(active_vehicles) > 6 else active_vehicles)
        radar_vehicles = radar_selection if radar_selection else active_vehicles
        if len(active_vehicles) > 6 and len(radar_selection) == len(active_vehicles[:6]):
            st.caption(f"Showing 6 of {len(active_vehicles)} vehicles by default — too many overlapping shapes gets unreadable. Add or remove vehicles above.")
        fig = go.Figure()
        for v in radar_vehicles:
            vals = [cat_score_of(v, c) for c in cat_names] + [cat_score_of(v, cat_names[0])]
            fig.add_trace(go.Scatterpolar(r=vals, theta=cat_names + [cat_names[0]], fill="toself" if v == base else None,
                                           name=v, opacity=0.85 if v == base else 0.5))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        cols = st.columns(min(3, max(1, len(non_base))))
        for i, v in enumerate(non_base):
            with cols[i % len(cols)]:
                st.caption(f"{base} vs {v}")
                fig = go.Figure()
                for name, color in [(base, ACCENT), (v, "#F59E0B")]:
                    vals = [cat_score_of(name, c) for c in cat_names] + [cat_score_of(name, cat_names[0])]
                    fig.add_trace(go.Scatterpolar(r=vals, theta=cat_names + [cat_names[0]], fill="toself", name=name, line_color=color))
                fig.update_layout(height=320, showlegend=True, margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

# ---------------- SCORE MATRIX ----------------
with tabs[7]:
    fr_distribution = [scores[v]["feature_rating"] for v in active_vehicles]
    matrix_rows = []
    for v in active_vehicles:
        row = {"Vehicle": v + (" (base)" if v == base else ""), "Class": class_of(v), "Price": money(price_of(v))}
        for c in cat_names:
            row[c] = round(scores[v]["normalized_category"].get(c, 0), 1)
        row["Feature Rating /10"] = round(scores[v]["feature_rating"], 1)
        row["Tier"] = tier_label_relative(scores[v]["feature_rating"], fr_distribution)[0]
        matrix_rows.append(row)
    matrix_df = pd.DataFrame(matrix_rows)
    st.caption("Tiers (Strong / Moderate / Below Avg / Weak) are ranked relative to the vehicles currently in scope, not a fixed absolute scale \u2014 as verified data gets deeper, the same vehicle's tier can shift even if its own score doesn't change.")

    def _tier_bg_col(col):
        dist = col.tolist()
        styles = []
        for val in col:
            if not isinstance(val, (int, float)):
                styles.append("")
                continue
            _, color, bg = tier_label_relative(val, dist)
            styles.append(f"background-color: {bg}; color: {color}; font-weight: 700;")
        return styles

    _score_cols = cat_names + ["Feature Rating /10"]
    _score_cols = [c for c in _score_cols if c in matrix_df.columns]
    styled = matrix_df.style.apply(_tier_bg_col, subset=_score_cols).format("{:.1f}", subset=_score_cols)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Score Matrix — Category Comparison")
    smx_chart = pd.DataFrame([{"Vehicle": v, "Category": c, "Score": round(scores[v]["normalized_category"].get(c, 0), 1)}
                               for v in active_vehicles for c in cat_names])
    fig_smx = px.bar(smx_chart, x="Vehicle", y="Score", color="Category", barmode="group",
                      color_discrete_map={catCfg["name"]: catCfg["color"] for catCfg in category_config})
    fig_smx.update_layout(height=380, xaxis=dict(tickangle=-45, automargin=True), legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_smx, use_container_width=True)

    st.subheader("Price Efficiency")
    eff = sorted(
        [{"Vehicle": v, "Price": money(price_of(v)), "Rating": round(scores[v]["feature_rating"], 1),
          "Rating ÷ Price": round(scores[v]["feature_rating"] / price_of(v), 2) if price_of(v) > 0 else 0} for v in active_vehicles],
        key=lambda x: -x["Rating ÷ Price"],
    )
    if eff:
        eff[0]["Verdict"] = "Best value"
    if eff:
        fig_eff = px.bar(pd.DataFrame(eff), x="Rating ÷ Price", y="Vehicle", orientation="h",
                          color="Rating ÷ Price", color_continuous_scale=["#DCE3EC", ACCENT])
        fig_eff.update_layout(height=max(280, 30 * len(eff)), yaxis=dict(autorange="reversed"),
                               showlegend=False, coloraxis_showscale=False, margin=dict(l=0))
        st.plotly_chart(fig_eff, use_container_width=True)
    st.dataframe(pd.DataFrame(eff), use_container_width=True, hide_index=True)

# ---------------- BASELINE VS VFM ----------------
with tabs[8]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Category Gap vs {base}")
        for v in non_base:
            data = [{"Category": c, "Gap": cat_score_of(v, c) - cat_score_of(base, c)} for c in cat_names]
            fig = px.bar(pd.DataFrame(data), x="Category", y="Gap", title=v,
                         color=pd.DataFrame(data)["Gap"] >= 0, color_discrete_map={True: POSITIVE, False: NEGATIVE})
            fig.update_layout(height=200, showlegend=False, margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Value for Money (Final Score ÷ Price)")
        vfm = sorted([{"Vehicle": v, "VFM": final_of(v) / price_of(v) if price_of(v) > 0 else 0} for v in active_vehicles], key=lambda x: -x["VFM"])
        fig = px.bar(pd.DataFrame(vfm), x="VFM", y="Vehicle", orientation="h", color_discrete_sequence=["#F59E0B"])
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Average Category Gap (all competitors)")
        avg_gap_rows = [{"Category": c, "Avg Gap": (sum(cat_score_of(v, c) for v in non_base) / len(non_base) - cat_score_of(base, c)) if non_base else 0}
                         for c in cat_names]
        fig_avg = px.bar(pd.DataFrame(avg_gap_rows), x="Category", y="Avg Gap",
                          color=pd.DataFrame(avg_gap_rows)["Avg Gap"] >= 0, color_discrete_map={True: POSITIVE, False: NEGATIVE})
        fig_avg.update_layout(height=340, showlegend=False)
        fig_avg.add_hline(y=0, line_color="#94A3B8")
        st.plotly_chart(fig_avg, use_container_width=True)
        st.caption(f"Positive = competitors average ahead of {base} in that category; negative = {base} leads on average.")
    with col4:
        st.subheader("VFM vs. Price")
        vfm_scatter = pd.DataFrame([{"Vehicle": v, "Price": price_of(v), "VFM": final_of(v) / price_of(v) if price_of(v) > 0 else 0, "Class": class_of(v)} for v in active_vehicles])
        fig_vfm = px.scatter(vfm_scatter, x="Price", y="VFM", color="Class", hover_name="Vehicle",
                              hover_data={"Class": True, "Price": ":.2f", "VFM": ":.2f"}, color_discrete_sequence=CLASS_COLORS)
        fig_vfm.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
        fig_vfm.update_layout(height=340, xaxis=dict(automargin=True))
        st.plotly_chart(fig_vfm, use_container_width=True)
        st.caption("Top-left = high value-for-money at a low price. Hover for vehicle names.")

# ---------------- VARIANT ANALYSIS ----------------
with tabs[9]:
    st.caption("A separate layer from the main feature matrix — this tracks variants *within* one model "
               "(e.g. Brezza LXi / VXi / ZXi / ZXi+), not model-to-model comparison. Add every trim of a "
               "model to see its price ladder and what each step up actually costs in features.")

    st.subheader("Add a Variant")
    with st.form("add_variant_form"):
        vc1, vc2, vc3, vc4 = st.columns(4)
        v_model = vc1.text_input("Model (e.g. Maruti Brezza)")
        v_name = vc2.text_input("Variant name (e.g. ZXi+)")
        v_price = vc3.number_input("Price", min_value=0.0, step=0.05)
        v_feat_count = vc4.number_input("Cumulative feature count", min_value=0, step=1,
                                         help="Total count of key features this variant has (not a delta) — used to compute cost-per-feature-added between trims.")
        v_notes = st.text_input("What's added over the previous trim (optional, for reference)")
        v_submitted = st.form_submit_button("Add variant")
        if v_submitted and v_model and v_name:
            st.session_state.variants.append({
                "id": st.session_state.next_variant_id, "model": v_model.strip(), "variant": v_name.strip(),
                "price": float(v_price), "feature_count": int(v_feat_count), "notes": v_notes.strip(),
            })
            st.session_state.next_variant_id += 1
            st.success(f"Added {v_model} {v_name}.")
            st.rerun()

    variants = st.session_state.variants
    if not variants:
        st.info("No variants added yet — add at least 2 variants of the same model above to see a price ladder and cost-per-feature analysis.")
    else:
        var_df = pd.DataFrame(variants)
        models = sorted(var_df["model"].unique())
        st.markdown("---")
        st.subheader("Variant Price Ladder")
        fig = go.Figure()
        for m in models:
            sub = var_df[var_df["model"] == m].sort_values("price")
            fig.add_trace(go.Scatter(x=sub["price"], y=[m] * len(sub), mode="markers+text",
                                      text=sub["variant"], textposition="top center",
                                      marker=dict(size=12), name=m))
        fig.update_layout(height=120 + 60 * len(models), xaxis_title=f"Price ({st.session_state.currency_symbol} {st.session_state.price_unit})",
                           showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cost Per Feature Added (trim-to-trim, within each model)")
        cost_rows = []
        for m in models:
            sub = var_df[var_df["model"] == m].sort_values("price").reset_index(drop=True)
            for i in range(1, len(sub)):
                prev, cur = sub.iloc[i - 1], sub.iloc[i]
                price_delta = cur["price"] - prev["price"]
                feat_delta = cur["feature_count"] - prev["feature_count"]
                cost_per_feature = (price_delta / feat_delta) if feat_delta > 0 else None
                cost_rows.append({
                    "Model": m, "From": prev["variant"], "To": cur["variant"],
                    "Price Delta": round(price_delta, 2), "Features Added": feat_delta,
                    "Cost per Feature": round(cost_per_feature, 3) if cost_per_feature is not None else "—",
                    "Notes": cur["notes"],
                })
        if cost_rows:
            chartable = [r for r in cost_rows if isinstance(r["Cost per Feature"], (int, float))]
            if chartable:
                cdf = pd.DataFrame(chartable)
                cdf["Step"] = cdf["Model"] + ": " + cdf["From"] + " \u2192 " + cdf["To"]
                fig_cost = px.bar(cdf, x="Cost per Feature", y="Step", orientation="h",
                                   color="Cost per Feature", color_continuous_scale=[POSITIVE, "#F59E0B", NEGATIVE])
                fig_cost.update_layout(height=max(240, 34 * len(cdf)), showlegend=False, coloraxis_showscale=False, margin=dict(l=0))
                st.plotly_chart(fig_cost, use_container_width=True)
            st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True)
            st.caption("Lower cost-per-feature = more feature value packed into that upgrade step. A trim with a big price jump but few new features stands out immediately here — useful for spotting badly-priced trims in your own lineup or a competitor's.")
        else:
            st.info("Add at least 2 variants of the same model to compute cost-per-feature steps.")

        st.markdown("---")
        st.subheader("All Variants")
        display_df = var_df[["model", "variant", "price", "feature_count", "notes"]].rename(
            columns={"model": "Model", "variant": "Variant", "price": "Price", "feature_count": "Feature Count", "notes": "Notes"})
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        to_remove = st.selectbox("Remove a variant", ["—"] + [f"{v['model']} {v['variant']}" for v in variants])
        if st.button("Remove selected variant") and to_remove != "—":
            st.session_state.variants = [v for v in variants if f"{v['model']} {v['variant']}" != to_remove]
            st.rerun()


with tabs[10]:
    audit_rows = []
    for r in rows:
        row = {"Category": r["category"], "Subgroup": r["subgroup"], "Feature": r["feature"], "Type": r["type"]}
        for v in active_vehicles:
            row[v] = display_value(r, v)
        row["Multiplier"] = cat_by_name.get(r["category"], {"multiplier": 1})["multiplier"]
        audit_rows.append(row)
    audit_df = pd.DataFrame(audit_rows)
    st.dataframe(audit_df, use_container_width=True, height=460)
    csv = audit_df.to_csv(index=False).encode("utf-8")
    st.download_button("Export Audit Trail (CSV)", csv, "feature-intelligence-audit.csv", "text/csv")

# ---------------- DATA INPUT ----------------
with tabs[11]:
    st.subheader("Upload a Feature Matrix (.xlsx)")
    uploaded = st.file_uploader("Excel file matching the standard schema", type=["xlsx"])
    if uploaded is not None and st.button("Load uploaded file"):
        v2, r2, p2, c2 = parse_excel_bytes(uploaded.read(), known_category_names=cat_names)
        st.session_state.vehicles, st.session_state.rows = v2, r2
        st.session_state.price, st.session_state.class_ = p2, c2
        st.session_state.base_vehicle = v2[0] if v2 else None
        st.session_state.data_source = f"Uploaded: {uploaded.name}"
        st.rerun()

    st.markdown("---")
    st.subheader("Edit the Feature Matrix directly")
    edit_df = rows_to_dataframe(vehicles, rows)
    edited = st.data_editor(edit_df, num_rows="dynamic", use_container_width=True, height=420, key="matrix_editor")
    if st.button("Apply edits to feature matrix"):
        st.session_state.rows = dataframe_to_rows(edited, vehicles)
        st.success("Feature matrix updated.")
        st.rerun()

    st.markdown("---")
    st.subheader("Vehicle Class & Price")
    vc_rows = [{"Vehicle": v, "Class": class_of(v), "Price": price_of(v)} for v in vehicles]
    vc_edited = st.data_editor(pd.DataFrame(vc_rows), num_rows="fixed", use_container_width=True, key="vc_editor")
    if st.button("Apply class/price edits"):
        for _, r in vc_edited.iterrows():
            st.session_state.class_[r["Vehicle"]] = r["Class"]
            st.session_state.price[r["Vehicle"]] = float(r["Price"])
        st.success("Updated.")
        st.rerun()

    st.markdown("---")
    st.subheader("Add a Vehicle")
    with st.form("add_vehicle_form"):
        new_name = st.text_input("Vehicle name")
        new_class = st.text_input("Class")
        new_price = st.number_input("Price", min_value=0.0, step=0.1)
        st.caption("New vehicle starts with every feature at its lowest value — edit them in the matrix editor above after adding.")
        submitted = st.form_submit_button("Add vehicle")
        if submitted and new_name:
            if new_name in st.session_state.vehicles:
                st.error("That vehicle name already exists.")
            else:
                st.session_state.vehicles.append(new_name)
                for r in st.session_state.rows:
                    r["values"][new_name] = "" if r["type"] == "Categorical" else 0
                st.session_state.price[new_name] = new_price
                st.session_state.class_[new_name] = new_class
                st.success(f"Added {new_name}.")
                st.rerun()

    st.markdown("---")
    st.subheader("Quick Add — paste a spec block")
    quick_text = st.text_area("Format: one 'Feature: value' per line, plus Name/Class/Price", height=150,
                               placeholder="Name: Baleno\nClass: Premium Hatchback\nPrice: 9.17\nABS with EBD: 1")
    if st.button("Add from pasted text"):
        kv = {}
        for line in quick_text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                kv[k.strip().lower()] = v.strip()
        name = kv.get("name", "").strip()
        if not name:
            st.error('Needs a "Name: ..." line.')
        elif name in st.session_state.vehicles:
            st.error("That vehicle name already exists.")
        else:
            st.session_state.vehicles.append(name)
            for r in st.session_state.rows:
                raw = kv.get(r["feature"].lower())
                if r["type"] == "Binary":
                    r["values"][name] = 1 if raw and raw.lower() in ("1", "yes", "true") else 0
                elif r["type"] == "Tier":
                    levels = r["levels"] or ["Absent", "Present"]
                    try:
                        r["values"][name] = max(0, min(len(levels) - 1, int(float(raw)))) if raw else 0
                    except Exception:
                        idx = next((i for i, l in enumerate(levels) if raw and l.lower() == raw.lower()), 0)
                        r["values"][name] = idx
                elif r["type"] == "Numeric":
                    try: r["values"][name] = float(raw) if raw else 0
                    except Exception: r["values"][name] = 0
                else:
                    r["values"][name] = raw or ""
            if "price" in kv:
                try: st.session_state.price[name] = float(kv["price"])
                except Exception: pass
            if "class" in kv:
                st.session_state.class_[name] = kv["class"]
            st.success(f"Added {name}.")
            st.rerun()

# ---------------- FULL REPORT ----------------
with tabs[12]:
    scope_on = st.checkbox(f"Scope to {class_of(base)} only", value=False)
    report_vehicles = [v for v in active_vehicles if not scope_on or class_of(v) == class_of(base)]
    report_ranked = sorted(report_vehicles, key=final_of, reverse=True)

    st.subheader("Report Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vehicles in Scope", len(report_vehicles))
    c2.metric("Best in Scope", report_ranked[0] if report_ranked else "—")
    c3.metric("Worst in Scope", report_ranked[-1] if report_ranked else "—")
    c4.metric("Base Vehicle", base)

    st.subheader("Final Score & Category Breakdown")
    rep_table = []
    for v in report_ranked:
        row = {"Vehicle": v + (" (base)" if v == base else ""), "Class": class_of(v), "Price": money(price_of(v))}
        for c in cat_names:
            row[c] = round(cat_score_of(v, c), 2)
        row["Final Score"] = round(final_of(v), 2)
        row["Gap"] = "—" if v == base else f"{gaps[v]:+.2f}"
        rep_table.append(row)
    report_df = pd.DataFrame(rep_table)
    fig_rep = px.bar(report_df, x="Vehicle", y="Final Score", color="Vehicle", color_discrete_sequence=CLASS_COLORS)
    fig_rep.update_layout(height=340, showlegend=False, xaxis=dict(tickangle=-45, automargin=True))
    st.plotly_chart(fig_rep, use_container_width=True)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Overall Vehicle Summary — every key detail, one row per vehicle")
    fr_dist_report = [scores[v]["feature_rating"] for v in report_vehicles]
    summary_rows = []
    for v in report_ranked:
        pool = [final_of(x) for x in vehicles if class_of(x) == class_of(v)]
        summary_rows.append({
            "Vehicle": v, "Class": class_of(v), "Price": money(price_of(v)),
            "Safety": round(cat_score_of(v, "Safety"), 2) if "Safety" in cat_names else "—",
            "Comfort": round(cat_score_of(v, "Comfort"), 2) if "Comfort" in cat_names else "—",
            "Technology": round(cat_score_of(v, "Technology"), 2) if "Technology" in cat_names else "—",
            "Utility": round(cat_score_of(v, "Utility"), 2) if "Utility" in cat_names else "—",
            "Final Score": round(final_of(v), 2),
            "Tier": tier_label_relative(scores[v]["feature_rating"], fr_dist_report)[0],
            "VFM (Score \u00f7 Price)": round(final_of(v) / price_of(v), 3) if price_of(v) > 0 else "—",
            "Percentile in Class": f"{percentile_rank(final_of(v), pool)}th",
            "Features Present": scores[v]["feature_count"],
            "Base Vehicle?": "Yes" if v == base else "No",
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.caption("This table is also included as its own sheet in the Excel export below, and drives both PDF report options.")

    st.markdown("---")
    st.subheader("Downloads")
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Vehicle Summary", index=False)
            report_df.to_excel(writer, sheet_name="Final Scores", index=False)
            pd.DataFrame(matrix_rows if "matrix_rows" in dir() else []).to_excel(writer, sheet_name="Score Matrix", index=False)
            audit_df.to_excel(writer, sheet_name="Audit Trail", index=False)
        st.download_button("\U0001F4CA Full Report (Excel)", buf.getvalue(), "feature-intelligence-report.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with dcol2:
        if REPORTLAB_AVAILABLE:
            top_priority_for_pdf = sorted(
                [{"Feature": r["feature"], "Category": r["category"],
                  "Priority": round(max(0, (sum(contribution_for(r, v, cat_by_name, numeric_stats, "relative") for v in [x for x in report_vehicles if x != base]) / max(1, len([x for x in report_vehicles if x != base])) - contribution_for(r, base, cat_by_name, numeric_stats, "relative"))) * cat_by_name.get(r["category"], {"weight": 0})["weight"], 3),
                  "Penetration %": round(sum(1 for v in [x for x in report_vehicles if x != base] if contribution_for(r, v, cat_by_name, numeric_stats, "relative") > 0) / max(1, len([x for x in report_vehicles if x != base])) * 100)}
                 for r in rows], key=lambda x: -x["Priority"])[:8]
            summary_pdf_bytes = build_summary_pdf(report_vehicles, rep_table, top_priority_for_pdf, report_ranked[0] if report_ranked else None,
                                                   report_ranked[-1] if report_ranked else None, base, money)
            st.download_button("\U0001F4C4 Summary Report (PDF)", summary_pdf_bytes, "feature-intelligence-summary.pdf",
                                "application/pdf", use_container_width=True)
        else:
            st.button("\U0001F4C4 Summary Report (PDF)", disabled=True, use_container_width=True, help="reportlab not installed")
    with dcol3:
        if REPORTLAB_AVAILABLE:
            full_pdf_bytes = build_full_pdf(rep_table, matrix_rows if "matrix_rows" in dir() else [], report_vehicles, rows, base, money)
            st.download_button("\U0001F4D1 Full Report (PDF, all pages)", full_pdf_bytes, "feature-intelligence-full-report.pdf",
                                "application/pdf", use_container_width=True)
            st.caption("The full PDF includes one readable table per vehicle — every tracked feature, so it runs long by design (roughly one page per vehicle). Use the Summary PDF for a quick read.")
        else:
            st.button("\U0001F4D1 Full Report (PDF, all pages)", disabled=True, use_container_width=True, help="reportlab not installed")
