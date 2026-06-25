"""
7_PDF_TO_EXCEL.py  —  FBC Investment Valuation System
PDF (scanned financial statements) + Image Screenshot → Excel converter

Two conversion modes:
  • PDF Mode  — scanned multi-page PDF → one Excel sheet per page
  • Image Mode — screenshot / photo (PNG, JPG, WEBP, BMP, TIFF) → single Excel sheet

System requirements (repo root):
  packages.txt  →  tesseract-ocr
                   poppler-utils
  requirements.txt → pytesseract
                     pdf2image
                     Pillow
                     openpyxl
"""

import io
import os
import re

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF / Image → Excel | FBC Valuation", layout="wide")

# ── Auth guard ────────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True
if "user" not in st.session_state or not st.session_state.get("user"):
    st.session_state["user"] = {
        "username": "analyst",
        "role": "analyst",
        "full_name": "Analyst",
    }

# ── Lazy imports ──────────────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image
    LIBS_OK = True
    _IMPORT_ERR = ""
except ImportError as _e:
    LIBS_OK = False
    _IMPORT_ERR = str(_e)

PDF_LIBS_OK = False
if LIBS_OK:
    try:
        from pdf2image import convert_from_bytes
        PDF_LIBS_OK = True
    except ImportError:
        pass

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment
    from openpyxl.utils import get_column_letter
    OPENPYXL_OK = True
except ImportError:
    OPENPYXL_OK = False

# ── Tesseract path ────────────────────────────────────────────────────────────
_tess_env = os.environ.get("TESSERACT_CMD", "")
if LIBS_OK:
    pytesseract.pytesseract.tesseract_cmd = _tess_env if _tess_env else "tesseract"

# ── FBC Design System ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

html, body, .stApp, .block-container,
p, div, label, h1, h2, h3, h4, h5, h6, li, ul, ol, a, small {
  font-family: "EB Garamond", Georgia, "Times New Roman", serif !important;
  color: #1a1a2e;
}
h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: "Playfair Display", Georgia, serif !important;
  font-weight: 700 !important;
}
.stApp { background: #f5f7fb !important; }
.main .block-container { background: #f5f7fb !important; padding-top: 1.5rem !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #001a5c 0%, #003399 45%, #0044cc 100%) !important;
    border-right: 2px solid rgba(245,180,0,0.25) !important;
}
section[data-testid="stSidebar"] * {
    color: #e8f0ff !important;
    font-family: "EB Garamond", Georgia, serif !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: "Playfair Display", serif !important;
    font-weight: 700 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #003399 0%, #0055ee 100%) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-family: "EB Garamond", serif !important; font-size: 15px !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 14px rgba(0,51,153,0.30) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }
.stButton > button * { color: #ffffff !important; }

.stDownloadButton > button {
    background: linear-gradient(135deg, #003399, #0044cc) !important;
    color: white !important; border-radius: 10px !important;
    font-weight: 700 !important; border: none !important;
}
.stDownloadButton > button * { color: #ffffff !important; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #f0f5ff, #fff8e6) !important;
    border: 1px solid rgba(0,51,153,0.12) !important;
    border-radius: 14px !important; padding: 14px 16px !important;
}

[data-testid="stFileUploader"] {
    background: rgba(0, 51, 153, 0.04);
    border: 1px dashed rgba(0, 51, 153, 0.25);
    border-radius: 14px; padding: 18px 20px;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #003399; border-radius: 999px; }

.fbc-section {
    display: block; padding: 16px 0; margin: 28px 0 18px 0;
    border-bottom: 2px solid rgba(0,51,153,0.15);
}
.fbc-section-title {
    font-family: "Playfair Display", serif !important;
    font-size: 21px; font-weight: 800; color: #001a5c !important;
}

.fbc-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #f5b400, #003399, transparent);
    border: none; margin: 20px 0; border-radius: 999px;
}

.fbc-footer {
    text-align: center; padding: 22px; margin-top: 40px;
    color: #5a7099 !important; font-size: 13px;
    border-top: 1px solid rgba(0,51,153,0.10); font-style: italic;
}
.fbc-footer b { color: #003399 !important; font-style: normal; }

/* Mode selector tabs */
.mode-card {
    background: #ffffff;
    border: 2px solid rgba(0,51,153,0.12);
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.mode-card.active {
    border-color: #003399;
    border-left: 6px solid #f5b400;
    background: linear-gradient(135deg, #f0f5ff, #fffdf0);
    box-shadow: 0 8px 22px rgba(0,51,153,0.14);
}
.mode-card-icon { font-size: 28px; margin-bottom: 6px; }
.mode-card-title {
    font-family: "Playfair Display", serif !important;
    font-size: 17px; font-weight: 800; color: #001a5c;
}
.mode-card-desc { font-size: 13px; color: #475569; margin-top: 4px; }

/* Result card */
.result-card {
    background: #ffffff;
    border: 1px solid rgba(0,51,153,0.12);
    border-left: 5px solid #003399;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

/* Warn card */
.fbc-warn-card {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border-radius: 14px; padding: 16px 22px; margin: 14px 0;
    border-left: 5px solid #f5b400;
}
.fbc-warn-title { font-family: "Playfair Display", serif !important; font-weight: 700; color: white !important; margin-bottom: 4px; font-size: 15px; }
.fbc-warn-body { color: rgba(255,255,255,0.88) !important; font-size: 13px; line-height: 1.6; }

/* Info card */
.fbc-info-card {
    background: linear-gradient(135deg, #001a5c, #003399);
    border-radius: 16px; padding: 20px 26px; margin-bottom: 20px;
    border-bottom: 3px solid #f5b400;
    box-shadow: 0 8px 24px rgba(0,26,92,0.22);
}
.fbc-info-card-title {
    font-family: "Playfair Display", serif !important;
    font-size: 19px; font-weight: 800; color: #ffffff !important; margin-bottom: 8px;
}
.fbc-info-card-body { color: rgba(255,255,255,0.85) !important; font-size: 14px; line-height: 1.8; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def section(title: str):
    st.markdown(
        f'<div class="fbc-section"><div class="fbc-section-title">{title}</div></div>',
        unsafe_allow_html=True,
    )

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(90deg, #001233 0%, #001a5c 40%, #003399 100%);
    border-radius: 16px; padding: 18px 28px; margin-bottom: 4px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 6px 24px rgba(0,26,92,0.32);
    border-bottom: 3px solid #f5b400;">
    <div>
        <div style="font-family:'Playfair Display',serif; font-size:22px; font-weight:900;
                    color:#ffffff; letter-spacing:-0.01em; text-shadow:0 2px 8px rgba(0,0,0,0.30);">
            📄 PDF &amp; Image → Excel Converter
        </div>
        <div style="font-family:'EB Garamond',serif; font-size:13px; font-style:italic;
                    color:rgba(255,255,255,0.65); margin-top:2px;">
            Extract scanned financial statements — PDF or screenshot — into structured Excel files
        </div>
    </div>
    <div style="background:rgba(245,180,0,0.22); border:1.5px solid rgba(245,180,0,0.60);
                color:#ffd040; font-size:13px; font-weight:700; padding:6px 18px;
                border-radius:999px; font-family:'EB Garamond',serif; white-space:nowrap;">
        FBC Securities
    </div>
</div>
<hr style="border:none; border-top:2px solid #dde6f5; margin:6px 0 20px 0;">
""", unsafe_allow_html=True)

# ── Dependency check ──────────────────────────────────────────────────────────
if not LIBS_OK:
    st.error(
        f"❌ Required libraries missing: `{_IMPORT_ERR}`\n\n"
        "Add to **requirements.txt**: `pytesseract`, `pdf2image`, `Pillow`\n\n"
        "Add to **packages.txt**: `tesseract-ocr`, `poppler-utils`"
    )
    st.stop()

if not OPENPYXL_OK:
    st.error("❌ `openpyxl` is not installed. Add it to requirements.txt.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# OCR PIPELINE (shared between PDF and Image modes)
# ═════════════════════════════════════════════════════════════════════════════

NUMERIC_RE = re.compile(r"^\(?-?[\d.,\s]+\)?%?$")


def ocr_image_to_words(pil_image: "Image.Image", upscale: int = 2) -> list:
    """Upscale → OCR → return word list with pixel positions."""
    if upscale != 1:
        img = pil_image.resize(
            (pil_image.width * upscale, pil_image.height * upscale),
            Image.LANCZOS,
        )
    else:
        img = pil_image

    # Convert to RGB if needed (handles RGBA screenshots, palette PNGs, etc.)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        if not txt:
            continue
        words.append({
            "text":   txt,
            "left":   data["left"][i]   // upscale,
            "top":    data["top"][i]    // upscale,
            "width":  data["width"][i]  // upscale,
            "height": data["height"][i] // upscale,
        })
    return words


def group_into_rows(words: list, y_tolerance: int = 12) -> list:
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: w["top"])
    rows, current_row = [], [words_sorted[0]]
    current_top = words_sorted[0]["top"]
    for w in words_sorted[1:]:
        if abs(w["top"] - current_top) <= y_tolerance:
            current_row.append(w)
            current_top = sum(x["top"] for x in current_row) / len(current_row)
        else:
            rows.append(current_row)
            current_row = [w]
            current_top = w["top"]
    rows.append(current_row)
    rows = [sorted(r, key=lambda w: w["left"]) for r in rows]
    rows.sort(key=lambda r: min(w["top"] for w in r))
    return rows


def merge_label_words(row_words: list, gap_threshold: int = 25) -> list:
    if not row_words:
        return []
    merged = []
    cur_text  = row_words[0]["text"]
    cur_left  = row_words[0]["left"]
    cur_right = row_words[0]["left"] + row_words[0]["width"]
    prev_right = cur_right
    for w in row_words[1:]:
        gap = w["left"] - prev_right
        if gap <= gap_threshold:
            cur_text  += " " + w["text"]
            cur_right  = w["left"] + w["width"]
        else:
            merged.append({"text": cur_text, "left": cur_left, "right": cur_right})
            cur_text  = w["text"]
            cur_left  = w["left"]
            cur_right = w["left"] + w["width"]
        prev_right = cur_right
    merged.append({"text": cur_text, "left": cur_left, "right": cur_right})
    return merged


def detect_column_bands(all_rows: list, gap_threshold: int = 60) -> list:
    lefts = sorted(cell["left"] for row in all_rows for cell in row)
    if not lefts:
        return []
    bands, band_start, prev = [], lefts[0], lefts[0]
    for x in lefts[1:]:
        if x - prev > gap_threshold:
            bands.append((band_start, prev))
            band_start = x
        prev = x
    bands.append((band_start, prev))
    return bands


def assign_cell_to_band(cell: dict, bands: list) -> int:
    best_idx, best_dist = 0, None
    for idx, (start, end) in enumerate(bands):
        if start - 40 <= cell["left"] <= end + 200:
            dist = abs(cell["left"] - start)
            if best_dist is None or dist < best_dist:
                best_dist, best_idx = dist, idx
    return best_idx


def words_to_grid(words: list, merge_gap: int = 25, col_gap: int = 60) -> list:
    """Full pipeline: words → rows → merged cells → column-aligned grid."""
    rows = group_into_rows(words)
    merged_rows = [merge_label_words(r, gap_threshold=merge_gap) for r in rows]
    bands = detect_column_bands(merged_rows, gap_threshold=col_gap)
    if not bands:
        return []
    grid = []
    for mr in merged_rows:
        grow = [""] * len(bands)
        for cell in mr:
            bi = assign_cell_to_band(cell, bands)
            ex = grow[bi]
            grow[bi] = (ex + " " + cell["text"]).strip() if ex else cell["text"]
        grid.append(grow)
    return grid


def clean_numeric_cell(text: str):
    raw = text.strip()
    if raw in ("", "-", "–", "—"):
        return raw
    letters_only = re.sub(r"[\d\s,.()\-]", "", raw)
    if letters_only and all(ch in "Oo" for ch in letters_only):
        raw = re.sub(r"[Oo]", "0", raw)
    candidate = raw.replace(" ", "")
    is_negative = candidate.startswith("(") and candidate.endswith(")")
    if is_negative:
        candidate = candidate[1:-1]
    if NUMERIC_RE.match(raw.replace(" ", "")) or re.match(r"^[\d,]+(\.\d+)?$", candidate):
        cleaned = candidate.replace(",", "")
        try:
            value = float(cleaned)
            if is_negative:
                value = -value
            if value == int(value):
                value = int(value)
            return value
        except ValueError:
            return raw
    return raw


def clean_grid(grid: list) -> list:
    return [[clean_numeric_cell(cell) for cell in row] for row in grid]


def flag_risky_cells(grid: list) -> list:
    """Flag negative numbers starting with '4' — known Tesseract misread pattern."""
    warnings = [""] * len(grid)
    for r_idx, row in enumerate(grid):
        flagged = []
        for c_idx, val in enumerate(row):
            if isinstance(val, (int, float)) and val < 0:
                if str(int(abs(val))).startswith("4"):
                    flagged.append(c_idx + 1)
        if flagged:
            warnings[r_idx] = (
                f"SPOT-CHECK: negative in col(s) {flagged} starts with '4' "
                f"— Tesseract sometimes misreads '1' as '4' after '('."
            )
    return warnings


def write_grid_to_sheet(ws, grid: list, sheet_title: str, warnings=None):
    ws.title = sheet_title[:31]
    n_cols = max((len(r) for r in grid), default=0)
    warn_col = n_cols + 2
    for r_idx, row in enumerate(grid, start=1):
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value if value != "" else None)
            if isinstance(value, (int, float)):
                cell.number_format = "#,##0;(#,##0)"
                cell.alignment = Alignment(horizontal="right")
        if warnings and warnings[r_idx - 1]:
            wcell = ws.cell(row=r_idx, column=warn_col, value=warnings[r_idx - 1])
            wcell.font = Font(color="9C0006", bold=True)
    for r_idx in range(1, min(4, len(grid) + 1)):
        for c_idx in range(1, n_cols + 1):
            ws.cell(row=r_idx, column=c_idx).font = Font(bold=True)
    if grid:
        for c_idx in range(1, n_cols + 1):
            max_len = max(
                (len(str(row[c_idx - 1])) for row in grid if c_idx - 1 < len(row)),
                default=10,
            )
            ws.column_dimensions[get_column_letter(c_idx)].width = min(max(max_len + 2, 10), 45)
        if warnings and any(warnings):
            ws.column_dimensions[get_column_letter(warn_col)].width = 65


def grid_to_excel_bytes(sheets: list) -> bytes:
    """
    sheets: list of (sheet_title, grid, warnings)
    Returns Excel file as bytes.
    """
    wb = Workbook()
    wb.remove(wb.active)
    for title, grid, warnings in sheets:
        ws = wb.create_sheet()
        write_grid_to_sheet(ws, grid, title, warnings)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════════════════════
# MODE SELECTOR
# ═════════════════════════════════════════════════════════════════════════════

section("🔀 Choose Conversion Mode")
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

mode = st.radio(
    "Conversion mode",
    ["📄 PDF (scanned multi-page document)", "🖼️ Image / Screenshot (PNG, JPG, WEBP …)"],
    horizontal=True,
    label_visibility="collapsed",
)

is_image_mode = mode.startswith("🖼️")

if is_image_mode:
    st.markdown("""
    <div class="fbc-info-card">
        <div class="fbc-info-card-title">🖼️ Image / Screenshot Mode</div>
        <div class="fbc-info-card-body">
            Upload any screenshot or photo of a financial table.
            Supported formats: <b>PNG, JPG / JPEG, WEBP, BMP, TIFF</b>.<br>
            Each image becomes one sheet in the output Excel.
            You can upload multiple images at once — useful if you have
            separate screenshots for Income Statement, Balance Sheet, and Cash Flow.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="fbc-info-card">
        <div class="fbc-info-card-title">📄 PDF Mode</div>
        <div class="fbc-info-card-body">
            Upload one or more scanned PDF financial statements.
            Each page is rendered at the chosen DPI and run through OCR.
            The output Excel has one sheet per PDF page.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="fbc-warn-card">
    <div class="fbc-warn-title">⚠️ Always spot-check the output</div>
    <div class="fbc-warn-body">
        OCR is not 100% accurate. Tesseract can occasionally misread digits — especially
        "1" as "4" after an opening parenthesis in negative numbers. Cells flagged in red
        in the Excel output should be verified against the original source. Never feed
        unchecked figures directly into a valuation model.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SHARED OPTIONS
# ═════════════════════════════════════════════════════════════════════════════

section("⚙️ OCR Options")
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

col_o1, col_o2, col_o3, col_o4 = st.columns(4)

with col_o1:
    upscale_choice = st.selectbox(
        "Upscale factor before OCR",
        [1, 2, 3],
        index=1,
        help="2× fixes most digit misreads. 3× is slower with diminishing returns.",
    )

with col_o2:
    dpi_choice = st.selectbox(
        "Render DPI (PDF only)",
        [200, 300, 400],
        index=1,
        help="300 DPI is the tested sweet-spot for financial statements.",
        disabled=is_image_mode,
    )

with col_o3:
    merge_gap = st.slider(
        "Word merge gap (px)",
        min_value=5, max_value=80, value=25,
        help="Words closer than this are joined into one cell. "
             "Increase if label words split; decrease if columns merge together.",
    )

with col_o4:
    col_band_gap = st.slider(
        "Column band gap (px)",
        min_value=20, max_value=200, value=60,
        help="Minimum gap that separates two different columns. "
             "Increase if adjacent column numbers land in the same cell.",
    )

st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# IMAGE MODE
# ═════════════════════════════════════════════════════════════════════════════
if is_image_mode:
    section("🖼️ Upload Screenshots / Images")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    st.markdown(
        "<div style='color:#475569;font-size:13px;font-style:italic;margin-bottom:12px;'>"
        "Tip: For the DCF model you need 3 sheets — upload your IS, BS and CF screenshots "
        "together so they all end up in one Excel file."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_images = st.file_uploader(
        "Upload image(s)",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_images:
        st.info("⬆️ Upload one or more images to get started.")
        st.stop()

    # Sheet naming: allow user to label each image
    section("🏷️ Name Each Sheet")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    st.markdown(
        "<div style='color:#374151;font-size:14px;margin-bottom:10px;'>"
        "Give each image a meaningful sheet name in the output Excel "
        "(e.g. <b>Income Statement</b>, <b>Balance Sheet</b>, <b>Cash Flow</b>)."
        "</div>",
        unsafe_allow_html=True,
    )

    sheet_names = []
    preview_cols = st.columns(min(len(uploaded_images), 3))
    for i, img_file in enumerate(uploaded_images):
        with preview_cols[i % 3]:
            # Preview thumbnail
            try:
                pil_preview = Image.open(img_file)
                img_file.seek(0)
                st.image(pil_preview, use_container_width=True, caption=img_file.name)
            except Exception:
                st.caption(img_file.name)

            default_name = (
                "Income Statement" if i == 0 else
                "Balance Sheet"    if i == 1 else
                "Cash Flow"        if i == 2 else
                f"Sheet {i + 1}"
            )
            sname = st.text_input(
                f"Sheet name for image {i + 1}",
                value=default_name,
                key=f"img_sheet_name_{i}",
                label_visibility="collapsed",
            )
            sheet_names.append(sname.strip() or f"Sheet {i + 1}")

    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_img_ocr = st.button("▶️ Run OCR & Convert", type="primary", use_container_width=True)
    with col_info:
        st.markdown(
            f"<div style='padding-top:10px;color:#5a7099;font-style:italic;font-size:14px;'>"
            f"{len(uploaded_images)} image(s) queued · {upscale_choice}× upscale"
            f"</div>",
            unsafe_allow_html=True,
        )

    if not run_img_ocr:
        st.stop()

    # ── Image conversion ──────────────────────────────────────────────────────
    section("🔬 Converting…")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    sheets_data = []
    overall_bar = st.progress(0.0, text="Starting…")

    for i, img_file in enumerate(uploaded_images):
        sname = sheet_names[i]
        status_el = st.empty()
        status_el.markdown(
            f"<span style='color:#003399;font-weight:700;'>"
            f"OCR — {img_file.name} → '{sname}'…</span>",
            unsafe_allow_html=True,
        )
        try:
            pil_img = Image.open(img_file)
            words   = ocr_image_to_words(pil_img, upscale=upscale_choice)
            grid    = words_to_grid(words, merge_gap=merge_gap, col_gap=col_band_gap)
            grid    = clean_grid(grid)
            grid    = [row for row in grid if any(str(c).strip() for c in row)]
            warns   = flag_risky_cells(grid)
            sheets_data.append((sname, grid, warns))
            n_rows = len(grid)
            n_cols = max((len(r) for r in grid), default=0)
            status_el.success(
                f"✅ '{sname}' — {n_rows} rows × {n_cols} columns detected"
            )
        except Exception as exc:
            status_el.error(f"❌ {img_file.name} failed: {exc}")

        overall_bar.progress(
            (i + 1) / len(uploaded_images),
            text=f"Processed {i + 1}/{len(uploaded_images)} image(s)",
        )

    overall_bar.progress(1.0, text="Done")

    if not sheets_data:
        st.error("❌ No images could be converted. Check the errors above.")
        st.stop()

    # Build combined Excel
    excel_bytes = grid_to_excel_bytes(sheets_data)
    excel_name  = "FBC_Screenshot_Extract.xlsx"

    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    section("⬇️ Download Result")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    st.success(f"✅ {len(sheets_data)} sheet(s) extracted into one Excel file.")

    col_dl, col_meta = st.columns([1, 2])
    with col_dl:
        st.markdown(f"""
        <div class="result-card">
            <div style="font-family:'Playfair Display',serif;font-weight:700;
                        color:#001a5c;font-size:15px;margin-bottom:4px;">
                📊 {excel_name}
            </div>
            <div style="color:#5a7099;font-size:13px;font-style:italic;">
                {len(sheets_data)} sheet(s) ·
                {", ".join(s[0] for s in sheets_data)} ·
                {len(excel_bytes)//1024:,} KB
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Excel",
            data=excel_bytes,
            file_name=excel_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with col_meta:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#f0f5ff,#fffdf0);
                    border:1px solid rgba(0,51,153,0.12);
                    border-left:5px solid #f5b400;border-radius:14px;
                    padding:18px 22px;">
            <div style="font-family:'Playfair Display',serif;font-size:16px;
                        font-weight:800;color:#001a5c;margin-bottom:8px;">
                💡 Next Steps
            </div>
            <div style="color:#374151;font-size:14px;line-height:1.9;">
                <b>1.</b> Download and open the Excel.<br>
                <b>2.</b> Verify numbers against your screenshots (check red-flagged cells).<br>
                <b>3.</b> The DCF model expects <b>Sheet 1 = Income Statement</b>,
                    <b>Sheet 2 = Balance Sheet</b>, <b>Sheet 3 = Cash Flow</b>,
                    each with an <i>Item</i> column + one column per year.<br>
                <b>4.</b> Go to <b>📊 DCF Model</b> and upload the cleaned Excel.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PDF MODE
# ═════════════════════════════════════════════════════════════════════════════
else:
    if not PDF_LIBS_OK:
        st.error(
            "❌ `pdf2image` is not installed. Add it to requirements.txt "
            "and `poppler-utils` to packages.txt."
        )
        st.stop()

    section("📤 Upload Scanned PDF(s)")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    uploaded_pdfs = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_pdfs:
        st.info(
            "⬆️ Upload one or more scanned PDF files to begin. "
            "Once converted, download the Excel and upload it into the DCF Model page."
        )
        st.stop()

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_ocr = st.button("▶️ Run OCR & Convert", type="primary", use_container_width=True)
    with col_info:
        st.markdown(
            f"<div style='padding-top:10px;color:#5a7099;font-style:italic;font-size:14px;'>"
            f"{len(uploaded_pdfs)} PDF(s) queued · DPI {dpi_choice} · {upscale_choice}× upscale"
            f"</div>",
            unsafe_allow_html=True,
        )

    if not run_ocr:
        st.stop()

    results = []
    errors  = []
    overall_bar = st.progress(0.0, text="Starting…")

    for pdf_i, pdf_file in enumerate(uploaded_pdfs):
        pdf_name   = pdf_file.name
        base_name  = os.path.splitext(pdf_name)[0]
        excel_name = base_name + ".xlsx"

        st.markdown(
            f"<div style='font-family:Playfair Display,serif;font-size:17px;"
            f"font-weight:700;color:#001a5c;margin:18px 0 6px 0;'>"
            f"📄 {pdf_name}</div>",
            unsafe_allow_html=True,
        )

        page_bar    = st.progress(0.0, text="Rendering pages…")
        page_status = st.empty()

        try:
            from pdf2image import convert_from_bytes as _cfb
            images  = _cfb(pdf_file.getvalue(), dpi=dpi_choice, poppler_path=None)
            n_pages = len(images)
            sheets_data = []

            for page_idx, image in enumerate(images, start=1):
                page_status.markdown(
                    f"<span style='color:#003399;font-weight:700;'>"
                    f"OCR — page {page_idx}/{n_pages}</span>",
                    unsafe_allow_html=True,
                )
                page_bar.progress(page_idx / n_pages, text=f"Page {page_idx}/{n_pages}")

                words = ocr_image_to_words(image, upscale=upscale_choice)
                grid  = words_to_grid(words, merge_gap=merge_gap, col_gap=col_band_gap)
                grid  = clean_grid(grid)
                grid  = [row for row in grid if any(str(c).strip() for c in row)]
                warns = flag_risky_cells(grid)
                sheets_data.append((f"Page {page_idx}", grid, warns))

            excel_bytes = grid_to_excel_bytes(sheets_data)
            results.append((excel_name, excel_bytes, n_pages))

            page_bar.progress(1.0, text="✅ Done")
            page_status.success(f"✅ Converted {n_pages} page(s) → {excel_name}")

        except Exception as exc:
            errors.append((pdf_name, str(exc)))
            page_status.error(f"❌ Failed: {exc}")

        overall_bar.progress(
            (pdf_i + 1) / len(uploaded_pdfs),
            text=f"Processed {pdf_i + 1}/{len(uploaded_pdfs)} files",
        )

    overall_bar.progress(1.0, text="All files processed")

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    section("⬇️ Download Results")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    if results:
        st.success(f"✅ {len(results)} file(s) converted successfully.")
        cols = st.columns(min(len(results), 3))
        for i, (excel_name, excel_bytes, n_pages) in enumerate(results):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-family:'Playfair Display',serif;font-weight:700;
                                color:#001a5c;font-size:15px;margin-bottom:4px;">
                        📊 {excel_name}
                    </div>
                    <div style="color:#5a7099;font-size:13px;font-style:italic;">
                        {n_pages} sheet(s) · {len(excel_bytes)//1024:,} KB
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.download_button(
                    "⬇️ Download",
                    data=excel_bytes,
                    file_name=excel_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_pdf_{i}",
                    use_container_width=True,
                )

    if errors:
        st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
        st.error(f"❌ {len(errors)} file(s) failed:")
        for fname, err in errors:
            st.markdown(
                f"<div style='color:#991b1b;font-weight:700;padding:6px 0;'>"
                f"• <b>{fname}</b>: {err}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("""
    <div style="background:linear-gradient(135deg,#f0f5ff,#fffdf0);
                border:1px solid rgba(0,51,153,0.12);
                border-left:5px solid #f5b400;border-radius:14px;
                padding:18px 22px;margin-top:16px;">
        <div style="font-family:'Playfair Display',serif;font-size:17px;font-weight:800;
                    color:#001a5c;margin-bottom:8px;">
            💡 Next Step — Upload into the DCF Model
        </div>
        <div style="color:#374151;font-size:14px;line-height:1.9;">
            <b>1.</b> Download and open the Excel.<br>
            <b>2.</b> Verify numbers against the original PDF (check red-flagged cells).<br>
            <b>3.</b> The DCF page expects <b>Sheet 1 = Income Statement</b>,
                <b>Sheet 2 = Balance Sheet</b>, <b>Sheet 3 = Cash Flow</b>,
                each with an <i>Item</i> column + one column per year.<br>
            <b>4.</b> Head to the <b>📊 DCF Model</b> page and upload the cleaned Excel.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fbc-footer">
    Powered by <b>FBC Securities</b> · Investment Research &amp; Valuation Dashboard
</div>
""", unsafe_allow_html=True)
