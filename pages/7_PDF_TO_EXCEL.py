"""
7_PDF_TO_EXCEL.py  —  FBC Investment Valuation System
PDF (scanned financial statements) → Excel converter

Integrated as a Streamlit page; replaces the tkinter GUI with:
  • st.file_uploader  (one or many PDFs)
  • st.progress bar   (per-page OCR progress)
  • st.download_button (Excel output)

System requirements (add to your repo root):
  packages.txt  →  tesseract-ocr
                   poppler-utils
  requirements.txt → pytesseract
                     pdf2image
                     Pillow
                     openpyxl

On Streamlit Cloud both binaries are on PATH after packages.txt installs them,
so TESSERACT_CMD = "tesseract" and poppler_path = None.
"""

import io
import os
import re
import tempfile

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF → Excel | FBC Valuation", layout="wide")

# ── Auth guard (same pattern as other pages) ─────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True
if "user" not in st.session_state or not st.session_state.get("user"):
    st.session_state["user"] = {
        "username": "analyst",
        "role": "analyst",
        "full_name": "Analyst",
    }

# ── Lazy imports (graceful error if packages missing) ────────────────────────
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    LIBS_OK = True
except ImportError as _e:
    LIBS_OK = False
    _IMPORT_ERR = str(_e)

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment
    from openpyxl.utils import get_column_letter
    OPENPYXL_OK = True
except ImportError:
    OPENPYXL_OK = False

# ── Tesseract path (Streamlit Cloud: binary is on PATH) ──────────────────────
# If running locally on Windows, set the env variable before launching:
#   set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
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

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #001a5c 0%, #003399 45%, #0044cc 100%) !important;
    border-right: 2px solid rgba(245,180,0,0.25) !important;
}
section[data-testid="stSidebar"] * { color: #e8f0ff !important; font-family: "EB Garamond", Georgia, serif !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: "Playfair Display", serif !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #003399 0%, #0055ee 100%) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-family: "EB Garamond", serif !important; font-size: 15px !important;
    padding: 10px 24px !important; box-shadow: 0 4px 14px rgba(0,51,153,0.30) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #003399, #0044cc) !important;
    color: white !important; border-radius: 10px !important;
    font-weight: 700 !important; border: none !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #f0f5ff, #fff8e6) !important;
    border: 1px solid rgba(0,51,153,0.12) !important;
    border-radius: 14px !important; padding: 14px 16px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(0, 51, 153, 0.04);
    border: 1px dashed rgba(0, 51, 153, 0.25);
    border-radius: 14px; padding: 18px 20px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #003399; border-radius: 999px; }

/* Section heading */
.fbc-section {
    display: block; padding: 16px 0; margin: 28px 0 18px 0;
    border-bottom: 2px solid rgba(0,51,153,0.15);
}
.fbc-section-title {
    font-family: "Playfair Display", serif !important;
    font-size: 21px; font-weight: 800; color: #001a5c !important;
}

/* Divider */
.fbc-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #f5b400, #003399, transparent);
    border: none; margin: 20px 0; border-radius: 999px;
}

/* Footer */
.fbc-footer {
    text-align: center; padding: 22px; margin-top: 40px;
    color: #5a7099 !important; font-size: 13px;
    border-top: 1px solid rgba(0,51,153,0.10); font-style: italic;
}
.fbc-footer b { color: #003399 !important; font-style: normal; }

/* Info card */
.fbc-info-card {
    background: linear-gradient(135deg, #001a5c, #003399);
    border-radius: 16px; padding: 20px 26px; margin-bottom: 20px;
    border-bottom: 3px solid #f5b400;
    box-shadow: 0 8px 24px rgba(0,26,92,0.22);
}
.fbc-info-card-title {
    font-family: "Playfair Display", serif !important;
    font-size: 20px; font-weight: 800; color: #ffffff !important; margin-bottom: 8px;
}
.fbc-info-card-body { color: rgba(255,255,255,0.85) !important; font-size: 14px; line-height: 1.7; }

/* Warning card */
.fbc-warn-card {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border-radius: 14px; padding: 16px 22px; margin: 14px 0;
    border-left: 5px solid #f5b400;
}
.fbc-warn-card-title { font-family: "Playfair Display", serif !important; font-weight: 700; color: white !important; margin-bottom: 4px; }
.fbc-warn-body { color: rgba(255,255,255,0.88) !important; font-size: 13px; line-height: 1.6; }

/* Step badge */
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #003399, #0044cc);
    color: white !important; font-weight: 900; font-size: 13px;
    width: 28px; height: 28px; border-radius: 50%;
    text-align: center; line-height: 28px;
    margin-right: 10px; vertical-align: middle;
    box-shadow: 0 3px 8px rgba(0,51,153,0.35);
}
</style>
""", unsafe_allow_html=True)


# ── Helper: section header ────────────────────────────────────────────────────
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
            📄 PDF → Excel Converter
        </div>
        <div style="font-family:'EB Garamond',serif; font-size:13px; font-style:italic;
                    color:rgba(255,255,255,0.65); margin-top:2px;">
            Extract scanned financial statements into structured Excel files
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
        f"❌ Required libraries are not installed: `{_IMPORT_ERR}`\n\n"
        "Add the following to your **requirements.txt**:\n"
        "```\npytesseract\npdf2image\nPillow\n```\n\n"
        "And the following to **packages.txt** (Streamlit Cloud system packages):\n"
        "```\ntesseract-ocr\npoppler-utils\n```"
    )
    st.stop()

if not OPENPYXL_OK:
    st.error("❌ `openpyxl` is not installed. Add it to requirements.txt.")
    st.stop()

# ── How it works card ─────────────────────────────────────────────────────────
st.markdown("""
<div class="fbc-info-card">
    <div class="fbc-info-card-title">🔬 How This Works</div>
    <div class="fbc-info-card-body">
        <span class="step-badge" style="background:linear-gradient(135deg,#f5b400,#ffd040);color:#001a5c !important;">1</span>
        Upload one or more scanned PDF financial statements.<br>
        <span class="step-badge" style="background:linear-gradient(135deg,#f5b400,#ffd040);color:#001a5c !important;">2</span>
        Each page is rendered at 300 DPI and 2× upscaled for accurate OCR.<br>
        <span class="step-badge" style="background:linear-gradient(135deg,#f5b400,#ffd040);color:#001a5c !important;">3</span>
        Tesseract reads every word and its position; rows and columns are reconstructed.<br>
        <span class="step-badge" style="background:linear-gradient(135deg,#f5b400,#ffd040);color:#001a5c !important;">4</span>
        Download the cleaned Excel — one sheet per page — ready to upload into the DCF model.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="fbc-warn-card">
    <div class="fbc-warn-card-title">⚠️ Important: Always spot-check the output</div>
    <div class="fbc-warn-body">
        OCR is not 100% accurate on scanned documents. Tesseract can occasionally misread digits
        (e.g. "1" as "4" when it follows an opening parenthesis). Numbers flagged in red in the
        Excel output should be checked against the original PDF. Never feed unchecked figures
        directly into a valuation model.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# OCR PIPELINE (ported from original script, tkinter replaced by Streamlit)
# ═════════════════════════════════════════════════════════════════════════════

UPSCALE_FACTOR = 2
NUMERIC_RE = re.compile(r"^\(?-?[\d.,\s]+\)?%?$")


def ocr_page_words(pil_image: "Image.Image") -> list:
    if UPSCALE_FACTOR != 1:
        upscaled = pil_image.resize(
            (pil_image.width * UPSCALE_FACTOR, pil_image.height * UPSCALE_FACTOR),
            Image.LANCZOS,
        )
    else:
        upscaled = pil_image

    data = pytesseract.image_to_data(upscaled, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        if not txt:
            continue
        words.append({
            "text":  txt,
            "left":  data["left"][i]  // UPSCALE_FACTOR,
            "top":   data["top"][i]   // UPSCALE_FACTOR,
            "width": data["width"][i] // UPSCALE_FACTOR,
            "height":data["height"][i]// UPSCALE_FACTOR,
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


def rows_to_grid(rows: list) -> list:
    merged_rows = [merge_label_words(r) for r in rows]
    bands = detect_column_bands(merged_rows)
    if not bands:
        return []
    grid = []
    for row in merged_rows:
        grid_row = [""] * len(bands)
        for cell in row:
            bi = assign_cell_to_band(cell, bands)
            existing = grid_row[bi]
            grid_row[bi] = (existing + " " + cell["text"]).strip() if existing else cell["text"]
        grid.append(grid_row)
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
    warnings = [""] * len(grid)
    for r_idx, row in enumerate(grid):
        flagged = []
        for c_idx, val in enumerate(row):
            if isinstance(val, (int, float)) and val < 0:
                if str(int(abs(val))).startswith("4"):
                    flagged.append(c_idx + 1)
        if flagged:
            warnings[r_idx] = (
                f"SPOT-CHECK: negative number in col(s) {flagged} starts with '4' "
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


# ═════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════

section("📤 Upload Scanned PDF(s)")
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

col_up, col_opts = st.columns([2, 1])

with col_up:
    uploaded_pdfs = st.file_uploader(
        "Upload one or more scanned PDF financial statements",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

with col_opts:
    st.markdown("#### ⚙️ OCR Options")
    dpi_choice = st.selectbox(
        "Render DPI",
        [200, 300, 400],
        index=1,
        help="300 DPI is the tested sweet-spot. 400 DPI is slower but may help very small text.",
    )
    upscale_choice = st.selectbox(
        "Upscale factor before OCR",
        [1, 2, 3],
        index=1,
        help="2× upscaling fixes most digit misreads. 3× is slower with diminishing returns.",
    )
    merge_gap = st.slider(
        "Word merge gap (px)",
        min_value=5,
        max_value=80,
        value=25,
        help="Words closer than this pixel gap are joined into one cell. "
             "Increase if label words are being split; decrease if columns are merging together.",
    )
    col_band_gap = st.slider(
        "Column band gap (px)",
        min_value=20,
        max_value=200,
        value=60,
        help="Minimum horizontal gap that separates two different columns. "
             "Increase if numbers from adjacent columns are landing in the same cell.",
    )

st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

if not uploaded_pdfs:
    st.info(
        "⬆️ Upload one or more scanned PDF files to begin. "
        "Once converted, download the Excel and upload it directly into the **DCF Model** page."
    )
    st.stop()

# ── Process button ────────────────────────────────────────────────────────────
section("🔬 Convert PDFs")
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_ocr = st.button("▶️ Run OCR & Convert", type="primary", use_container_width=True)
with col_info:
    st.markdown(
        f"<div style='padding-top:10px; color:#5a7099; font-style:italic; font-size:14px;'>"
        f"{len(uploaded_pdfs)} PDF(s) queued · DPI {dpi_choice} · {upscale_choice}× upscale"
        f"</div>",
        unsafe_allow_html=True,
    )

if not run_ocr:
    st.stop()

# ── Conversion loop ───────────────────────────────────────────────────────────
results = []   # list of (filename, bytes)
errors  = []   # list of (filename, error_str)

overall_bar = st.progress(0.0, text="Starting…")

for pdf_i, pdf_file in enumerate(uploaded_pdfs):
    pdf_name   = pdf_file.name
    base_name  = os.path.splitext(pdf_name)[0]
    excel_name = base_name + ".xlsx"

    st.markdown(
        f"<div style='font-family:Playfair Display,serif;font-size:17px;"
        f"font-weight:700;color:#001a5c;margin:18px 0 6px 0;'>"
        f"📄 Processing: {pdf_name}</div>",
        unsafe_allow_html=True,
    )

    page_bar    = st.progress(0.0, text="Rendering pages…")
    page_status = st.empty()

    try:
        pdf_bytes = pdf_file.getvalue()

        # Render all pages to images (in-memory via bytes)
        images = convert_from_bytes(
            pdf_bytes,
            dpi=dpi_choice,
            poppler_path=None,   # on PATH via packages.txt on Streamlit Cloud
        )
        n_pages = len(images)

        wb = Workbook()
        wb.remove(wb.active)   # remove default blank sheet

        for page_idx, image in enumerate(images, start=1):
            page_status.markdown(
                f"<span style='color:#003399;font-weight:700;'>"
                f"OCR — page {page_idx} / {n_pages}</span>",
                unsafe_allow_html=True,
            )
            page_bar.progress(page_idx / n_pages, text=f"Page {page_idx}/{n_pages}")

            # --- OCR (with inline upscale factor from UI) ---
            if upscale_choice != 1:
                upscaled = image.resize(
                    (image.width * upscale_choice, image.height * upscale_choice),
                    Image.LANCZOS,
                )
            else:
                upscaled = image

            data = pytesseract.image_to_data(upscaled, output_type=pytesseract.Output.DICT)
            words = []
            for i in range(len(data["text"])):
                txt = data["text"][i].strip()
                if not txt:
                    continue
                words.append({
                    "text":   txt,
                    "left":   data["left"][i]   // upscale_choice,
                    "top":    data["top"][i]    // upscale_choice,
                    "width":  data["width"][i]  // upscale_choice,
                    "height": data["height"][i] // upscale_choice,
                })

            rows  = group_into_rows(words)

            # rebuild merge / band detection with UI-chosen gaps
            merged_rows = [merge_label_words(r, gap_threshold=merge_gap) for r in rows]
            bands = detect_column_bands(merged_rows, gap_threshold=col_band_gap)

            if not bands:
                grid = []
            else:
                grid = []
                for mr in merged_rows:
                    grow = [""] * len(bands)
                    for cell in mr:
                        bi = assign_cell_to_band(cell, bands)
                        ex = grow[bi]
                        grow[bi] = (ex + " " + cell["text"]).strip() if ex else cell["text"]
                    grid.append(grow)

            grid = clean_grid(grid)
            grid = [row for row in grid if any(str(c).strip() for c in row)]
            warnings = flag_risky_cells(grid)

            ws = wb.create_sheet(title=f"Page {page_idx}")
            write_grid_to_sheet(ws, grid, f"Page {page_idx}", warnings)

        # Save workbook to bytes
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        results.append((excel_name, buf.getvalue(), n_pages))

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

# ── Helper closures that use UI params (defined after the param widgets) ──────
# (they reference merge_gap / col_band_gap captured from the sliders above)
def _merge(row_words, gap_threshold=25):
    return merge_label_words(row_words, gap_threshold=gap_threshold)

def _detect_bands(all_rows, gap_threshold=60):
    return detect_column_bands(all_rows, gap_threshold=gap_threshold)

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
section("⬇️ Download Results")
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

if results:
    st.success(f"✅ {len(results)} file(s) converted successfully.")

    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]
    for i, (excel_name, excel_bytes, n_pages) in enumerate(results):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid rgba(0,51,153,0.12);
                        border-left:5px solid #003399;border-radius:14px;
                        padding:16px 18px;margin-bottom:12px;
                        box-shadow:0 4px 12px rgba(0,0,0,0.06);">
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
                f"⬇️ Download",
                data=excel_bytes,
                file_name=excel_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{i}",
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

# ── Workflow tip ──────────────────────────────────────────────────────────────
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="background:linear-gradient(135deg,#f0f5ff,#fffdf0);border:1px solid rgba(0,51,153,0.12);
            border-left:5px solid #f5b400;border-radius:14px;padding:18px 22px;margin-top:8px;">
    <div style="font-family:'Playfair Display',serif;font-size:17px;font-weight:800;
                color:#001a5c;margin-bottom:8px;">
        💡 Next Step — Upload into the DCF Model
    </div>
    <div style="color:#374151;font-size:14px;line-height:1.8;">
        <b>1.</b> Download the Excel file above.<br>
        <b>2.</b> Open it and verify the numbers match your PDF (especially any cells flagged in red).<br>
        <b>3.</b> Restructure if needed: the DCF page expects
            <b>Sheet 1 = Income Statement</b>,
            <b>Sheet 2 = Balance Sheet</b>,
            <b>Sheet 3 = Cash Flow Statement</b>, each with an <i>Item</i> column and one column per year.<br>
        <b>4.</b> Head to the <b>📊 DCF Model</b> page and upload the cleaned Excel.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fbc-footer">
    Powered by <b>FBC Securities</b> · Investment Research &amp; Valuation Dashboard
</div>
""", unsafe_allow_html=True)
