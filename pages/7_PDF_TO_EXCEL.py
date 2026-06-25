import io, os, re
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF / Image → Excel | FBC", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True
if "user" not in st.session_state or not st.session_state.get("user"):
    st.session_state["user"] = {"username":"analyst","role":"analyst","full_name":"Analyst"}

# ── Lazy imports ──────────────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance
    LIBS_OK = True; _IMPORT_ERR = ""
except ImportError as _e:
    LIBS_OK = False; _IMPORT_ERR = str(_e)

PDF_LIBS_OK = False
if LIBS_OK:
    try:
        from pdf2image import convert_from_bytes
        PDF_LIBS_OK = True
    except ImportError:
        pass

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter
    OPENPYXL_OK = True
except ImportError:
    OPENPYXL_OK = False

_tess_env = os.environ.get("TESSERACT_CMD","")
if LIBS_OK:
    pytesseract.pytesseract.tesseract_cmd = _tess_env if _tess_env else "tesseract"

# ── FBC styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');
html,body,.stApp,.block-container,p,div,label,h1,h2,h3,h4,h5,h6,li,ul,ol,a,small{
  font-family:"EB Garamond",Georgia,"Times New Roman",serif!important;color:#1a1a2e;}
h1,h2,h3,h4,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3{
  font-family:"Playfair Display",Georgia,serif!important;font-weight:700!important;}
.stApp{background:#f5f7fb!important;}
.main .block-container{background:#f5f7fb!important;padding-top:1.5rem!important;}
section[data-testid="stSidebar"]{
  background:linear-gradient(175deg,#001a5c 0%,#003399 45%,#0044cc 100%)!important;
  border-right:2px solid rgba(245,180,0,.25)!important;}
section[data-testid="stSidebar"] *{color:#e8f0ff!important;font-family:"EB Garamond",Georgia,serif!important;}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{
  color:#fff!important;font-family:"Playfair Display",serif!important;font-weight:700!important;}
.stButton>button{
  background:linear-gradient(135deg,#003399,#0055ee)!important;color:#fff!important;
  border:none!important;border-radius:10px!important;font-weight:700!important;
  font-family:"EB Garamond",serif!important;font-size:15px!important;padding:10px 24px!important;
  box-shadow:0 4px 14px rgba(0,51,153,.30)!important;transition:all .2s ease!important;}
.stButton>button:hover{transform:translateY(-2px)!important;}
.stButton>button *{color:#fff!important;}
.stDownloadButton>button{
  background:linear-gradient(135deg,#003399,#0044cc)!important;color:#fff!important;
  border-radius:10px!important;font-weight:700!important;border:none!important;}
.stDownloadButton>button *{color:#fff!important;}
[data-testid="stFileUploader"]{
  background:rgba(0,51,153,.04);border:1px dashed rgba(0,51,153,.25);
  border-radius:14px;padding:18px 20px;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-thumb{background:#003399;border-radius:999px;}
.fbc-section{display:block;padding:16px 0;margin:28px 0 18px 0;border-bottom:2px solid rgba(0,51,153,.15);}
.fbc-section-title{font-family:"Playfair Display",serif!important;font-size:21px;font-weight:800;color:#001a5c!important;}
.fbc-divider{height:2px;background:linear-gradient(90deg,transparent,#f5b400,#003399,transparent);
  border:none;margin:20px 0;border-radius:999px;}
.fbc-footer{text-align:center;padding:22px;margin-top:40px;color:#5a7099!important;font-size:13px;
  border-top:1px solid rgba(0,51,153,.10);font-style:italic;}
.fbc-footer b{color:#003399!important;font-style:normal;}
.fbc-info-card{background:linear-gradient(135deg,#001a5c,#003399);border-radius:16px;
  padding:20px 26px;margin-bottom:20px;border-bottom:3px solid #f5b400;
  box-shadow:0 8px 24px rgba(0,26,92,.22);}
.fbc-info-card-title{font-family:"Playfair Display",serif!important;font-size:19px;
  font-weight:800;color:#fff!important;margin-bottom:8px;}
.fbc-info-card-body{color:rgba(255,255,255,.85)!important;font-size:14px;line-height:1.8;}
.fbc-warn-card{background:linear-gradient(135deg,#7f1d1d,#991b1b);border-radius:14px;
  padding:16px 22px;margin:14px 0;border-left:5px solid #f5b400;}
.fbc-warn-title{font-family:"Playfair Display",serif!important;font-weight:700;
  color:#fff!important;margin-bottom:4px;font-size:15px;}
.fbc-warn-body{color:rgba(255,255,255,.88)!important;font-size:13px;line-height:1.6;}
.result-card{background:#fff;border:1px solid rgba(0,51,153,.12);border-left:5px solid #003399;
  border-radius:14px;padding:16px 18px;margin-bottom:12px;box-shadow:0 4px 12px rgba(0,0,0,.06);}
.next-steps{background:linear-gradient(135deg,#f0f5ff,#fffdf0);
  border:1px solid rgba(0,51,153,.12);border-left:5px solid #f5b400;
  border-radius:14px;padding:18px 22px;margin-top:16px;}
.next-steps-title{font-family:"Playfair Display",serif!important;font-size:16px;
  font-weight:800;color:#001a5c;margin-bottom:8px;}
.next-steps-body{color:#374151;font-size:14px;line-height:1.9;}
</style>
""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="fbc-section"><div class="fbc-section-title">{title}</div></div>',
                unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#001233,#001a5c 40%,#003399);border-radius:16px;
  padding:18px 28px;margin-bottom:4px;display:flex;align-items:center;
  justify-content:space-between;box-shadow:0 6px 24px rgba(0,26,92,.32);
  border-bottom:3px solid #f5b400;">
  <div>
    <div style="font-family:'Playfair Display',serif;font-size:22px;font-weight:900;
      color:#fff;letter-spacing:-.01em;text-shadow:0 2px 8px rgba(0,0,0,.30);">
      📄 PDF &amp; Image → Excel Converter
    </div>
    <div style="font-family:'EB Garamond',serif;font-size:13px;font-style:italic;
      color:rgba(255,255,255,.65);margin-top:2px;">
      Financial-statement-aware OCR — columns anchored from header row
    </div>
  </div>
  <div style="background:rgba(245,180,0,.22);border:1.5px solid rgba(245,180,0,.60);
    color:#ffd040;font-size:13px;font-weight:700;padding:6px 18px;border-radius:999px;
    font-family:'EB Garamond',serif;white-space:nowrap;">FBC Securities</div>
</div>
<hr style="border:none;border-top:2px solid #dde6f5;margin:6px 0 20px 0;">
""", unsafe_allow_html=True)

if not LIBS_OK:
    st.error(f"❌ Missing libraries: `{_IMPORT_ERR}`\n\nAdd to requirements.txt: pytesseract, pdf2image, Pillow\nAdd to packages.txt: tesseract-ocr, poppler-utils")
    st.stop()
if not OPENPYXL_OK:
    st.error("❌ openpyxl not installed.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# FINANCIAL-STATEMENT-AWARE OCR PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

NOISE_PAT   = re.compile(r"^[^a-zA-Z0-9()\-.]+$")
NOISE_WORDS = {"it","ot","be","a=","bet","Ml","oe","i","—","~~","1060","MIl","Mi","Ml","—"}


def _preprocess(pil_image, upscale=3):
    """Upscale + contrast + sharpen for best digit accuracy."""
    img = pil_image.convert("RGB")
    img = img.resize((img.width * upscale, img.height * upscale), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = img.filter(ImageFilter.SHARPEN)
    return img, upscale


def _ocr_words(preprocessed_img, upscale):
    """Run Tesseract and return word list with original-scale pixel positions."""
    data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        if not txt:
            continue
        words.append({
            "text":  txt,
            "left":  data["left"][i]  // upscale,
            "top":   data["top"][i]   // upscale,
            "width": data["width"][i] // upscale,
            "right": (data["left"][i] + data["width"][i]) // upscale,
        })
    return words


def _group_rows(words, tol=6):
    """Cluster words into text rows by y-position."""
    if not words:
        return []
    ws = sorted(words, key=lambda w: w["top"])
    rows = [[ws[0]]]; cur_top = ws[0]["top"]
    for w in ws[1:]:
        if abs(w["top"] - cur_top) <= tol:
            rows[-1].append(w)
            cur_top = sum(x["top"] for x in rows[-1]) / len(rows[-1])
        else:
            rows.append([w]); cur_top = w["top"]
    for r in rows:
        r.sort(key=lambda w: w["left"])
    return rows


def _detect_anchors(rows):
    """
    Find column anchors from the header row containing 'Note' and currency labels.
    Returns (note_col, val1_col, val2_col, header_top).
    Falls back to heuristic positions if header not found.
    """
    for r in rows:
        texts_lower = [w["text"].lower() for w in r]
        if "note" in texts_lower:
            note_col   = next(w["left"] for w in r if w["text"].lower() == "note")
            header_top = r[0]["top"]
            val_words  = [w for w in r if w["left"] > note_col + 50]
            if len(val_words) >= 2:
                return note_col, val_words[0]["left"], val_words[-1]["left"], header_top
            elif len(val_words) == 1:
                # single value column document
                return note_col, val_words[0]["left"], None, header_top

    # Fallback: look for two 4-digit year numbers on the same row
    for r in rows:
        year_ws = [w for w in r if re.match(r"^20\d\d$", w["text"])]
        if len(year_ws) >= 2:
            # Estimate note col as ~60% of the way to first year
            note_col = int(year_ws[0]["left"] * 0.5)
            return note_col, year_ws[0]["left"], year_ws[1]["left"], r[0]["top"]

    return None, None, None, None


def _col_of(left, note_col, midpoint):
    """Assign a pixel x-position to a named column."""
    if midpoint is not None and left >= midpoint:
        return "val2"
    if note_col is not None and left >= note_col + 50:
        return "val1"
    if note_col is not None and left >= note_col - 20:
        return "note"
    return "label"


def _is_noise(t):
    return bool(NOISE_PAT.match(t)) or t in NOISE_WORDS


def _clean_note(tokens):
    """Keep only valid note references (e.g. 8.1, 10.2, 12)."""
    return " ".join(t for t in tokens if re.match(r"^\d+\.?\d*$", t))


def _merge_tokens(tokens):
    """
    Join a list of tokens that belong to one number cell.
    Fixes:
      - "$" misread as "8" when followed by digits  e.g. "$31" → "831"
      - "O" misread as "0" inside digit runs
    """
    joined = "".join(tokens)
    joined = re.sub(r"\$(?=\d)", "8", joined)       # $ → 8
    joined = re.sub(r"(?<=\d)[Oo](?=\d)", "0", joined)  # O → 0 between digits
    joined = re.sub(r"[~\-—:=`']+", "", joined)     # strip noise chars
    return joined


def _parse_number(raw: str):
    """
    Convert a raw merged token string to int/float.
    Returns the parsed number, None for blank/dash, or original string if unparseable.
    """
    s = raw.strip()
    if not s or s in ("-", "—", "–", ""):
        return None
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]
    # Remove thousands separators (commas, spaces, dots used as separators)
    s = s.replace(",", "").replace(" ", "")
    # If a dot remains and it separates a 3-digit suffix treat as thousands separator
    if "." in s and len(s.split(".")[-1]) == 3:
        s = s.replace(".", "")
    try:
        v = int(s)
        return -v if negative else v
    except ValueError:
        try:
            v = float(s)
            return -v if negative else v
        except ValueError:
            return raw


def _flag_risky(v):
    """Return True if this value looks like a potential Tesseract '1'→'4' misread."""
    if isinstance(v, (int, float)) and v < 0:
        return str(int(abs(v))).startswith("4")
    return False


def process_image_to_rows(pil_image, upscale=3):
    """
    Full pipeline: PIL image → list of row dicts
    Each dict: {label, note, val1, val2, risky}
    """
    proc, up = _preprocess(pil_image, upscale)
    words = _ocr_words(proc, up)
    rows  = _group_rows(words, tol=6)

    note_col, val1_col, val2_col, header_top = _detect_anchors(rows)

    # Compute midpoint for val1/val2 split, biased slightly left to handle edge tokens
    if val1_col is not None and val2_col is not None:
        midpoint = (val1_col + val2_col) / 2 - 5
    elif val2_col is None and val1_col is not None:
        # single value column
        midpoint = None
    else:
        midpoint = None

    output = []
    for r in rows:
        is_pre_header = (header_top is not None and r[0]["top"] < header_top)

        if is_pre_header:
            # Title / sub-title rows — join all non-noise text as one label
            line = " ".join(w["text"] for w in r if not _is_noise(w["text"]))
            output.append({"label": line, "note":"", "val1":"", "val2":"", "risky": False})
            continue

        buckets = {"label":[], "note":[], "val1":[], "val2":[]}
        for w in r:
            if _is_noise(w["text"]):
                continue
            col = _col_of(w["left"], note_col, midpoint)
            buckets[col].append(w["text"])

        label = " ".join(buckets["label"])
        note  = _clean_note(buckets["note"])

        raw1 = _merge_tokens(buckets["val1"])
        raw2 = _merge_tokens(buckets["val2"])
        v1   = _parse_number(raw1) if raw1 else ""
        v2   = _parse_number(raw2) if raw2 else ""

        risky = _flag_risky(v1) or _flag_risky(v2)
        output.append({"label": label, "note": note, "val1": v1, "val2": v2, "risky": risky})

    return output, {"note_col": note_col, "val1_col": val1_col,
                    "val2_col": val2_col, "header_top": header_top,
                    "midpoint": midpoint}


# ── Excel writer ──────────────────────────────────────────────────────────────

def rows_to_excel_bytes(sheets):
    """
    sheets: list of (sheet_title, row_dicts)
    Each row_dict: {label, note, val1, val2, risky}
    """
    wb = Workbook()
    wb.remove(wb.active)

    BLUE_FILL  = PatternFill("solid", fgColor="071426")
    GOLD_FILL  = PatternFill("solid", fgColor="F5B400")
    LIGHT_FILL = PatternFill("solid", fgColor="F0F5FF")
    RED_FONT   = Font(color="9C0006", bold=True)
    WHITE_FONT = Font(bold=True, color="FFFFFF")
    BOLD       = Font(bold=True)
    RIGHT_ALIGN = Alignment(horizontal="right", vertical="center")
    MONEY_FMT   = "#,##0;(#,##0)"

    for title, row_dicts in sheets:
        ws = wb.create_sheet(title=title[:31])

        # Header
        headers = ["Item", "Note", "2024", "2023"]
        for ci, h in enumerate(headers, 1):
            c = ws.cell(1, ci, h)
            c.font  = WHITE_FONT
            c.fill  = BLUE_FILL
            c.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 20

        for ri, row in enumerate(row_dicts, 2):
            label = row.get("label","")
            note  = row.get("note","")
            v1    = row.get("val1","")
            v2    = row.get("val2","")
            risky = row.get("risky", False)

            # Skip rows that are entirely blank
            if not any([str(label).strip(), str(note).strip(),
                        str(v1).strip(), str(v2).strip()]):
                continue

            ws.cell(ri, 1, label)
            ws.cell(ri, 2, note)

            for ci, val in [(3, v1), (4, v2)]:
                cell = ws.cell(ri, ci)
                if isinstance(val, (int, float)):
                    cell.value          = val
                    cell.number_format  = MONEY_FMT
                    cell.alignment      = RIGHT_ALIGN
                    if risky:
                        cell.font = RED_FONT
                elif val not in ("", None):
                    cell.value = str(val)
                    cell.alignment = RIGHT_ALIGN

            # Zebra stripe
            if ri % 2 == 0:
                for ci in range(1, 5):
                    ws.cell(ri, ci).fill = LIGHT_FILL

            # Bold total-looking rows (all caps label or blank label with values)
            if label.isupper() or (not label and (v1 or v2)):
                for ci in range(1, 5):
                    existing = ws.cell(ri, ci).font
                    ws.cell(ri, ci).font = Font(bold=True,
                                                color=existing.color if existing.color else "000000")

        # Column widths
        ws.column_dimensions["A"].width = 48
        ws.column_dimensions["B"].width = 8
        ws.column_dimensions["C"].width = 16
        ws.column_dimensions["D"].width = 16

        # Freeze header
        ws.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════

section("🔀 Choose Conversion Mode")
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

mode = st.radio(
    "mode",
    ["📄 PDF — scanned multi-page document", "🖼️ Image / Screenshot (PNG, JPG, WEBP …)"],
    horizontal=True,
    label_visibility="collapsed",
)
is_image_mode = mode.startswith("🖼️")

st.markdown("""
<div class="fbc-info-card">
  <div class="fbc-info-card-title">🔬 How This Works</div>
  <div class="fbc-info-card-body">
    This converter uses a <b>financial-statement-aware</b> OCR pipeline — not a generic one.<br>
    It first finds the <b>Note / USS / USS header row</b> in your document and uses those
    pixel positions to anchor the label, note, and two value columns precisely. Numbers split
    across multiple OCR tokens are automatically merged. Known Tesseract digit misreads
    (e.g. <code>$</code>→<code>8</code>, <code>O</code>→<code>0</code>) are corrected.<br>
    Cells flagged in <span style="color:#fca5a5;font-weight:700;">red</span> in the Excel
    output should still be spot-checked against the source.
  </div>
</div>
<div class="fbc-warn-card">
  <div class="fbc-warn-title">⚠️ Always verify before using in a valuation model</div>
  <div class="fbc-warn-body">
    OCR accuracy on scanned documents is typically 98–99% but not 100%.
    One confirmed pattern: Tesseract sometimes misreads a leading "1" as "4" inside
    parenthesised negatives — these are flagged in red automatically.
    Sum each column and compare to the printed totals before uploading to DCF.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

# ── OCR options ───────────────────────────────────────────────────────────────
section("⚙️ OCR Options")
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

col_o1, col_o2, col_o3 = st.columns(3)
with col_o1:
    upscale_choice = st.selectbox("Upscale factor", [2, 3], index=1,
        help="3× is recommended for financial statements. 2× is faster.")
with col_o2:
    dpi_choice = st.selectbox("Render DPI (PDF only)", [200, 300, 400], index=1,
        disabled=is_image_mode,
        help="300 DPI is the tested sweet-spot.")
with col_o3:
    show_debug = st.checkbox("Show column anchor debug info",
        help="Shows detected Note/Val1/Val2 pixel positions for troubleshooting.")

st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE MODE
# ─────────────────────────────────────────────────────────────────────────────
if is_image_mode:
    section("🖼️ Upload Screenshots / Images")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    st.markdown(
        "<div style='color:#475569;font-size:13px;font-style:italic;margin-bottom:12px;'>"
        "Tip: Upload IS, BS and CF screenshots together — they'll become separate named "
        "sheets in one Excel file, ready to drop straight into the DCF model."
        "</div>", unsafe_allow_html=True)

    uploaded_images = st.file_uploader(
        "Upload images", type=["png","jpg","jpeg","webp","bmp","tiff","tif"],
        accept_multiple_files=True, label_visibility="collapsed")

    if not uploaded_images:
        st.info("⬆️ Upload one or more screenshots to begin.")
        st.stop()

    # Sheet naming + preview
    section("🏷️ Name Each Sheet")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#374151;font-size:14px;margin-bottom:10px;'>"
        "Name each sheet to match its statement type. "
        "The DCF model expects <b>Income Statement</b>, <b>Balance Sheet</b>, "
        "<b>Cash Flow</b> as sheet names."
        "</div>", unsafe_allow_html=True)

    sheet_names = []
    default_names = ["Income Statement","Balance Sheet","Cash Flow"]
    prev_cols = st.columns(min(len(uploaded_images), 3))
    for i, img_file in enumerate(uploaded_images):
        with prev_cols[i % 3]:
            try:
                pil_prev = Image.open(img_file); img_file.seek(0)
                st.image(pil_prev, use_container_width=True, caption=img_file.name)
            except Exception:
                st.caption(img_file.name)
            sname = st.text_input(
                f"Sheet name {i+1}",
                value=default_names[i] if i < len(default_names) else f"Sheet {i+1}",
                key=f"img_sheet_name_{i}", label_visibility="collapsed")
            sheet_names.append(sname.strip() or f"Sheet {i+1}")

    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    col_btn, col_inf = st.columns([1,3])
    with col_btn:
        run_img = st.button("▶️ Run OCR & Convert", type="primary", use_container_width=True)
    with col_inf:
        st.markdown(f"<div style='padding-top:10px;color:#5a7099;font-style:italic;font-size:14px;'>"
                    f"{len(uploaded_images)} image(s) · {upscale_choice}× upscale</div>",
                    unsafe_allow_html=True)

    if not run_img:
        st.stop()

    section("🔬 Converting…")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    sheets_data = []
    bar = st.progress(0.0, text="Starting…")

    for i, img_file in enumerate(uploaded_images):
        sname  = sheet_names[i]
        status = st.empty()
        status.markdown(f"<span style='color:#003399;font-weight:700;'>"
                        f"OCR — {img_file.name} → '{sname}'…</span>", unsafe_allow_html=True)
        try:
            pil = Image.open(img_file)
            row_dicts, debug = process_image_to_rows(pil, upscale=upscale_choice)

            if show_debug:
                st.info(f"**Anchors for '{sname}':** "
                        f"Note col={debug['note_col']}px · "
                        f"Val1 col={debug['val1_col']}px · "
                        f"Val2 col={debug['val2_col']}px · "
                        f"Midpoint={debug['midpoint']:.0f}px · "
                        f"Header top={debug['header_top']}px")

            non_blank = [r for r in row_dicts if any([
                str(r.get("label","")).strip(), str(r.get("val1","")).strip(),
                str(r.get("val2","")).strip()])]
            sheets_data.append((sname, row_dicts))
            status.success(f"✅ '{sname}' — {len(non_blank)} data rows extracted")
        except Exception as exc:
            status.error(f"❌ {img_file.name}: {exc}")

        bar.progress((i+1)/len(uploaded_images), text=f"Processed {i+1}/{len(uploaded_images)}")

    bar.progress(1.0, text="Done")

    if not sheets_data:
        st.error("❌ No images converted.")
        st.stop()

    excel_bytes = rows_to_excel_bytes(sheets_data)
    excel_name  = "FBC_Screenshot_Extract.xlsx"

    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    section("⬇️ Download Result")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    st.success(f"✅ {len(sheets_data)} sheet(s) extracted.")
    col_dl, col_nxt = st.columns([1,2])
    with col_dl:
        st.markdown(f"""<div class="result-card">
          <div style="font-family:'Playfair Display',serif;font-weight:700;color:#001a5c;font-size:15px;margin-bottom:4px;">
            📊 {excel_name}</div>
          <div style="color:#5a7099;font-size:13px;font-style:italic;">
            {len(sheets_data)} sheet(s) · {", ".join(s[0] for s in sheets_data)} · {len(excel_bytes)//1024:,} KB
          </div></div>""", unsafe_allow_html=True)
        st.download_button("⬇️ Download Excel", data=excel_bytes, file_name=excel_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    with col_nxt:
        st.markdown("""<div class="next-steps">
          <div class="next-steps-title">💡 Next Steps</div>
          <div class="next-steps-body">
            <b>1.</b> Download and open the Excel.<br>
            <b>2.</b> Check any <span style="color:#991b1b;font-weight:700;">red-flagged cells</span> against your screenshots.<br>
            <b>3.</b> Verify column totals sum correctly.<br>
            <b>4.</b> Go to <b>📊 DCF Model</b> and upload the cleaned Excel.
          </div></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PDF MODE
# ─────────────────────────────────────────────────────────────────────────────
else:
    if not PDF_LIBS_OK:
        st.error("❌ pdf2image not installed. Add to requirements.txt + poppler-utils to packages.txt.")
        st.stop()

    section("📤 Upload Scanned PDF(s)")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    uploaded_pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"],
        accept_multiple_files=True, label_visibility="collapsed")

    if not uploaded_pdfs:
        st.info("⬆️ Upload one or more scanned PDF files to begin.")
        st.stop()

    col_btn, col_inf = st.columns([1,3])
    with col_btn:
        run_pdf = st.button("▶️ Run OCR & Convert", type="primary", use_container_width=True)
    with col_inf:
        st.markdown(f"<div style='padding-top:10px;color:#5a7099;font-style:italic;font-size:14px;'>"
                    f"{len(uploaded_pdfs)} PDF(s) · DPI {dpi_choice} · {upscale_choice}× upscale</div>",
                    unsafe_allow_html=True)

    if not run_pdf:
        st.stop()

    results = []; errors = []
    overall = st.progress(0.0, text="Starting…")

    for pdf_i, pdf_file in enumerate(uploaded_pdfs):
        pdf_name  = pdf_file.name
        base_name = os.path.splitext(pdf_name)[0]
        xl_name   = base_name + ".xlsx"

        st.markdown(f"<div style='font-family:Playfair Display,serif;font-size:17px;"
                    f"font-weight:700;color:#001a5c;margin:18px 0 6px 0;'>📄 {pdf_name}</div>",
                    unsafe_allow_html=True)
        page_bar    = st.progress(0.0, text="Rendering…")
        page_status = st.empty()

        try:
            from pdf2image import convert_from_bytes as _cfb
            images  = _cfb(pdf_file.getvalue(), dpi=dpi_choice, poppler_path=None)
            n_pages = len(images)
            sheets_data = []

            for pg_i, image in enumerate(images, 1):
                page_status.markdown(f"<span style='color:#003399;font-weight:700;'>"
                                     f"OCR — page {pg_i}/{n_pages}</span>", unsafe_allow_html=True)
                page_bar.progress(pg_i/n_pages, text=f"Page {pg_i}/{n_pages}")

                row_dicts, debug = process_image_to_rows(image, upscale=upscale_choice)
                if show_debug:
                    st.info(f"Page {pg_i} anchors: Note={debug['note_col']}px · "
                            f"Val1={debug['val1_col']}px · Val2={debug['val2_col']}px · "
                            f"Mid={debug.get('midpoint','?')}px")
                sheets_data.append((f"Page {pg_i}", row_dicts))

            excel_bytes = rows_to_excel_bytes(sheets_data)
            results.append((xl_name, excel_bytes, n_pages))
            page_bar.progress(1.0, text="✅ Done")
            page_status.success(f"✅ {n_pages} page(s) → {xl_name}")

        except Exception as exc:
            errors.append((pdf_name, str(exc)))
            page_status.error(f"❌ Failed: {exc}")

        overall.progress((pdf_i+1)/len(uploaded_pdfs),
                         text=f"Processed {pdf_i+1}/{len(uploaded_pdfs)}")

    overall.progress(1.0, text="All done")

    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    section("⬇️ Download Results")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    if results:
        st.success(f"✅ {len(results)} file(s) converted.")
        cols = st.columns(min(len(results), 3))
        for i, (xl_name, xl_bytes, n_pages) in enumerate(results):
            with cols[i % 3]:
                st.markdown(f"""<div class="result-card">
                  <div style="font-family:'Playfair Display',serif;font-weight:700;color:#001a5c;font-size:15px;margin-bottom:4px;">
                    📊 {xl_name}</div>
                  <div style="color:#5a7099;font-size:13px;font-style:italic;">
                    {n_pages} page(s) · {len(xl_bytes)//1024:,} KB</div></div>""",
                    unsafe_allow_html=True)
                st.download_button("⬇️ Download", data=xl_bytes, file_name=xl_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_pdf_{i}", use_container_width=True)

    if errors:
        st.error(f"❌ {len(errors)} file(s) failed:")
        for fname, err in errors:
            st.markdown(f"<div style='color:#991b1b;font-weight:700;'>• <b>{fname}</b>: {err}</div>",
                        unsafe_allow_html=True)

    st.markdown("""<div class="next-steps">
      <div class="next-steps-title">💡 Next Step — Upload into the DCF Model</div>
      <div class="next-steps-body">
        <b>1.</b> Download and open the Excel.<br>
        <b>2.</b> Check <span style="color:#991b1b;font-weight:700;">red-flagged cells</span> against the PDF and verify column totals.<br>
        <b>3.</b> The DCF page expects <b>Sheet 1 = Income Statement</b>,
          <b>Sheet 2 = Balance Sheet</b>, <b>Sheet 3 = Cash Flow Statement</b>,
          each with an <i>Item</i> column and one column per year.<br>
        <b>4.</b> Head to <b>📊 DCF Model</b> and upload the cleaned Excel.
      </div></div>""", unsafe_allow_html=True)

st.markdown('<div class="fbc-footer">Powered by <b>FBC Securities</b> · Investment Research &amp; Valuation Dashboard</div>',
            unsafe_allow_html=True)
