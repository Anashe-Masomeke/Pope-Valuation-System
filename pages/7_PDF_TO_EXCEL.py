
import io, os, re
import streamlit as st

st.set_page_config(page_title="PDF / Image → Excel | FBC", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True
if "user" not in st.session_state or not st.session_state.get("user"):
    st.session_state["user"] = {"username":"analyst","role":"analyst","full_name":"Analyst"}

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
.stButton>button{background:linear-gradient(135deg,#003399,#0055ee)!important;color:#fff!important;
  border:none!important;border-radius:10px!important;font-weight:700!important;
  font-family:"EB Garamond",serif!important;font-size:15px!important;padding:10px 24px!important;
  box-shadow:0 4px 14px rgba(0,51,153,.30)!important;transition:all .2s ease!important;}
.stButton>button:hover{transform:translateY(-2px)!important;}
.stButton>button *,.stDownloadButton>button *{color:#fff!important;}
.stDownloadButton>button{background:linear-gradient(135deg,#003399,#0044cc)!important;
  color:#fff!important;border-radius:10px!important;font-weight:700!important;border:none!important;}
[data-testid="stFileUploader"]{background:rgba(0,51,153,.04);
  border:1px dashed rgba(0,51,153,.25);border-radius:14px;padding:18px 20px;}
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
      Financial-statement-aware OCR — robust column detection for any statement format
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

# ═══════════════════════════════════════════════════════════════════════════════
# OCR PIPELINE — ROBUST MULTI-FORMAT COLUMN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

NOISE_PAT   = re.compile(r"^[^a-zA-Z0-9()\-.]+$")
NOISE_WORDS = {"it","ot","be","a=","bet","Ml","oe","i","—","~~","MIl","Mi","Ml"}

# ── Pre-processing ────────────────────────────────────────────────────────────
def _preprocess(pil_image, upscale=3):
    img = pil_image.convert("RGB")
    img = img.resize((img.width * upscale, img.height * upscale), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = img.filter(ImageFilter.SHARPEN)
    return img, upscale

def _ocr_words(preprocessed_img, upscale):
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

# ── ROBUST ANCHOR DETECTION ───────────────────────────────────────────────────
def _detect_anchors(rows, img_width):
    """
    Detect note_col, val1_col, val2_col from the header area using multiple
    strategies, so it works for all these real-world formats:

    Format A:  Note    2024      2023          (plain year headers)
    Format B:  Note    US$       US$           (currency headers)
    Format C:  Notes   March 2026 US$  March 2025 US$  (FBC/ZSE style)
    Format D:  Note    USS       USS           (OCR misread of US$)
    Fallback:  Geometric split — right 55% of image split into two halves
    """

    # ── Strategy 1: look for "note" / "notes" keyword in any row ─────────────
    for r in rows:
        texts_lower = [w["text"].lower() for w in r]
        has_note = any(t in ("note","notes") for t in texts_lower)
        if not has_note:
            continue

        note_w = next(w for w in r if w["text"].lower() in ("note","notes"))
        note_col   = note_w["left"]
        header_top = r[0]["top"]

        # All words to the RIGHT of the note column are potential value headers
        right_words = sorted(
            [w for w in r if w["left"] > note_col + 30],
            key=lambda w: w["left"]
        )

        # ── Sub-strategy 1a: two or more distinct x-clusters on the right ────
        # Cluster right-side words by x proximity (handles multi-word col headers
        # like "March 2026" and "US$" that appear on the same row)
        clusters = []
        if right_words:
            cur = [right_words[0]]
            for w in right_words[1:]:
                if w["left"] - cur[-1]["right"] < 80:   # same cluster
                    cur.append(w)
                else:
                    clusters.append(cur)
                    cur = [w]
            clusters.append(cur)

        # Take the leftmost x of each cluster as the anchor
        cluster_lefts = [c[0]["left"] for c in clusters]

        if len(cluster_lefts) >= 2:
            return note_col, cluster_lefts[0], cluster_lefts[-1], header_top
        elif len(cluster_lefts) == 1:
            return note_col, cluster_lefts[0], None, header_top

        # ── Sub-strategy 1b: note row found but no right words ────────────────
        # Look ONE row below for the actual column header words
        note_row_idx = rows.index(r)
        for look in rows[note_row_idx+1 : note_row_idx+4]:
            rw = sorted([w for w in look if w["left"] > note_col + 30],
                        key=lambda w: w["left"])
            if len(rw) >= 2:
                return note_col, rw[0]["left"], rw[-1]["left"], header_top
            elif len(rw) == 1:
                return note_col, rw[0]["left"], None, header_top

    # ── Strategy 2: two 4-digit years on the same row ────────────────────────
    for r in rows:
        year_ws = sorted(
            [w for w in r if re.match(r"^20\d\d$", w["text"])],
            key=lambda w: w["left"]
        )
        if len(year_ws) >= 2:
            note_col = int(year_ws[0]["left"] * 0.5)
            return note_col, year_ws[0]["left"], year_ws[1]["left"], r[0]["top"]

    # ── Strategy 3: look for "US$" / "USS" / "USD" column headers ────────────
    for r in rows:
        usd_ws = sorted(
            [w for w in r if re.match(r"(?i)^(us\$|uss|usd|us)$", w["text"])],
            key=lambda w: w["left"]
        )
        if len(usd_ws) >= 2:
            # Estimate note col as ~40% of left edge of first USD word
            note_col = int(usd_ws[0]["left"] * 0.4)
            return note_col, usd_ws[0]["left"], usd_ws[1]["left"], r[0]["top"]
        elif len(usd_ws) == 1:
            note_col = int(usd_ws[0]["left"] * 0.4)
            return note_col, usd_ws[0]["left"], None, r[0]["top"]

    # ── Strategy 4: look for "March" / month words as column starters ────────
    MONTHS = {"january","february","march","april","may","june","july",
               "august","september","october","november","december"}
    for r in rows:
        month_ws = sorted(
            [w for w in r if w["text"].lower() in MONTHS],
            key=lambda w: w["left"]
        )
        if len(month_ws) >= 2:
            note_col = int(month_ws[0]["left"] * 0.4)
            return note_col, month_ws[0]["left"], month_ws[1]["left"], r[0]["top"]

    # ── Strategy 5: look for rows where two large numbers appear ─────────────
    # Find the first data row that has exactly two numbers and use their x-positions
    NUM_RE = re.compile(r"^\(?\d[\d\s,.']*\)?$")
    for r in rows:
        num_ws = sorted(
            [w for w in r if NUM_RE.match(w["text"]) and len(w["text"]) >= 3],
            key=lambda w: w["left"]
        )
        if len(num_ws) >= 2:
            note_col  = int(num_ws[0]["left"] * 0.3)
            val1_col  = num_ws[0]["left"]
            val2_col  = num_ws[-1]["left"]
            header_top = r[0]["top"]
            return note_col, val1_col, val2_col, header_top

    # ── Strategy 6: pure geometry fallback ───────────────────────────────────
    # Left 45% = label, right 55% split into two equal value columns
    note_col  = int(img_width * 0.38)
    val1_col  = int(img_width * 0.55)
    val2_col  = int(img_width * 0.75)
    header_top = 0
    return note_col, val1_col, val2_col, header_top


def _col_of(left, note_col, val1_col, val2_col, midpoint):
    """Map a word's x position to a column bucket."""
    if val2_col is not None and left >= midpoint:
        return "val2"
    if val1_col is not None and left >= val1_col - 20:
        return "val1"
    if note_col is not None and left >= note_col - 10:
        return "note"
    return "label"

def _is_noise(t):
    return bool(NOISE_PAT.match(t)) or t in NOISE_WORDS

def _clean_note(tokens):
    return " ".join(t for t in tokens if re.match(r"^\d+\.?\d*$", t))

def _merge_tokens(tokens):
    """Join tokens with spaces so space-separated thousands survive into _parse_number."""
    joined = " ".join(tokens)
    joined = re.sub(r"\$(?=\d)", "8", joined)
    joined = re.sub(r"(?<=\d)[Oo](?=\d)", "0", joined)
    joined = re.sub(r"[~:=`']+", "", joined)
    return joined.strip()

def _parse_number(raw: str):
    """
    Parse any financial number format:
      "93 226 105"  → 93226105   (space-as-thousands, ZW/SA style)
      "93,226,105"  → 93226105   (comma-as-thousands, US style)
      "(66 522 818)"→ -66522818  (parenthesised negative)
      "892.13"      → 892.13     (decimal)
      "2.737.517"   → 2737517    (dot-as-thousands, European)
      "-"           → None
    """
    s = raw.strip()
    if not s or s in ("-","—","–",""):
        return None

    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1].strip()

    # Collapse spaces FIRST — this is the critical step for ZW format
    s_nospace = s.replace(" ", "")

    has_comma = "," in s_nospace
    has_dot   = "." in s_nospace

    if not has_comma and not has_dot:
        cleaned = s_nospace
    elif has_comma and not has_dot:
        cleaned = s_nospace.replace(",", "")
    elif has_dot and not has_comma:
        parts = s_nospace.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[-1]) == 3):
            cleaned = s_nospace.replace(".", "")   # dot = thousands separator
        else:
            cleaned = s_nospace                    # dot = decimal point
    else:
        cleaned = s_nospace.replace(",", "")       # US style: comma=thou, dot=dec

    if not re.match(r"^\d+(\.\d+)?$", cleaned):
        return raw

    try:
        fval = float(cleaned)
        result = int(fval) if fval == int(fval) else fval
        return -result if negative else result
    except ValueError:
        return raw

def _flag_risky(v):
    if isinstance(v, (int, float)) and v < 0:
        return str(int(abs(v))).startswith("4")
    return False

# ── Main pipeline ─────────────────────────────────────────────────────────────
def process_image_to_rows(pil_image, upscale=3):
    img_width  = pil_image.width
    proc, up   = _preprocess(pil_image, upscale)
    words      = _ocr_words(proc, up)
    rows       = _group_rows(words, tol=6)

    note_col, val1_col, val2_col, header_top = _detect_anchors(rows, img_width)

    # Midpoint between val1 and val2 (biased slightly left)
    if val1_col is not None and val2_col is not None:
        midpoint = (val1_col + val2_col) / 2 - 5
    else:
        midpoint = val2_col  # None if only one value col

    debug = {
        "note_col": note_col, "val1_col": val1_col,
        "val2_col": val2_col, "header_top": header_top,
        "midpoint": midpoint, "img_width": img_width,
    }

    output = []
    for r in rows:
        is_pre_header = (header_top is not None and header_top > 0
                         and r[0]["top"] < header_top)
        if is_pre_header:
            line = " ".join(w["text"] for w in r if not _is_noise(w["text"]))
            output.append({"label": line, "note":"", "val1":"", "val2":"", "risky": False})
            continue

        buckets = {"label":[], "note":[], "val1":[], "val2":[]}
        for w in r:
            if _is_noise(w["text"]):
                continue
            col = _col_of(w["left"], note_col, val1_col, val2_col, midpoint)
            buckets[col].append(w["text"])

        label = " ".join(buckets["label"])
        note  = _clean_note(buckets["note"])
        raw1  = _merge_tokens(buckets["val1"])
        raw2  = _merge_tokens(buckets["val2"])
        v1    = _parse_number(raw1) if raw1 else ""
        v2    = _parse_number(raw2) if raw2 else ""

        risky = _flag_risky(v1) or _flag_risky(v2)
        output.append({"label": label, "note": note,
                       "val1": v1, "val2": v2, "risky": risky})

    return output, debug


# ── Excel writer ──────────────────────────────────────────────────────────────
def rows_to_excel_bytes(sheets, col_headers=None):
    """
    col_headers: optional list of 4 strings for row 1 header.
    Default: ["Item", "Note", "Period 1", "Period 2"]
    """
    if col_headers is None:
        col_headers = ["Item", "Note", "Period 1", "Period 2"]

    wb = Workbook()
    wb.remove(wb.active)

    BLUE_FILL  = PatternFill("solid", fgColor="071426")
    LIGHT_FILL = PatternFill("solid", fgColor="F0F5FF")
    RED_FONT   = Font(color="9C0006", bold=True)
    WHITE_FONT = Font(bold=True, color="FFFFFF")
    RIGHT_ALIGN = Alignment(horizontal="right", vertical="center")
    MONEY_FMT   = "#,##0;(#,##0)"

    for title, row_dicts in sheets:
        ws = wb.create_sheet(title=title[:31])

        for ci, h in enumerate(col_headers, 1):
            c = ws.cell(1, ci, h)
            c.font  = WHITE_FONT
            c.fill  = BLUE_FILL
            c.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 22

        excel_ri = 2
        for row in row_dicts:
            label = row.get("label","")
            note  = row.get("note","")
            v1    = row.get("val1","")
            v2    = row.get("val2","")
            risky = row.get("risky", False)

            if not any([str(label).strip(), str(note).strip(),
                        str(v1).strip(), str(v2).strip()]):
                continue

            ws.cell(excel_ri, 1, label)
            ws.cell(excel_ri, 2, note)

            for ci, val in [(3, v1), (4, v2)]:
                cell = ws.cell(excel_ri, ci)
                if isinstance(val, (int, float)):
                    cell.value         = val
                    cell.number_format = MONEY_FMT
                    cell.alignment     = RIGHT_ALIGN
                    if risky:
                        cell.font = RED_FONT
                elif val not in ("", None):
                    # Try one more time to parse — catches cases where
                    # _parse_number returned the raw string
                    reparsed = _parse_number(str(val))
                    if isinstance(reparsed, (int, float)):
                        cell.value         = reparsed
                        cell.number_format = MONEY_FMT
                        cell.alignment     = RIGHT_ALIGN
                        if risky:
                            cell.font = RED_FONT
                    else:
                        cell.value     = str(val)
                        cell.alignment = RIGHT_ALIGN

            if excel_ri % 2 == 0:
                for ci in range(1, 5):
                    ws.cell(excel_ri, ci).fill = LIGHT_FILL

            if label.isupper() or (not label and (v1 or v2)):
                for ci in range(1, 5):
                    ws.cell(excel_ri, ci).font = Font(bold=True)

            excel_ri += 1

        ws.column_dimensions["A"].width = 46
        ws.column_dimensions["B"].width = 8
        ws.column_dimensions["C"].width = 17
        ws.column_dimensions["D"].width = 17
        ws.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

section("🔀 Choose Conversion Mode")
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

mode = st.radio("mode",
    ["📄 PDF — scanned multi-page document",
     "🖼️ Image / Screenshot (PNG, JPG, WEBP …)"],
    horizontal=True, label_visibility="collapsed")
is_image_mode = mode.startswith("🖼️")

st.markdown("""
<div class="fbc-info-card">
  <div class="fbc-info-card-title">🔬 How This Works</div>
  <div class="fbc-info-card-body">
    Uses a <b>6-strategy column detector</b> that handles all these real-world header formats:<br>
    • <code>Note | 2024 | 2023</code> &nbsp;·&nbsp;
      <code>Note | US$ | US$</code> &nbsp;·&nbsp;
      <code>Notes | March 2026 US$ | March 2025 US$</code><br>
    • Month-based headers &nbsp;·&nbsp; Geometry fallback (when no header is found)<br>
    Space-separated thousands (<code>93 226 105</code>) are correctly parsed as integers.
    Known Tesseract misreads (<code>$→8</code>, <code>O→0</code>) are auto-corrected.
    Risky cells are <span style="color:#fca5a5;font-weight:700;">flagged red</span> for manual check.
  </div>
</div>
<div class="fbc-warn-card">
  <div class="fbc-warn-title">⚠️ Always verify totals before uploading to the DCF model</div>
  <div class="fbc-warn-body">
    OCR accuracy on scanned documents is ~98–99%. Sum each column and cross-check against
    the printed statement totals before feeding figures into a valuation model.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

# ── Options ───────────────────────────────────────────────────────────────────
section("⚙️ OCR Options")
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

col_o1, col_o2, col_o3, col_o4 = st.columns(4)
with col_o1:
    upscale_choice = st.selectbox("Upscale factor", [2, 3], index=1,
        help="3× recommended for financial statements.")
with col_o2:
    dpi_choice = st.selectbox("Render DPI (PDF only)", [200, 300, 400], index=1,
        disabled=is_image_mode)
with col_o3:
    col1_header = st.text_input("Col 3 header (period 1)", value="2024",
        help="Label for the first value column in the Excel output.")
    col2_header = st.text_input("Col 4 header (period 2)", value="2023",
        help="Label for the second value column in the Excel output.")
with col_o4:
    show_debug = st.checkbox("Show column anchor debug info",
        help="Shows detected pixel positions — useful if columns are still misaligned.")

col_headers = ["Item", "Note", col1_header or "Period 1", col2_header or "Period 2"]

st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE MODE
# ─────────────────────────────────────────────────────────────────────────────
if is_image_mode:
    section("🖼️ Upload Screenshots / Images")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#475569;font-size:13px;font-style:italic;margin-bottom:12px;'>"
        "Upload IS, BS and CF screenshots together — they become separate named sheets "
        "in one Excel file ready for the DCF model."
        "</div>", unsafe_allow_html=True)

    uploaded_images = st.file_uploader("Upload images",
        type=["png","jpg","jpeg","webp","bmp","tiff","tif"],
        accept_multiple_files=True, label_visibility="collapsed")

    if not uploaded_images:
        st.info("⬆️ Upload one or more screenshots to begin.")
        st.stop()

    section("🏷️ Name Each Sheet")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#374151;font-size:14px;margin-bottom:10px;'>"
        "DCF model expects <b>Income Statement</b>, <b>Balance Sheet</b>, "
        "<b>Cash Flow</b> as sheet names."
        "</div>", unsafe_allow_html=True)

    sheet_names   = []
    default_names = ["Income Statement","Balance Sheet","Cash Flow"]
    prev_cols     = st.columns(min(len(uploaded_images), 3))
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
        run_img = st.button("▶️ Run OCR & Convert", type="primary",
                            use_container_width=True)
    with col_inf:
        st.markdown(
            f"<div style='padding-top:10px;color:#5a7099;font-style:italic;font-size:14px;'>"
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
        status.markdown(
            f"<span style='color:#003399;font-weight:700;'>"
            f"OCR — {img_file.name} → '{sname}'…</span>", unsafe_allow_html=True)
        try:
            pil = Image.open(img_file)
            row_dicts, debug = process_image_to_rows(pil, upscale=upscale_choice)

            if show_debug:
                st.info(
                    f"**Anchors for '{sname}'** (img width={debug['img_width']}px): "
                    f"note_col=**{debug['note_col']}px** · "
                    f"val1_col=**{debug['val1_col']}px** · "
                    f"val2_col=**{debug['val2_col']}px** · "
                    f"midpoint=**{debug.get('midpoint','?')}px** · "
                    f"header_top=**{debug['header_top']}px**")

            non_blank = [r for r in row_dicts
                         if any([str(r.get("label","")).strip(),
                                 str(r.get("val1","")).strip(),
                                 str(r.get("val2","")).strip()])]
            sheets_data.append((sname, row_dicts))
            status.success(f"✅ '{sname}' — {len(non_blank)} data rows extracted")
        except Exception as exc:
            status.error(f"❌ {img_file.name}: {exc}")
            import traceback; st.code(traceback.format_exc())

        bar.progress((i+1)/len(uploaded_images),
                     text=f"Processed {i+1}/{len(uploaded_images)}")

    bar.progress(1.0, text="Done")

    if not sheets_data:
        st.error("❌ No images converted.")
        st.stop()

    excel_bytes = rows_to_excel_bytes(sheets_data, col_headers=col_headers)
    excel_name  = "FBC_Screenshot_Extract.xlsx"

    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    section("⬇️ Download Result")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    st.success(f"✅ {len(sheets_data)} sheet(s) extracted.")
    col_dl, col_nxt = st.columns([1,2])
    with col_dl:
        st.markdown(f"""<div class="result-card">
          <div style="font-family:'Playfair Display',serif;font-weight:700;
            color:#001a5c;font-size:15px;margin-bottom:4px;">📊 {excel_name}</div>
          <div style="color:#5a7099;font-size:13px;font-style:italic;">
            {len(sheets_data)} sheet(s) · {", ".join(s[0] for s in sheets_data)}
             · {len(excel_bytes)//1024:,} KB</div></div>""",
            unsafe_allow_html=True)
        st.download_button("⬇️ Download Excel", data=excel_bytes,
            file_name=excel_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    with col_nxt:
        st.markdown("""<div class="next-steps">
          <div class="next-steps-title">💡 Next Steps</div>
          <div class="next-steps-body">
            <b>1.</b> Download and open the Excel.<br>
            <b>2.</b> Check any <span style="color:#991b1b;font-weight:700;">
              red-flagged cells</span> against your screenshots.<br>
            <b>3.</b> Verify column totals sum correctly.<br>
            <b>4.</b> Go to <b>📊 DCF Model</b> and upload the cleaned Excel.
          </div></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PDF MODE
# ─────────────────────────────────────────────────────────────────────────────
else:
    if not PDF_LIBS_OK:
        st.error("❌ pdf2image not installed. "
                 "Add to requirements.txt + poppler-utils to packages.txt.")
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
        run_pdf = st.button("▶️ Run OCR & Convert", type="primary",
                            use_container_width=True)
    with col_inf:
        st.markdown(
            f"<div style='padding-top:10px;color:#5a7099;font-style:italic;font-size:14px;'>"
            f"{len(uploaded_pdfs)} PDF(s) · DPI {dpi_choice} · "
            f"{upscale_choice}× upscale</div>",
            unsafe_allow_html=True)

    if not run_pdf:
        st.stop()

    results = []; errors = []
    overall = st.progress(0.0, text="Starting…")

    for pdf_i, pdf_file in enumerate(uploaded_pdfs):
        pdf_name  = pdf_file.name
        xl_name   = os.path.splitext(pdf_name)[0] + ".xlsx"

        st.markdown(
            f"<div style='font-family:Playfair Display,serif;font-size:17px;"
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
                page_status.markdown(
                    f"<span style='color:#003399;font-weight:700;'>"
                    f"OCR — page {pg_i}/{n_pages}</span>",
                    unsafe_allow_html=True)
                page_bar.progress(pg_i/n_pages, text=f"Page {pg_i}/{n_pages}")

                row_dicts, debug = process_image_to_rows(image, upscale=upscale_choice)
                if show_debug:
                    st.info(
                        f"Page {pg_i}: note={debug['note_col']}px · "
                        f"val1={debug['val1_col']}px · val2={debug['val2_col']}px · "
                        f"mid={debug.get('midpoint','?')}px")
                sheets_data.append((f"Page {pg_i}", row_dicts))

            excel_bytes = rows_to_excel_bytes(sheets_data, col_headers=col_headers)
            results.append((xl_name, excel_bytes, n_pages))
            page_bar.progress(1.0, text="✅ Done")
            page_status.success(f"✅ {n_pages} page(s) → {xl_name}")

        except Exception as exc:
            errors.append((pdf_name, str(exc)))
            page_status.error(f"❌ Failed: {exc}")
            import traceback; st.code(traceback.format_exc())

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
                  <div style="font-family:'Playfair Display',serif;font-weight:700;
                    color:#001a5c;font-size:15px;margin-bottom:4px;">📊 {xl_name}</div>
                  <div style="color:#5a7099;font-size:13px;font-style:italic;">
                    {n_pages} page(s) · {len(xl_bytes)//1024:,} KB</div></div>""",
                    unsafe_allow_html=True)
                st.download_button("⬇️ Download", data=xl_bytes, file_name=xl_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_pdf_{i}", use_container_width=True)

    if errors:
        st.error(f"❌ {len(errors)} file(s) failed:")
        for fname, err in errors:
            st.markdown(
                f"<div style='color:#991b1b;font-weight:700;'>"
                f"• <b>{fname}</b>: {err}</div>",
                unsafe_allow_html=True)

    st.markdown("""<div class="next-steps">
      <div class="next-steps-title">💡 Next Step — Upload into the DCF Model</div>
      <div class="next-steps-body">
        <b>1.</b> Download and open the Excel.<br>
        <b>2.</b> Check <span style="color:#991b1b;font-weight:700;">red-flagged cells</span>
          against the PDF and verify column totals.<br>
        <b>3.</b> DCF expects <b>Sheet 1 = Income Statement</b>,
          <b>Sheet 2 = Balance Sheet</b>, <b>Sheet 3 = Cash Flow Statement</b>,
          each with an <i>Item</i> column and one column per year.<br>
        <b>4.</b> Head to <b>📊 DCF Model</b> and upload.
      </div></div>""", unsafe_allow_html=True)

st.markdown(
    '<div class="fbc-footer">Powered by <b>FBC Securities</b> · '
    'Investment Research &amp; Valuation Dashboard</div>',
    unsafe_allow_html=True)
