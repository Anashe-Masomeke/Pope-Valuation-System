
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
    import pdfplumber
    PDFPLUMBER_OK = True
except ImportError:
    PDFPLUMBER_OK = False

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

# ── CSS ───────────────────────────────────────────────────────────────────────
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
      Smart extraction: direct text for digital PDFs · OCR for scanned documents
    </div>
  </div>
  <div style="background:rgba(245,180,0,.22);border:1.5px solid rgba(245,180,0,.60);
    color:#ffd040;font-size:13px;font-weight:700;padding:6px 18px;border-radius:999px;
    font-family:'EB Garamond',serif;white-space:nowrap;">FBC Securities</div>
</div>
<hr style="border:none;border-top:2px solid #dde6f5;margin:6px 0 20px 0;">
""", unsafe_allow_html=True)

if not OPENPYXL_OK:
    st.error("❌ openpyxl not installed.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: DIRECT TEXT EXTRACTION (pdfplumber) — for digital PDFs
# ═══════════════════════════════════════════════════════════════════════════════

# Financial line patterns — label + one or two numbers
# Handles: "Revenue 3 93 226 105 59 721 449"
#          "Cost of sales (66 522 818) (43 504 675)"
#          "Property, plant and equipment 8 8 755 711 5 329 016"
NUM_TOKEN  = r"[\(\-]?\d[\d\s,\.]*\d[\)]?"   # number possibly with spaces/commas
NOTE_TOKEN = r"\d{1,2}"                        # short note reference

FIN_LINE_RE = re.compile(
    r"^(.+?)\s+"                               # label (greedy up to numbers)
    r"(?:(" + NOTE_TOKEN + r")\s+)?"           # optional note
    r"(" + NUM_TOKEN + r")"                    # val1
    r"(?:\s+(" + NUM_TOKEN + r"))?"            # optional val2
    r"\s*$"
)

# Keywords that signal start of a financial statement section
FS_HEADERS = re.compile(
    r"(?i)(revenue|total income|total assets|equity and liabilities|"
    r"cash flow|operating profit|profit or loss|profit for|"
    r"cost of sales|gross profit|expenditure|non.current assets|"
    r"current assets|current liabilities)",
    re.IGNORECASE
)

# Lines to skip — pure narrative text (long lines with no numbers)
NARRATIVE_RE = re.compile(r"^[A-Za-z ,\.\-\'\"\/\(\)]{60,}$")

def _is_number_str(s):
    """Check if a string (possibly with spaces as thousands sep) is numeric."""
    s2 = s.replace(" ", "").replace(",", "").replace("(", "").replace(")", "").replace("-","")
    return bool(re.match(r"^\d+(\.\d+)?$", s2))

def _parse_fin_line(line):
    """
    Parse a financial statement line into (label, note, val1, val2).
    Returns None if the line doesn't look like a financial data row.
    Handles ZW space-separated thousands: "93 226 105" → 93226105
    """
    line = line.strip()
    if not line:
        return None
    # Skip pure narrative lines (long, no numbers)
    if NARRATIVE_RE.match(line):
        return None

    # Find all number-like tokens in the line
    # A number token is: optional( followed by digits and spaces, optional )
    # We scan right-to-left to find the numbers at the end
    num_pat = re.compile(r"\([\d\s,\.]+\)|[\d][\d\s,\.]*[\d]|\d")

    tokens = line.split()
    if not tokens:
        return None

    # Try to find numbers at the RIGHT side of the line
    # Strategy: scan from right, collect consecutive number groups
    values = []
    note_ref = None
    label_end = len(tokens)

    i = len(tokens) - 1
    while i >= 0:
        t = tokens[i]
        # Single parenthesised number like (66) or (66 522)
        # or plain number
        clean = t.replace("(","").replace(")","").replace(",","").replace(".","")
        if clean.isdigit() or re.match(r"^\d+\.\d+$", t.replace("(","").replace(")","")):
            values.insert(0, i)
            i -= 1
        else:
            break

    if not values:
        return None

    # Determine value groups — consecutive tokens can form one space-sep number
    # Group: find runs where each token is a pure digit cluster (no letters)
    # Then merge adjacent digit runs into one number

    # Simpler: split line into label part and number part at first digit run
    # that's followed only by more digits/spaces/brackets
    m = re.search(r"(\([\d ,]+\)|[\d][\d ,]*[\d]|\d)(\s+(\([\d ,]+\)|[\d][\d ,]*[\d]|\d))?$", line)
    if not m:
        return None

    num_part = line[m.start():]
    label_part = line[:m.start()].strip()

    if not label_part:
        return None

    # Parse label for trailing note reference (single 1-2 digit number)
    label_tokens = label_part.split()
    if label_tokens and re.match(r"^\d{1,2}$", label_tokens[-1]):
        note_ref = label_tokens[-1]
        label_part = " ".join(label_tokens[:-1])

    # Parse num_part into val1, val2
    # Split on large gaps (2+ spaces) or bracket boundaries
    num_tokens = re.findall(r"\([\d\s,]+\)|[\d][\d\s,]*[\d]|\d+", num_part)

    def _clean_num(s):
        neg = s.startswith("(") and s.endswith(")")
        s2 = s.replace("(","").replace(")","").replace(",","").replace(" ","")
        try:
            v = int(s2) if "." not in s2 else float(s2)
            return -v if neg else v
        except:
            return None

    val1 = _clean_num(num_tokens[0]) if len(num_tokens) >= 1 else None
    val2 = _clean_num(num_tokens[1]) if len(num_tokens) >= 2 else None

    if val1 is None:
        return None

    # Reject lines where the "label" is just a number (subtotal-only rows)
    label_clean = label_part.strip()
    if not label_clean or re.match(r"^[\d\s,\.\(\)]+$", label_clean):
        # Still useful as a blank-label subtotal row
        label_clean = ""

    return {"label": label_clean, "note": note_ref or "", "val1": val1, "val2": val2}


def _segment_financial_lines(all_lines):
    """
    From a mixed list of text lines (narrative + financial), extract only
    the financial data rows by detecting financial statement sections.
    Returns list of parsed row dicts.
    """
    results = []
    in_fs_section = False
    consecutive_non_fin = 0

    for line in all_lines:
        line = line.strip()
        if not line:
            continue

        # Detect start of a financial section
        if FS_HEADERS.search(line):
            in_fs_section = True
            consecutive_non_fin = 0
            # The header itself might be a label row
            parsed = _parse_fin_line(line)
            if parsed:
                results.append(parsed)
            else:
                results.append({"label": line, "note": "", "val1": "", "val2": ""})
            continue

        if in_fs_section:
            parsed = _parse_fin_line(line)
            if parsed:
                results.append(parsed)
                consecutive_non_fin = 0
            else:
                # Allow a few non-financial lines (section headers, blank labels)
                consecutive_non_fin += 1
                if consecutive_non_fin <= 3:
                    # Might be a sub-heading
                    results.append({"label": line, "note": "", "val1": "", "val2": ""})
                elif consecutive_non_fin > 8:
                    # Too many non-financial lines — we've left the section
                    in_fs_section = False
                    consecutive_non_fin = 0

    return results


def extract_text_pdf(pdf_bytes, col1_header="Period 1", col2_header="Period 2"):
    """
    Extract financial data from a digital PDF using pdfplumber text extraction.
    Returns list of (sheet_name, row_dicts) tuples.
    """
    results = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        all_lines = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_lines.extend(text.split("\n"))

    if not all_lines:
        return None  # No text — fall back to OCR

    rows = _segment_financial_lines(all_lines)
    if len(rows) < 3:
        return None  # Too few rows — fall back to OCR

    return [("Extracted", rows)]


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: OCR PIPELINE — for scanned PDFs and images
# ═══════════════════════════════════════════════════════════════════════════════

NOISE_PAT   = re.compile(r"^[^a-zA-Z0-9()\-.]+$")
NOISE_WORDS = {"it","ot","be","a=","bet","Ml","oe","i","—","~~","MIl","Mi","Ml"}
MONTHS = {"january","february","march","april","may","june","july",
          "august","september","october","november","december"}
USD_PAT = re.compile(r"(?i)^(us[\$5s8sS]|u[\$5]s|usd|u\.s\.\$?)$")

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

def _x_clusters(words, gap=60):
    if not words:
        return []
    ws = sorted(words, key=lambda w: w["left"])
    clusters = [[ws[0]]]
    for w in ws[1:]:
        if w["left"] - clusters[-1][-1]["right"] < gap:
            clusters[-1].append(w)
        else:
            clusters.append([w])
    return clusters

def _detect_anchors(rows, img_width):
    # Strategy 1: month row + US$ row (two-row header)
    for ri, r in enumerate(rows):
        month_ws = [w for w in r if w["text"].lower() in MONTHS]
        if len(month_ws) < 2:
            continue
        month_ws = sorted(month_ws, key=lambda w: w["left"])
        usd_same = sorted([w for w in r if USD_PAT.match(w["text"])], key=lambda w: w["left"])
        if len(usd_same) >= 2:
            return int(month_ws[0]["left"]*0.35), usd_same[0]["left"], usd_same[1]["left"], r[0]["top"]
        for look_r in rows[ri+1:ri+6]:
            usd_ws = sorted([w for w in look_r if USD_PAT.match(w["text"])], key=lambda w: w["left"])
            if len(usd_ws) >= 2:
                return int(month_ws[0]["left"]*0.35), usd_ws[0]["left"], usd_ws[1]["left"], look_r[0]["top"]
            right_ws = [w for w in look_r if w["left"] > month_ws[0]["left"]]
            cls = _x_clusters(right_ws, gap=60)
            if len(cls) >= 2:
                return int(month_ws[0]["left"]*0.35), cls[0][0]["left"], cls[-1][0]["left"], look_r[0]["top"]

    # Strategy 2: "Notes" keyword
    for r in rows:
        has_note = any(w["text"].lower() in ("note","notes") for w in r)
        if not has_note:
            continue
        note_w = next(w for w in r if w["text"].lower() in ("note","notes"))
        note_col = note_w["left"]
        right_words = sorted([w for w in r if w["left"] > note_col+30], key=lambda w: w["left"])
        clusters = _x_clusters(right_words, gap=80)
        cl = [c[0]["left"] for c in clusters]
        if len(cl) >= 2:
            return note_col, cl[0], cl[-1], r[0]["top"]
        note_row_idx = rows.index(r)
        for look in rows[note_row_idx+1:note_row_idx+5]:
            rw = sorted([w for w in look if w["left"] > note_col+30], key=lambda w: w["left"])
            cls2 = _x_clusters(rw, gap=80)
            if len(cls2) >= 2:
                return note_col, cls2[0][0]["left"], cls2[-1][0]["left"], r[0]["top"]

    # Strategy 3: two 4-digit years
    for r in rows:
        yr = sorted([w for w in r if re.match(r"^20\d\d$", w["text"])], key=lambda w: w["left"])
        if len(yr) >= 2:
            return int(yr[0]["left"]*0.5), yr[0]["left"], yr[1]["left"], r[0]["top"]

    # Strategy 4: two US$ on same row
    for r in rows:
        usd = sorted([w for w in r if USD_PAT.match(w["text"])], key=lambda w: w["left"])
        if len(usd) >= 2:
            return int(usd[0]["left"]*0.4), usd[0]["left"], usd[1]["left"], r[0]["top"]

    # Strategy 5: month words same row
    for r in rows:
        mw = sorted([w for w in r if w["text"].lower() in MONTHS], key=lambda w: w["left"])
        if len(mw) >= 2:
            return int(mw[0]["left"]*0.4), mw[0]["left"], mw[1]["left"], r[0]["top"]

    # Strategy 6: first row with two large numbers
    NUM_RE = re.compile(r"^\(?\d[\d\s,.']*\)?$")
    for r in rows:
        nw = sorted([w for w in r if NUM_RE.match(w["text"]) and len(w["text"])>=3], key=lambda w: w["left"])
        if len(nw) >= 2:
            return int(nw[0]["left"]*0.3), nw[0]["left"], nw[-1]["left"], r[0]["top"]

    # Fallback geometry
    return int(img_width*0.38), int(img_width*0.55), int(img_width*0.75), 0

def _col_of(word, note_col, val1_col, val2_col, midpoint):
    left = word["left"]
    text = word["text"]
    is_number = bool(re.match(r"^\(?\d[\d\s,.']*\)?$", text) and len(text)>=2)
    if val2_col is not None and midpoint is not None and left >= midpoint:
        return "val2"
    if val1_col is not None and left >= val1_col - 15:
        return "val1"
    if note_col is not None and left >= note_col - 10 and not is_number:
        return "note"
    if is_number and val1_col is not None and left >= note_col:
        return "val1"
    return "label"

def _is_noise(t):
    return bool(NOISE_PAT.match(t)) or t in NOISE_WORDS

def _clean_note(tokens):
    return " ".join(t for t in tokens if re.match(r"^\d+\.?\d*$", t))

def _merge_tokens(tokens):
    joined = " ".join(tokens)
    joined = re.sub(r"\$(?=\d)", "8", joined)
    joined = re.sub(r"(?<=\d)[Oo](?=\d)", "0", joined)
    joined = re.sub(r"[~:=`']+", "", joined)
    return joined.strip()

def _parse_number(raw):
    s = raw.strip()
    if not s or s in ("-","—","–",""):
        return None
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1].strip()
    s_nospace = s.replace(" ","")
    has_comma = "," in s_nospace
    has_dot   = "." in s_nospace
    if not has_comma and not has_dot:
        cleaned = s_nospace
    elif has_comma and not has_dot:
        cleaned = s_nospace.replace(",","")
    elif has_dot and not has_comma:
        parts = s_nospace.split(".")
        cleaned = s_nospace.replace(".","") if (len(parts)>2 or (len(parts)==2 and len(parts[-1])==3)) else s_nospace
    else:
        cleaned = s_nospace.replace(",","")
    if not re.match(r"^\d+(\.\d+)?$", cleaned):
        return raw
    try:
        fval = float(cleaned)
        result = int(fval) if fval == int(fval) else fval
        return -result if negative else result
    except:
        return raw

def process_image_to_rows(pil_image, upscale=3):
    img_width = pil_image.width
    proc, up  = _preprocess(pil_image, upscale)
    words     = _ocr_words(proc, up)
    rows      = _group_rows(words, tol=6)
    note_col, val1_col, val2_col, header_top = _detect_anchors(rows, img_width)
    midpoint = (val1_col + val2_col)/2 - 5 if (val1_col is not None and val2_col is not None) else None
    debug = {"note_col":note_col,"val1_col":val1_col,"val2_col":val2_col,
             "header_top":header_top,"midpoint":midpoint,"img_width":img_width}
    output = []
    for r in rows:
        is_pre = header_top>0 and r[0]["top"]<header_top
        if is_pre:
            line = " ".join(w["text"] for w in r if not _is_noise(w["text"]))
            output.append({"label":line,"note":"","val1":"","val2":"","risky":False})
            continue
        buckets = {"label":[],"note":[],"val1":[],"val2":[]}
        for w in r:
            if _is_noise(w["text"]): continue
            col = _col_of(w, note_col, val1_col, val2_col, midpoint)
            buckets[col].append(w["text"])
        label = " ".join(buckets["label"])
        note  = _clean_note(buckets["note"])
        raw1  = _merge_tokens(buckets["val1"])
        raw2  = _merge_tokens(buckets["val2"])
        v1 = _parse_number(raw1) if raw1 else ""
        v2 = _parse_number(raw2) if raw2 else ""
        output.append({"label":label,"note":note,"val1":v1,"val2":v2,"risky":False})
    return output, debug

# ── Excel writer ──────────────────────────────────────────────────────────────
def rows_to_excel_bytes(sheets, col_headers=None):
    if col_headers is None:
        col_headers = ["Item","Note","Period 1","Period 2"]
    wb = Workbook(); wb.remove(wb.active)
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
            c.font = WHITE_FONT; c.fill = BLUE_FILL
            c.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 22
        excel_ri = 2
        for row in row_dicts:
            label = row.get("label",""); note = row.get("note","")
            v1 = row.get("val1",""); v2 = row.get("val2","")
            risky = row.get("risky", False)
            if not any([str(label).strip(), str(note).strip(), str(v1).strip(), str(v2).strip()]):
                continue
            ws.cell(excel_ri, 1, label); ws.cell(excel_ri, 2, note)
            for ci, val in [(3,v1),(4,v2)]:
                cell = ws.cell(excel_ri, ci)
                if isinstance(val, (int,float)):
                    cell.value = val; cell.number_format = MONEY_FMT
                    cell.alignment = RIGHT_ALIGN
                    if risky: cell.font = RED_FONT
                elif val not in ("",None):
                    rep = _parse_number(str(val))
                    if isinstance(rep, (int,float)):
                        cell.value = rep; cell.number_format = MONEY_FMT
                        cell.alignment = RIGHT_ALIGN
                    else:
                        cell.value = str(val); cell.alignment = RIGHT_ALIGN
            if excel_ri % 2 == 0:
                for ci in range(1,5): ws.cell(excel_ri,ci).fill = LIGHT_FILL
            if str(label).isupper() or (not label and (v1 or v2)):
                for ci in range(1,5): ws.cell(excel_ri,ci).font = Font(bold=True)
            excel_ri += 1
        ws.column_dimensions["A"].width = 46
        ws.column_dimensions["B"].width = 8
        ws.column_dimensions["C"].width = 17
        ws.column_dimensions["D"].width = 17
        ws.freeze_panes = "A2"
    buf = io.BytesIO(); wb.save(buf); buf.seek(0)
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

section("🔀 Choose Conversion Mode")
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
mode = st.radio("mode",
    ["📄 PDF — digital or scanned document",
     "🖼️ Image / Screenshot (PNG, JPG, WEBP …)"],
    horizontal=True, label_visibility="collapsed")
is_image_mode = mode.startswith("🖼️")

st.markdown("""
<div class="fbc-info-card">
  <div class="fbc-info-card-title">🔬 How This Works</div>
  <div class="fbc-info-card-body">
    <b>Smart dual-engine extraction:</b><br>
    • <b>Digital PDFs</b> (e.g. African Distillers, ZSE announcements) — direct text extraction, 
      instantly pulls financial lines even from multi-column newspaper layouts.<br>
    • <b>Scanned PDFs &amp; images</b> (e.g. ZAS, hand-scanned statements) — OCR with 
      6-strategy column detection handles all header formats.<br>
    The engine auto-detects which method to use — no manual switching needed.
  </div>
</div>
<div class="fbc-warn-card">
  <div class="fbc-warn-title">⚠️ Always verify totals before uploading to the DCF model</div>
  <div class="fbc-warn-body">Always cross-check column totals against the printed statement before feeding into valuation models.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

section("⚙️ Options")
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
col_o1, col_o2, col_o3, col_o4 = st.columns(4)
with col_o1:
    upscale_choice = st.selectbox("Upscale factor (OCR only)", [2,3], index=1)
with col_o2:
    dpi_choice = st.selectbox("Render DPI (scanned PDF OCR)", [200,300,400], index=1, disabled=is_image_mode)
with col_o3:
    col1_header = st.text_input("Col 3 header", value="2024")
    col2_header = st.text_input("Col 4 header", value="2023")
with col_o4:
    show_debug = st.checkbox("Show debug info")

col_headers = ["Item","Note", col1_header or "Period 1", col2_header or "Period 2"]
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE MODE
# ─────────────────────────────────────────────────────────────────────────────
if is_image_mode:
    if not LIBS_OK:
        st.error(f"❌ Missing OCR libraries: {_IMPORT_ERR}")
        st.stop()
    section("🖼️ Upload Screenshots / Images")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    uploaded_images = st.file_uploader("Upload images",
        type=["png","jpg","jpeg","webp","bmp","tiff","tif"],
        accept_multiple_files=True, label_visibility="collapsed")
    if not uploaded_images:
        st.info("⬆️ Upload one or more screenshots to begin.")
        st.stop()
    section("🏷️ Name Each Sheet")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    sheet_names = []; default_names = ["Income Statement","Balance Sheet","Cash Flow"]
    prev_cols = st.columns(min(len(uploaded_images),3))
    for i, img_file in enumerate(uploaded_images):
        with prev_cols[i%3]:
            try:
                pil_prev = Image.open(img_file); img_file.seek(0)
                st.image(pil_prev, use_container_width=True, caption=img_file.name)
            except: st.caption(img_file.name)
            sname = st.text_input(f"Sheet name {i+1}",
                value=default_names[i] if i<len(default_names) else f"Sheet {i+1}",
                key=f"img_sheet_name_{i}", label_visibility="collapsed")
            sheet_names.append(sname.strip() or f"Sheet {i+1}")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    col_btn, col_inf = st.columns([1,3])
    with col_btn:
        run_img = st.button("▶️ Run OCR & Convert", type="primary", use_container_width=True)
    with col_inf:
        st.markdown(f"<div style='padding-top:10px;color:#5a7099;font-style:italic;font-size:14px;'>"
                    f"{len(uploaded_images)} image(s) · {upscale_choice}× upscale</div>", unsafe_allow_html=True)
    if not run_img: st.stop()
    section("🔬 Converting…")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    sheets_data = []; bar = st.progress(0.0, text="Starting…")
    for i, img_file in enumerate(uploaded_images):
        sname = sheet_names[i]; status = st.empty()
        status.markdown(f"<span style='color:#003399;font-weight:700;'>OCR — {img_file.name} → '{sname}'…</span>", unsafe_allow_html=True)
        try:
            pil = Image.open(img_file)
            row_dicts, debug = process_image_to_rows(pil, upscale=upscale_choice)
            if show_debug:
                st.info(f"Anchors '{sname}': note={debug['note_col']}px · val1={debug['val1_col']}px · val2={debug['val2_col']}px · mid={debug.get('midpoint','?')}px")
            non_blank = [r for r in row_dicts if any([str(r.get("label","")).strip(),str(r.get("val1","")).strip(),str(r.get("val2","")).strip()])]
            sheets_data.append((sname, row_dicts))
            status.success(f"✅ '{sname}' — {len(non_blank)} data rows extracted")
        except Exception as exc:
            status.error(f"❌ {img_file.name}: {exc}")
            import traceback; st.code(traceback.format_exc())
        bar.progress((i+1)/len(uploaded_images), text=f"Processed {i+1}/{len(uploaded_images)}")
    bar.progress(1.0, text="Done")
    if not sheets_data: st.error("❌ No images converted."); st.stop()
    excel_bytes = rows_to_excel_bytes(sheets_data, col_headers=col_headers)
    excel_name  = "FBC_Screenshot_Extract.xlsx"
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    section("⬇️ Download Result")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    st.success(f"✅ {len(sheets_data)} sheet(s) extracted.")
    col_dl, col_nxt = st.columns([1,2])
    with col_dl:
        st.markdown(f"""<div class="result-card">
          <div style="font-family:'Playfair Display',serif;font-weight:700;color:#001a5c;font-size:15px;margin-bottom:4px;">📊 {excel_name}</div>
          <div style="color:#5a7099;font-size:13px;font-style:italic;">{len(sheets_data)} sheet(s) · {len(excel_bytes)//1024:,} KB</div></div>""", unsafe_allow_html=True)
        st.download_button("⬇️ Download Excel", data=excel_bytes, file_name=excel_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with col_nxt:
        st.markdown("""<div class="next-steps"><div class="next-steps-title">💡 Next Steps</div>
          <div class="next-steps-body"><b>1.</b> Download and open the Excel.<br>
          <b>2.</b> Verify column totals sum correctly.<br>
          <b>3.</b> Go to <b>📊 DCF Model</b> and upload the cleaned Excel.</div></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PDF MODE — SMART DUAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────
else:
    section("📤 Upload PDF(s)")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    uploaded_pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"],
        accept_multiple_files=True, label_visibility="collapsed")
    if not uploaded_pdfs:
        st.info("⬆️ Upload one or more PDF files to begin. Works with both digital and scanned PDFs.")
        st.stop()
    col_btn, col_inf = st.columns([1,3])
    with col_btn:
        run_pdf = st.button("▶️ Extract & Convert", type="primary", use_container_width=True)
    with col_inf:
        st.markdown(f"<div style='padding-top:10px;color:#5a7099;font-style:italic;font-size:14px;'>"
                    f"{len(uploaded_pdfs)} PDF(s) · auto-detects digital vs scanned</div>", unsafe_allow_html=True)
    if not run_pdf: st.stop()

    results = []; errors = []
    overall = st.progress(0.0, text="Starting…")

    for pdf_i, pdf_file in enumerate(uploaded_pdfs):
        pdf_name = pdf_file.name
        xl_name  = os.path.splitext(pdf_name)[0] + ".xlsx"
        pdf_bytes = pdf_file.getvalue()

        st.markdown(f"<div style='font-family:Playfair Display,serif;font-size:17px;font-weight:700;color:#001a5c;margin:18px 0 6px 0;'>📄 {pdf_name}</div>", unsafe_allow_html=True)
        page_status = st.empty()

        try:
            sheets_data = None
            method_used = ""

            # ── Try direct text extraction first ─────────────────────────────
            if PDFPLUMBER_OK:
                page_status.markdown("<span style='color:#003399;font-weight:700;'>🔍 Trying direct text extraction…</span>", unsafe_allow_html=True)
                try:
                    sheets_data = extract_text_pdf(pdf_bytes, col1_header=col1_header, col2_header=col2_header)
                    if sheets_data:
                        method_used = "direct text extraction"
                except Exception as e:
                    if show_debug:
                        st.warning(f"Text extraction failed: {e}")
                    sheets_data = None

            # ── Fall back to OCR if text extraction failed ────────────────────
            if not sheets_data:
                if not LIBS_OK:
                    st.error("❌ This appears to be a scanned PDF but OCR libraries (pytesseract) are not installed.")
                    errors.append((pdf_name, "Scanned PDF but OCR not available"))
                    continue
                if not PDF_LIBS_OK:
                    st.error("❌ pdf2image not installed — needed for scanned PDFs.")
                    errors.append((pdf_name, "pdf2image not installed"))
                    continue

                page_status.markdown("<span style='color:#003399;font-weight:700;'>🔬 Scanned PDF detected — running OCR…</span>", unsafe_allow_html=True)
                method_used = "OCR"
                from pdf2image import convert_from_bytes as _cfb
                images = _cfb(pdf_bytes, dpi=dpi_choice, poppler_path=None)
                n_pages = len(images)
                page_bar = st.progress(0.0)
                sheets_data = []
                for pg_i, image in enumerate(images, 1):
                    page_bar.progress(pg_i/n_pages, text=f"OCR page {pg_i}/{n_pages}")
                    row_dicts, debug = process_image_to_rows(image, upscale=upscale_choice)
                    if show_debug:
                        st.info(f"Page {pg_i}: note={debug['note_col']}px · val1={debug['val1_col']}px · val2={debug['val2_col']}px")
                    sheets_data.append((f"Page {pg_i}", row_dicts))
                page_bar.progress(1.0)

            excel_bytes_out = rows_to_excel_bytes(sheets_data, col_headers=col_headers)
            n_rows = sum(len(s[1]) for s in sheets_data)
            results.append((xl_name, excel_bytes_out, method_used, n_rows))
            page_status.success(f"✅ {pdf_name} → {xl_name} ({method_used}, {n_rows} rows)")

        except Exception as exc:
            errors.append((pdf_name, str(exc)))
            page_status.error(f"❌ Failed: {exc}")
            import traceback; st.code(traceback.format_exc())

        overall.progress((pdf_i+1)/len(uploaded_pdfs), text=f"Processed {pdf_i+1}/{len(uploaded_pdfs)}")

    overall.progress(1.0, text="All done")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    section("⬇️ Download Results")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    if results:
        st.success(f"✅ {len(results)} file(s) converted.")
        cols = st.columns(min(len(results),3))
        for i, (xl_name, xl_bytes, method, n_rows) in enumerate(results):
            with cols[i%3]:
                st.markdown(f"""<div class="result-card">
                  <div style="font-family:'Playfair Display',serif;font-weight:700;color:#001a5c;font-size:15px;margin-bottom:4px;">📊 {xl_name}</div>
                  <div style="color:#5a7099;font-size:13px;font-style:italic;">{method} · {n_rows} rows · {len(xl_bytes)//1024:,} KB</div></div>""", unsafe_allow_html=True)
                st.download_button("⬇️ Download", data=xl_bytes, file_name=xl_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_pdf_{i}", use_container_width=True)

    if errors:
        st.error(f"❌ {len(errors)} file(s) failed:")
        for fname, err in errors:
            st.markdown(f"<div style='color:#991b1b;font-weight:700;'>• <b>{fname}</b>: {err}</div>", unsafe_allow_html=True)

    st.markdown("""<div class="next-steps">
      <div class="next-steps-title">💡 Next Step — Upload into the DCF Model</div>
      <div class="next-steps-body">
        <b>1.</b> Download and open the Excel.<br>
        <b>2.</b> Verify totals against the original PDF.<br>
        <b>3.</b> DCF expects Sheet 1 = Income Statement, Sheet 2 = Balance Sheet, Sheet 3 = Cash Flow.<br>
        <b>4.</b> Head to <b>📊 DCF Model</b> and upload.
      </div></div>""", unsafe_allow_html=True)

st.markdown('<div class="fbc-footer">Powered by <b>FBC Securities</b> · Investment Research &amp; Valuation Dashboard</div>', unsafe_allow_html=True)
