import io, os, re, gc, traceback
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
.fbc-fail-card{background:#fff5f5;border:1px solid rgba(153,27,27,.25);border-left:5px solid #991b1b;
  border-radius:14px;padding:14px 18px;margin-bottom:10px;}
.fbc-fail-title{font-weight:700;color:#991b1b;font-family:"Playfair Display",serif!important;font-size:14px;}
.fbc-fail-body{color:#7f1d1d;font-size:13px;margin-top:2px;}
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
      Smart extraction: column-aware text reading for digital PDFs · OCR for scanned documents
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
# STRATEGY 1 — DIGITAL PDF EXTRACTION (column-aware, word-position based)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Why this exists: a naive page.extract_text() dump reads strictly top-to-bottom
# and will happily interleave text from two side-by-side columns that share the
# same vertical position (e.g. a "magazine style" results PDF with a narrative
# column next to a financial table). That silently produces garbled numbers
# (two different figures concatenated into one) rather than a crash — which is
# arguably worse, since it looks like real data. The fix below works on each
# page's individual word positions: for every visual row it finds the trailing
# block of numbers (note / value columns) using gaps between word boxes, then
# only pulls in the label words that sit directly beside that number block —
# so unrelated text from a neighbouring column never gets glued onto a figure.

NUM_TOKEN_RE = re.compile(r"^\(?-?\d[\d,\.]*\)?$")

def _is_num_tok(t):
    return bool(NUM_TOKEN_RE.match(t))

def _clean_numeric_group(token_texts):
    """token_texts: list of word strings that together form ONE number."""
    raw = "".join(token_texts)
    neg = raw.startswith("(") and raw.endswith(")")
    core = raw.strip("()").replace(",", "").replace(" ", "")
    if not core:
        return None
    try:
        val = float(core) if "." in core else int(core)
        return -val if neg else val
    except (ValueError, TypeError):
        return None

def _group_words_by_row(words, tol=3):
    if not words:
        return []
    ws = sorted(words, key=lambda w: w["top"])
    rows = [[ws[0]]]
    cur_top = ws[0]["top"]
    for w in ws[1:]:
        if abs(w["top"] - cur_top) <= tol:
            rows[-1].append(w)
            cur_top = sum(x["top"] for x in rows[-1]) / len(rows[-1])
        else:
            rows.append([w])
            cur_top = w["top"]
    return rows

def _extract_rows_from_pdf_page(page, num_gap=60, label_gap=16, sub_gap=10):
    """
    Column-aware extraction for one digital PDF page.
    Returns a list of dicts: {label, note, val1, val2, has_value}
    """
    try:
        words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)
    except Exception:
        return []
    if not words:
        return []

    rows = _group_words_by_row(words)
    out = []
    for r in rows:
        r2 = sorted(r, key=lambda w: w["x0"])
        n = len(r2)

        # Walk backwards from the end of the row collecting a contiguous
        # run of number-like tokens (this is the note/val1/val2 block).
        i = n - 1
        while i >= 0 and _is_num_tok(r2[i]["text"]):
            if i < n - 1 and r2[i+1]["x0"] - r2[i]["x1"] > num_gap:
                break
            i -= 1
        num_start_idx = i + 1

        if num_start_idx >= n or num_start_idx == 0:
            # Either no trailing numbers at all, or the whole row is numbers
            # with nothing readable before it — pass it through untouched so
            # nothing gets lost; it'll show up as a label-only / heading row.
            line = " ".join(w["text"] for w in r2)
            out.append({"label": line, "note": "", "val1": "", "val2": "", "has_value": False})
            continue

        # Only walk left from the number block while the gap between
        # consecutive words stays small — this is what keeps an unrelated
        # column's text from being absorbed into the label.
        j = num_start_idx - 1
        while j > 0 and r2[j]["x0"] - r2[j-1]["x1"] <= label_gap:
            j -= 1
        label = " ".join(w["text"] for w in r2[j:num_start_idx])

        # Split the trailing number block into separate figures using a much
        # tighter gap threshold (thousand-separator spacing is only a couple
        # of points; the gap between two distinct columns is much wider).
        numtoks = r2[num_start_idx:]
        groups = [[numtoks[0]]]
        for t in numtoks[1:]:
            prev = groups[-1][-1]
            if t["x0"] - prev["x1"] <= sub_gap:
                groups[-1].append(t)
            else:
                groups.append([t])

        note, val1, val2 = "", "", ""
        if len(groups) >= 3:
            note = "".join(w["text"] for w in groups[0])
            val1 = _clean_numeric_group([w["text"] for w in groups[1]])
            val2 = _clean_numeric_group([w["text"] for w in groups[2]])
        elif len(groups) == 2:
            val1 = _clean_numeric_group([w["text"] for w in groups[0]])
            val2 = _clean_numeric_group([w["text"] for w in groups[1]])
        elif len(groups) == 1:
            val1 = _clean_numeric_group([w["text"] for w in groups[0]])

        out.append({
            "label": label, "note": note,
            "val1": val1 if val1 is not None else "",
            "val2": val2 if val2 is not None else "",
            "has_value": True,
        })
    return out

# Recognised financial-statement section headers, used to split a long
# document into separate, named sheets (matches what the DCF model expects:
# Income Statement / Balance Sheet / Cash Flow).
SECTION_PATTERNS = [
    ("Income Statement",   re.compile(r"(?i)(profit\s+or\s+loss|income\s+statement|statement\s+of\s+comprehensive\s+income)")),
    ("Balance Sheet",       re.compile(r"(?i)(financial\s+position|balance\s+sheet)")),
    ("Cash Flow",           re.compile(r"(?i)(cash\s*flows?)")),
    ("Changes in Equity",   re.compile(r"(?i)(changes\s+in\s+equity)")),
]

def _classify_section(label, current):
    for name, pat in SECTION_PATTERNS:
        if pat.search(label):
            return name
    return current

def _is_short_heading(label, max_words=6):
    words = label.strip().split()
    return 0 < len(words) <= max_words

def extract_digital_pdf(pdf_bytes):
    """
    Reads every page of a digital PDF, isolates financial line-items into
    section-named sheets, and also returns the full raw text (page by page)
    as a safety net.

    Returns (sheets, raw_lines):
      sheets    -> list of (sheet_name, row_dicts) or None if the PDF has no
                   extractable text at all (i.e. it's a scanned image and the
                   caller should fall back to OCR).
      raw_lines -> list of every line of text pulled from the document,
                   regardless of whether it was recognised as a data row.
    """
    sections = {}
    order = []
    raw_lines = []
    current_section = "Other"
    found_any_text = False

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            try:
                if page.chars:
                    found_any_text = True
            except Exception:
                pass

            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if page_text:
                raw_lines.extend(page_text.split("\n"))

            rows = _extract_rows_from_pdf_page(page)
            for row in rows:
                current_section = _classify_section(row["label"], current_section)
                if row["has_value"] or _is_short_heading(row["label"]):
                    if current_section not in sections:
                        sections[current_section] = []
                        order.append(current_section)
                    sections[current_section].append({
                        "label": row["label"], "note": row["note"],
                        "val1": row["val1"], "val2": row["val2"],
                    })

    if not found_any_text:
        return None, None  # scanned PDF — caller falls back to OCR

    sheets = [(name, sections[name]) for name in order if sections[name]]
    return sheets, raw_lines


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: OCR PIPELINE — for scanned PDFs and images
# ═══════════════════════════════════════════════════════════════════════════════

NOISE_PAT   = re.compile(r"^[^a-zA-Z0-9()\-.]+$")
NOISE_WORDS = {"it","ot","be","a=","bet","Ml","oe","i","—","~~","MIl","Mi","Ml"}
MONTHS = {"january","february","march","april","may","june","july",
          "august","september","october","november","december"}
USD_PAT = re.compile(r"(?i)^(us[\$5s8sS]|u[\$5]s|usd|u\.s\.\$?)$")

# Hard cap on the number of pixels we'll ever hand to PIL/Tesseract after
# upscaling. Colourful, image-heavy PDFs (magazine layouts, scanned photos)
# rendered at high DPI and then upscaled 2-3x can balloon into gigabyte-scale
# arrays, which is the kind of thing that takes the whole app down rather
# than raising a catchable Python exception. We scale the upscale factor
# down automatically instead of ever allocating past this ceiling.
MAX_OCR_PIXELS = 16_000_000  # ~16 megapixels post-upscale

def _preprocess(pil_image, upscale=3):
    img = pil_image.convert("RGB")
    w, h = img.width, img.height
    eff_upscale = upscale
    if w * h * (upscale ** 2) > MAX_OCR_PIXELS and w * h > 0:
        eff_upscale = max(1.0, (MAX_OCR_PIXELS / (w * h)) ** 0.5)
    new_w = max(1, int(w * eff_upscale))
    new_h = max(1, int(h * eff_upscale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = img.filter(ImageFilter.SHARPEN)
    return img, eff_upscale

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
             "header_top":header_top,"midpoint":midpoint,"img_width":img_width,
             "upscale_used":up}
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

def run_ocr_pipeline(pdf_bytes, dpi_choice, upscale_choice, show_debug):
    """
    Renders every page of a (scanned) PDF to an image and OCRs it. Designed
    to degrade gracefully: a render failure retries at a lower DPI, and a
    failure on any single page is skipped (with a warning) rather than
    aborting the whole document.
    Returns (sheets, raw_lines).
    """
    from pdf2image import convert_from_bytes as _cfb
    try:
        images = _cfb(pdf_bytes, dpi=dpi_choice)
    except Exception as e:
        if show_debug:
            st.warning(f"Render at {dpi_choice} DPI failed ({e}); retrying at 150 DPI…")
        images = _cfb(pdf_bytes, dpi=150)

    n_pages = len(images)
    page_bar = st.progress(0.0)
    sheets_data = []
    raw_lines = []
    for pg_i, image in enumerate(images, 1):
        page_bar.progress(pg_i/n_pages, text=f"OCR page {pg_i}/{n_pages}")
        try:
            row_dicts, debug = process_image_to_rows(image, upscale=upscale_choice)
            if show_debug:
                st.info(f"Page {pg_i}: note={debug['note_col']}px · val1={debug['val1_col']}px · "
                        f"val2={debug['val2_col']}px · upscale used={debug['upscale_used']:.2f}x")
            sheets_data.append((f"Page {pg_i}", row_dicts))
            raw_lines.extend(r.get("label","") for r in row_dicts if str(r.get("label","")).strip())
        except MemoryError:
            st.warning(f"⚠️ Page {pg_i} skipped — image was too large to process safely.")
        except Exception as e:
            st.warning(f"⚠️ Page {pg_i} OCR failed: {e}")
            if show_debug:
                st.code(traceback.format_exc())
        finally:
            del image
            gc.collect()
    page_bar.progress(1.0)
    return sheets_data, raw_lines

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
    used_names = set()
    for title, row_dicts in sheets:
        safe_title = (title or "Sheet")[:31]
        base, n = safe_title, 1
        while safe_title in used_names:
            n += 1
            safe_title = f"{base[:28]}-{n}"
        used_names.add(safe_title)

        ws = wb.create_sheet(title=safe_title)
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
            try:
                ws.cell(excel_ri, 1, str(label)[:2000]); ws.cell(excel_ri, 2, str(note)[:50])
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
                            cell.value = str(val)[:2000]; cell.alignment = RIGHT_ALIGN
                if excel_ri % 2 == 0:
                    for ci in range(1,5): ws.cell(excel_ri,ci).fill = LIGHT_FILL
                if str(label).isupper() or (not label and (v1 or v2)):
                    for ci in range(1,5): ws.cell(excel_ri,ci).font = Font(bold=True)
                excel_ri += 1
            except Exception:
                # Never let one malformed row blow up the whole export —
                # skip it and keep going.
                continue
        ws.column_dimensions["A"].width = 46
        ws.column_dimensions["B"].width = 8
        ws.column_dimensions["C"].width = 17
        ws.column_dimensions["D"].width = 17
        ws.freeze_panes = "A2"
    buf = io.BytesIO(); wb.save(buf); buf.seek(0)
    return buf.getvalue()

def raw_text_sheet(raw_lines, max_lines=2000):
    """Builds a plain 'Raw Text' sheet from whatever text was extracted —
    the safety net so nothing is ever fully lost, even when structured
    parsing can't make sense of a document's layout."""
    if not raw_lines:
        return None
    rows = [{"label": ln, "note": "", "val1": "", "val2": ""}
            for ln in raw_lines[:max_lines] if str(ln).strip()]
    return ("Raw Text", rows) if rows else None

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
    • <b>Digital PDFs</b> — column-aware text reading. Handles plain single-column
      statements as well as "magazine style" layouts (e.g. a chairman's statement
      column sitting next to a financial table) without letting the two bleed
      into each other.<br>
    • <b>Scanned PDFs &amp; images</b> (e.g. hand-scanned statements) — OCR with
      6-strategy column detection handles all header formats.<br>
    Every conversion also includes a <b>Raw Text</b> sheet with everything that
    was read from the document, so you always have a fallback to check against
    even if a line wasn't recognised as a data row.
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
            except Exception:
                st.caption(img_file.name)
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
    sheets_data = []; raw_lines_all = []; failed_images = []
    bar = st.progress(0.0, text="Starting…")
    for i, img_file in enumerate(uploaded_images):
        sname = sheet_names[i]; status = st.empty()
        status.markdown(f"<span style='color:#003399;font-weight:700;'>OCR — {img_file.name} → '{sname}'…</span>", unsafe_allow_html=True)
        try:
            pil = Image.open(img_file)
            row_dicts, debug = process_image_to_rows(pil, upscale=upscale_choice)
            if show_debug:
                st.info(f"Anchors '{sname}': note={debug['note_col']}px · val1={debug['val1_col']}px · "
                        f"val2={debug['val2_col']}px · mid={debug.get('midpoint','?')}px · "
                        f"upscale used={debug['upscale_used']:.2f}x")
            non_blank = [r for r in row_dicts if any([str(r.get("label","")).strip(),str(r.get("val1","")).strip(),str(r.get("val2","")).strip()])]
            sheets_data.append((sname, row_dicts))
            raw_lines_all.extend(r.get("label","") for r in row_dicts if str(r.get("label","")).strip())
            status.success(f"✅ '{sname}' — {len(non_blank)} data rows extracted")
        except MemoryError:
            status.error(f"❌ {img_file.name}: image too large to process safely — try a smaller file or lower upscale.")
            failed_images.append((img_file.name, "Image too large"))
        except Exception as exc:
            status.error(f"❌ Failed to generate Excel for {img_file.name}: {exc}")
            failed_images.append((img_file.name, str(exc)))
            if show_debug:
                st.code(traceback.format_exc())
        finally:
            gc.collect()
        bar.progress((i+1)/len(uploaded_images), text=f"Processed {i+1}/{len(uploaded_images)}")
    bar.progress(1.0, text="Done")
    raw_sheet = raw_text_sheet(raw_lines_all)
    if raw_sheet:
        sheets_data.append(raw_sheet)
    if not sheets_data:
        st.error("❌ No images converted.")
        st.stop()
    try:
        excel_bytes = rows_to_excel_bytes(sheets_data, col_headers=col_headers)
    except Exception as e:
        st.error(f"❌ Failed to generate Excel: {e}")
        if show_debug:
            st.code(traceback.format_exc())
        st.stop()
    excel_name  = "FBC_Screenshot_Extract.xlsx"
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    section("⬇️ Download Result")
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    st.success(f"✅ {len(sheets_data)} sheet(s) extracted.")
    if failed_images:
        st.error(f"❌ {len(failed_images)} image(s) failed:")
        for fname, err in failed_images:
            st.markdown(f"""<div class="fbc-fail-card">
              <div class="fbc-fail-title">{fname}</div>
              <div class="fbc-fail-body">{err}</div></div>""", unsafe_allow_html=True)
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
          <b>2.</b> Verify column totals sum correctly — check the <b>Raw Text</b> sheet if anything looks off.<br>
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

        st.markdown(f"<div style='font-family:Playfair Display,serif;font-size:17px;font-weight:700;color:#001a5c;margin:18px 0 6px 0;'>📄 {pdf_name}</div>", unsafe_allow_html=True)
        page_status = st.empty()

        # Everything for this file lives inside one try/except so that
        # whatever goes wrong, we report it cleanly and move on to the next
        # file instead of taking the whole app down.
        try:
            pdf_bytes = pdf_file.getvalue()
            sheets_data = None
            raw_lines = None
            method_used = ""

            # ── Strategy 1: column-aware digital text extraction ─────────────
            if PDFPLUMBER_OK:
                page_status.markdown("<span style='color:#003399;font-weight:700;'>🔍 Reading digital text…</span>", unsafe_allow_html=True)
                try:
                    sheets_data, raw_lines = extract_digital_pdf(pdf_bytes)
                    if sheets_data:
                        method_used = "digital text (column-aware)"
                except Exception as e:
                    if show_debug:
                        st.warning(f"Digital text extraction error: {e}")
                        st.code(traceback.format_exc())
                    sheets_data, raw_lines = None, None

            # ── Strategy 2: OCR fallback for scanned PDFs ────────────────────
            if not sheets_data:
                if not LIBS_OK:
                    st.error("❌ No extractable text found, and OCR libraries (pytesseract) are not installed.")
                    errors.append((pdf_name, "No extractable text; OCR not available"))
                    overall.progress((pdf_i+1)/len(uploaded_pdfs), text=f"Processed {pdf_i+1}/{len(uploaded_pdfs)}")
                    continue
                if not PDF_LIBS_OK:
                    st.error("❌ pdf2image not installed — needed for scanned PDFs.")
                    errors.append((pdf_name, "pdf2image not installed"))
                    overall.progress((pdf_i+1)/len(uploaded_pdfs), text=f"Processed {pdf_i+1}/{len(uploaded_pdfs)}")
                    continue

                page_status.markdown("<span style='color:#003399;font-weight:700;'>🔬 Scanned PDF detected — running OCR…</span>", unsafe_allow_html=True)
                method_used = "OCR"
                sheets_data, raw_lines = run_ocr_pipeline(pdf_bytes, dpi_choice, upscale_choice, show_debug)

            if not sheets_data:
                errors.append((pdf_name, "No data rows could be extracted from this file"))
                page_status.error(f"❌ Failed to generate Excel for {pdf_name}: no readable rows found.")
                overall.progress((pdf_i+1)/len(uploaded_pdfs), text=f"Processed {pdf_i+1}/{len(uploaded_pdfs)}")
                continue

            # Always attach the raw-text safety-net sheet, whichever path was used.
            all_sheets = list(sheets_data)
            rsheet = raw_text_sheet(raw_lines)
            if rsheet:
                all_sheets.append(rsheet)

            try:
                excel_bytes_out = rows_to_excel_bytes(all_sheets, col_headers=col_headers)
            except Exception as e:
                errors.append((pdf_name, f"Failed to build Excel file: {e}"))
                page_status.error(f"❌ Failed to generate Excel for {pdf_name}.")
                if show_debug:
                    st.code(traceback.format_exc())
                overall.progress((pdf_i+1)/len(uploaded_pdfs), text=f"Processed {pdf_i+1}/{len(uploaded_pdfs)}")
                continue

            n_rows = sum(len(s[1]) for s in sheets_data)
            results.append((xl_name, excel_bytes_out, method_used, n_rows))
            page_status.success(f"✅ {pdf_name} → {xl_name} ({method_used}, {n_rows} data rows + raw text sheet)")

        except MemoryError:
            errors.append((pdf_name, "Ran out of memory while processing this file"))
            page_status.error(f"❌ Failed to generate Excel for {pdf_name}: file too large/complex to process safely.")
        except Exception as exc:
            errors.append((pdf_name, str(exc)))
            page_status.error(f"❌ Failed to generate Excel for {pdf_name}: {exc}")
            if show_debug:
                st.code(traceback.format_exc())
        finally:
            gc.collect()

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
            st.markdown(f"""<div class="fbc-fail-card">
              <div class="fbc-fail-title">{fname}</div>
              <div class="fbc-fail-body">{err}</div></div>""", unsafe_allow_html=True)

    if results:
        st.markdown("""<div class="next-steps">
          <div class="next-steps-title">💡 Next Step — Upload into the DCF Model</div>
          <div class="next-steps-body">
            <b>1.</b> Download and open the Excel.<br>
            <b>2.</b> Verify totals against the original PDF — use the <b>Raw Text</b> sheet as a reference for anything that looks off.<br>
            <b>3.</b> Digital PDFs are split into sheets named <b>Income Statement</b> / <b>Balance Sheet</b> / <b>Cash Flow</b> where those sections are detected (scanned PDFs use Page 1, Page 2, …).<br>
            <b>4.</b> Head to <b>📊 DCF Model</b> and upload.
          </div></div>""", unsafe_allow_html=True)

st.markdown('<div class="fbc-footer">Powered by <b>FBC Securities</b> · Investment Research &amp; Valuation Dashboard</div>', unsafe_allow_html=True)
