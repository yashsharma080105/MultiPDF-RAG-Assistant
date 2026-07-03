# Fixes Applied

Three separate issues were causing wrong/garbage output. Here's what changed and
what you still need to do on your own machine.

## 1. `.env` had a space before `=` (fixed)
Old:  `GOOGLE_API_KEY = AQ.Ab8RN6...`
New:  `GOOGLE_API_KEY=your_real_key_here`

No spaces around `=`. See `.env.example`.

## 2. Wrong type of API key (you must fix this)
Your old key started with `AQ.A` and was 53 characters — that's not a valid
Generative AI Studio key. A real Gemini key starts with `AIza` and is ~39
characters.

Get one here: https://aistudio.google.com/apikey

Paste it into `.env` like:
```
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## 3. Broken text extraction on some PDFs (fixed, but needs one install)
Some PDFs (especially ones with Hindi/Devanagari text, like government
reports) use fonts with a broken character map. PyMuPDF still returns
*something* for these pages, but it's garbled gibberish — which then
poisons your chunks, embeddings, and final answers.

`read_pdf.py` now:
- Extracts text normally first (fast path, unchanged for 99% of pages)
- Detects when a page's text looks broken (empty, too short, or full of
  stray Latin-Extended/combining-mark characters typical of a bad font map)
- Falls back to OCR (render the page as an image, run Tesseract) only on
  those specific bad pages

**You need to install Tesseract + language packs on your machine:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-hin
```
(`hin` = Hindi; swap or add other language codes if your PDFs use different
scripts, e.g. `tesseract-ocr-tam` for Tamil)

If you only have `tesseract-ocr-eng` installed (the default), OCR fallback
will still run but won't correctly read non-English text — you'll see this
confirmed if you test and still get garbled output specifically on
non-English pages.

You can change which languages OCR uses by editing this line in `read_pdf.py`:
```python
OCR_LANGUAGES = "eng+hin"
```

## Verified results (tested against your actual PDFs)
- `NIC.pdf`: 608 pages, only 3 pages were actually broken (not the whole
  file) — OCR fallback triggers only on those, so the pipeline stays fast.
- `FRUSG_2024.pdf` / AI-ML book: extracted cleanly already, with a couple of
  edge-case pages (mostly images/diagrams) now also handled by the same
  fallback.

## Next steps
1. Replace the key in `.env` with a real one.
2. `pip install -r requirements.txt`
3. `sudo apt-get install tesseract-ocr tesseract-ocr-hin`
4. Delete your old `output/`, `embeddings/`, `vector_db/` folders (so they
   regenerate from the fixed extraction) and re-run `main.py`.
