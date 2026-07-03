import fitz  # PyMUpdf
import os     # os system
import json    # save python data into json
import pytesseract    # ocr
from PIL import Image  # used for image ->ocr
import io    # convert image bytes into an image obj



PDF_FOLDER = "pdfs"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "extracted.json")

# OCR Settings
OCR_LANGUAGE = "eng+hin"
OCR_ZOOM = 2   # page enlarged


# OCR Function

def ocr_page(page):
    """Convert PDF page to image and extract text using OCR."""

    pix = page.get_pixmap(matrix=fitz.Matrix(OCR_ZOOM, OCR_ZOOM))  # pdf page into img
    image = Image.open(io.BytesIO(pix.tobytes("png"))) # convert image byte into pillow image(graphic)

    text = pytesseract.image_to_string(image, lang=OCR_LANGUAGE) # run ocr

    return text.strip()  # return extract text after removing extra space



# Read All PDFs


def read_pdfs():

    all_documents = []

    for filename in os.listdir(PDF_FOLDER): #read every file inside folder

        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER, filename)

        print(f"\nReading: {filename}")

        doc = fitz.open(pdf_path)

        ocr_count = 0

        for page_number, page in enumerate(doc, start=1):

            # Try normal text extraction
            text = page.get_text().strip()

            # If text is empty or very short, use OCR
            if len(text) < 10:
                try:
                    text = ocr_page(page)
                    ocr_count += 1
                except Exception as e:
                    print(f"OCR Failed on Page {page_number}: {e}")

            all_documents.append({
                "pdf_name": filename,
                "page": page_number,
                "text": text
            })

        doc.close()

        print(f"Completed: {filename}")

        if ocr_count:
            print(f"OCR used on {ocr_count} page(s)")

    # Save extracted text
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(all_documents, file, indent=4, ensure_ascii=False)

    print("\nText Extraction Completed!")
    print(f"Total Pages: {len(all_documents)}")

    return all_documents


if __name__ == "__main__":
    read_pdfs()