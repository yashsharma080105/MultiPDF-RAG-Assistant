import json
import re  # text cleaning


INPUT_FILE = "output/extracted.json"
OUTPUT_FILE = "output/chunks.json"



# Clean Text

def clean_text(text):
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Chunk Function (word-based, with overlap)

def chunk_text(text, chunk_size=250, overlap=50):

    words = text.split()

    chunks = []

    start = 0  # staring indx

    while start < len(words):  # loop until all word processed

        end = min(start + chunk_size, len(words)) # ending indx

        chunk = " ".join(words[start:end])

        # Track the word offset so we can map back to a page number
        chunks.append({"text": chunk, "start_word": start})

        if end == len(words):
            break

        start = end - overlap  # create overlap

    return chunks



# Create Chunks


def create_chunks():

    # Read Extracted Data
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f) # convert json -> python list

    # Group pages by PDF, preserving page order
    pdfs = {}
    for document in documents:
        pdfs.setdefault(document["pdf_name"], []).append(document)

    all_chunks = []
    chunk_id = 1

    for pdf_name, pages in pdfs.items():

        # Sort pages by page number just in case extraction order varies
        pages = sorted(pages, key=lambda p: p["page"])

        # Build full document text, remembering the word offset where
        # each page starts so we can map chunks back to a page number.
        full_words = []
        page_boundaries = []  # list of (start_word_index, page_number)

        for page in pages:
            cleaned = clean_text(page["text"])
            page_words = cleaned.split()

            page_boundaries.append((len(full_words), page["page"]))
            full_words.extend(page_words)

        full_text = " ".join(full_words)

        chunks = chunk_text(full_text)

        for chunk in chunks:

            # Find which page this chunk's starting word falls on:
            # the last page boundary whose start_word <= chunk start_word
            chunk_page = page_boundaries[0][1]
            for boundary_start, page_num in page_boundaries:
                if boundary_start <= chunk["start_word"]:
                    chunk_page = page_num
                else:
                    break

            all_chunks.append({
                "chunk_id": chunk_id,
                "pdf_name": pdf_name,
                "page": chunk_page,
                "text": chunk["text"]
            })

            chunk_id += 1

    # Save Chunk File
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    print(f"Total Chunks Created: {len(all_chunks)}")

    return all_chunks


# Run Independently

if __name__ == "__main__":
    create_chunks()