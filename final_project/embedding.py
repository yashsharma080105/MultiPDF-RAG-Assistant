import json
import numpy as np # store array
import os
from sentence_transformers import SentenceTransformer#class loads a pre-trained 
#embedding model that converts text into vectors

# File Paths

INPUT_FILE = "output/chunks.json"

EMBEDDING_FOLDER = "embeddings"  # create folder name
os.makedirs(EMBEDDING_FOLDER, exist_ok=True)

EMBEDDING_FILE = os.path.join(  # store for vector
    EMBEDDING_FOLDER,
    "embeddings.npy"
)

METADATA_FILE = os.path.join(   # store info about every chunk
    EMBEDDING_FOLDER,
    "metadata.json"
)



# Generate Embeddings


def generate_embeddings():

    print("\nLoading Embedding Model...")

    model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Embedding Model Loaded Successfully")


    # Read Chunks

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = []
    metadata = []

    print("\nGenerating Embeddings...\n")

    for chunk in chunks:  # process one chunk at a time

        vector = model.encode(  # convert tect into an embedding vector
            chunk["text"],
            convert_to_numpy=True  # fast
        )

        embeddings.append(vector)

        metadata.append({  # vector contain numbers

            "chunk_id": chunk["chunk_id"],
            "pdf_name": chunk["pdf_name"],
            "page": chunk["page"],
            "text": chunk["text"]

        })

    embeddings = np.array(   #  convert to numpy array
        embeddings,
        dtype="float32"   # bcz embedding contain decimal number
    )

    # Save Embeddings


    np.save(
        EMBEDDING_FILE,
        embeddings
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nEmbeddings Generated Successfully")
    print("Total Embeddings :", len(embeddings))
    print("Embedding Dimension :", embeddings.shape[1])

    return embeddings, metadata



# Run Independently


if __name__ == "__main__":
    generate_embeddings()