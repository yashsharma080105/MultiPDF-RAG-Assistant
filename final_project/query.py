import faiss  # use for searching the vector database
import json
import numpy as np  # embedding purpose
from sentence_transformers import SentenceTransformer  # load embedding model

from llm import generate_answer

# Load Embedding Model
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Load FAISS Index

index = faiss.read_index("vector_db/faiss.index")

# Load Metadata

with open("embeddings/metadata.json", "r", encoding="utf-8") as f:  ## utf-8 translate text char into 1 and 0
    metadata = json.load(f)



# Ask Question

def ask_question(query, k=8):  # k=8 retrieve top 8 chunk

    # Convert question into embedding
    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    ).astype("float32")
    # expand dimansion
    query_embedding = np.expand_dims(query_embedding, axis=0)

    # Normalize the query the same way the stored embeddings were
    # normalized in faiss_store.py, so cosine similarity is consistent.
    faiss.normalize_L2(query_embedding)

    # Search FAISS=> compares  the query with every stored vector
    # and return distance and indiced like 
    # distance contain similarity scores and indices = which vector matched
    distances, indices = index.search(query_embedding, k)

    context = ""
    sources = []

    print("\nRetrieved Chunks:\n")

    for score, idx in zip(distances[0], indices[0]):  #loop through result

        # Skip invalid indices
        if idx == -1:
            continue

        chunk = metadata[idx]
        ## print simialrity score
        print(f"[score={score:.3f}] {chunk['pdf_name']} (Page {chunk['page']})")
        print(chunk["text"][:200].replace("\n", " ") + "...\n")

        context += chunk["text"] + "\n\n"

        sources.append({
            "pdf_name": chunk["pdf_name"],
            "page": chunk["page"]
        })

    # No results found
    if context.strip() == "":
        return (
            "No relevant information found in the uploaded PDFs.",
            []
        )

    # Generate answer using Gemini
    answer = generate_answer(query, context)

    return answer, sources



# Test Query File Directly

if __name__ == "__main__":

    while True:

        question = input("\nAsk a Question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        answer, sources = ask_question(question)

        print("\nAnswer\n")
        print(answer)

        print("\nSources")
        for source in sources:  # print source like pdf name  and page 
            print(
                f"{source['pdf_name']} (Page {source['page']})"
            )



## metadata.json contain all the pdf name