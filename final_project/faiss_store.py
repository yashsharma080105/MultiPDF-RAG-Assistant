import faiss
import numpy as np  # store and loading embedding vecotrs
import os

EMBEDDING_FILE = "embeddings/embeddings.npy"

VECTOR_DB = "vector_db"
os.makedirs(VECTOR_DB, exist_ok=True)  # create folder if exit no error occur

INDEX_FILE = os.path.join(  # file store the faiss index
    VECTOR_DB,
    "faiss.index"
)



# Build FAISS Index


def build_faiss():

   
    # Load Embeddings
   
    embeddings = np.load(EMBEDDING_FILE)

    print("\nEmbeddings Loaded")
    print("Shape :", embeddings.shape)

  
    # Convert Data Type
    
    embeddings = embeddings.astype("float32")

    
    # Normalize Embeddings (L2 norm)
    
    faiss.normalize_L2(embeddings)  # convert embedding into unit vector

    
    # Embedding Dimension
   
    dimension = embeddings.shape[1]

    print("Embedding Dimension :", dimension)

    
    # Create FAISS Index (cosine similarity via inner product)
    
    index = faiss.IndexFlatIP(dimension)

    
    # Add Embeddings
    
    index.add(embeddings)

    print("Vectors Stored :", index.ntotal)

  
    # Save FAISS Index
    
    faiss.write_index(index, INDEX_FILE)

    print("\nFAISS Index Saved Successfully")

    return index



if __name__ == "__main__":
    build_faiss()