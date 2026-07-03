from read_pdf import read_pdfs
from chunk import create_chunks
from embedding import generate_embeddings
from faiss_store import build_faiss



def main():

    print("=" * 60)
    print("        MultiPDF RAG Assistant")
    print("=" * 60)

    # Step 1 : Read PDFs
   
    print("\n[1/4] Reading PDF files...")
    read_pdfs()


    # Step 2 : Chunk Text

    print("\n[2/4] Creating text chunks...")
    create_chunks()


    # Step 3 : Generate Embeddings
 
    print("\n[3/4] Generating embeddings...")
    generate_embeddings()


    # Step 4 : Build FAISS Index

    print("\n[4/4] Building FAISS index...")
    build_faiss()

    print("\n✅ System is Ready!")
    print("-" * 60)

    from query import ask_question

    # Question Loop

    while True:

        question = input("\nAsk a Question (type 'exit' to quit): ")

        if question.lower() == "exit":
            print("\nThank you for using MultiPDF RAG Assistant.")
            break

        answer, sources = ask_question(question)


        print("\n" + "=" * 60)
        print("Answer")
        print("=" * 60)
        print(answer)

        print("\n" + "=" * 60)
        print("Sources")
        print("=" * 60)

        for source in sources:
            print(f"{source['pdf_name']} (Page {source['page']})")


if __name__ == "__main__":
    main()