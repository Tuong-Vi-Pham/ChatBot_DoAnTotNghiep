import os
import sys

# Add project root to path to run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.loaders.faq_loader import FAQLoader
from src.loaders.document_loader import DocumentLoader
from src.chunking.splitter import DocumentSplitter
from src.embeddings.embedder import Embedder
from src.vectordb.database import VectorDBManager

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    faq_path = os.path.join(base_dir, "data/questions/Dataset_QandA.xlsx")
    docs_dir = os.path.join(base_dir, "data/documents/Tech_Team")
    db_path = os.path.join(base_dir, "chroma_db")
    
    print("=========================================")
    print("PHASE 4 VERIFICATION: CHROMADB COLLECTIONS")
    print("=========================================")
    
    # 1. Initialize Embedder and Database
    print("Initializing components...")
    embedder = Embedder()
    db_manager = VectorDBManager(db_path=db_path, embedder=embedder)
    
    # 2. Check if rebuild is needed
    needs_rebuild = db_manager.needs_rebuild(faq_path, docs_dir)
    print(f"Initial check: Needs rebuild? {needs_rebuild}")
    
    if needs_rebuild:
        print("\nLoading and splitting data for database population...")
        faq_loader = FAQLoader(faq_path)
        doc_loader = DocumentLoader(show_ocr_log=False)
        
        try:
            faq_docs = faq_loader.load()
            documents = doc_loader.load_directory(docs_dir)
            all_docs = faq_docs + documents
            
            splitter = DocumentSplitter(chunk_size=800, chunk_overlap=150)
            chunks = splitter.split_documents(all_docs, print_stats=False)
            
            faq_chunks = [c for c in chunks if c.metadata.get("source_type") == "faq"]
            doc_chunks = [c for c in chunks if c.metadata.get("source_type") == "document"]
            
            # Rebuild database
            db_manager.rebuild_database(
                faq_documents=faq_chunks,
                doc_chunks=doc_chunks,
                faq_path=faq_path,
                docs_dir=docs_dir
            )
        except Exception as e:
            print(f"[ERROR] Rebuild failed: {e}")
            return
    else:
        print("[OK] Existing database found, skipped loading/rebuilding.")
        
    # 3. Verify counts
    faq_count, doc_count = db_manager.count()
    print(f"\nCollection counts after first build:")
    print(f"  faq_collection:      {faq_count} entities")
    print(f"  document_collection: {doc_count} entities")
    
    if faq_count == 260:
        print("  [OK] faq_collection count is correct (260).")
    else:
        print(f"  [FAIL] faq_collection count mismatch! Expected 260, got {faq_count}")
        
    if doc_count > 0:
        print(f"  [OK] document_collection has data ({doc_count} chunks).")
    else:
        print("  [FAIL] document_collection is empty!")

    # 4. Initialize database manager again and check if it skips rebuild
    print("\nRe-initializing Database Manager to test skip rebuild logic...")
    db_manager_2 = VectorDBManager(db_path=db_path, embedder=embedder)
    needs_rebuild_2 = db_manager_2.needs_rebuild(faq_path, docs_dir)
    print(f"Second check (no file changes): Needs rebuild? {needs_rebuild_2}")
    
    if not needs_rebuild_2:
        print("  [OK] Correctly skipped rebuilding since source files are unchanged.")
    else:
        print("  [FAIL] Incorrectly triggered rebuild when no changes occurred!")
        
    # 5. Verify basic query capabilities of ChromaDB
    print("\nVerifying basic query retrieval on collections...")
    try:
        faq_sample_query = embedder.embed_query("What is the project budget?")
        results = db_manager_2.faq_collection.query(
            query_embeddings=[faq_sample_query],
            n_results=1
        )
        print("  [OK] Successfully queried faq_collection.")
        if results and results['documents'] and results['documents'][0]:
            print(f"    Match document: {results['documents'][0][0]}")
            print(f"    Match distance: {results['distances'][0][0] if 'distances' in results else 'N/A'}")
    except Exception as e:
        print(f"  [FAIL] Query test failed: {e}")
        
    print("=========================================")

if __name__ == "__main__":
    main()
