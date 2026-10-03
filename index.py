import os
import sys
import argparse

# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch  # Import torch first to avoid shm loading conflicts on CPU
from src.embeddings.embedder import Embedder
from src.vectordb.database import VectorDBManager
from src.loaders.faq_loader import FAQLoader
from src.loaders.document_loader import DocumentLoader
from src.chunking.splitter import DocumentSplitter


def run_indexing(mode: str = "incremental", document_target: str = None, db_path: str = None):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    crm_dir = os.path.join(data_dir, "documents/CRM")
    tech_dir = os.path.join(data_dir, "documents/Tech_Team")
    docs_dirs = [crm_dir, tech_dir]
    
    faq_path = os.path.join(data_dir, "questions/Dataset_QandA.xlsx")
    target_db_path = db_path if db_path else os.path.join(base_dir, "chroma_db")
    documents_base_dir = os.path.join(data_dir, "documents")

    print("=============================================================")
    print(f"       SMARTLOGI RAG KNOWLEDGE BASE INDEXER ({mode.upper()})     ")
    print("=============================================================")
    print(f"Target DB Path:     {target_db_path}")
    print(f"Knowledge Sources:  CRM ({crm_dir})")
    print(f"                    Tech_Team ({tech_dir})")
    print(f"FAQ Dataset:        {faq_path}")
    print("=============================================================\n")

    embedder = Embedder(model_name="BAAI/bge-m3")
    db_manager = VectorDBManager(db_path=target_db_path, embedder=embedder)

    if mode == "full":
        print("[FULL REBUILD MODE] Loading all documents from scratch...")
        faq_docs = []
        if os.path.exists(faq_path):
            faq_loader = FAQLoader(faq_path)
            faq_docs = faq_loader.load()

        doc_loader = DocumentLoader()
        all_docs = []
        for ddir in docs_dirs:
            if os.path.exists(ddir):
                all_docs.extend(doc_loader.load_directory(ddir, base_dir=documents_base_dir))

        splitter = DocumentSplitter(chunk_size=800, chunk_overlap=150)
        all_chunks = splitter.split_documents(faq_docs + all_docs, print_stats=True)

        faq_chunks = [c for c in all_chunks if c.metadata.get("source_type") == "faq"]
        doc_chunks = [c for c in all_chunks if c.metadata.get("source_type") != "faq"]

        db_manager.rebuild_database(
            faq_documents=faq_chunks,
            doc_chunks=doc_chunks,
            faq_path=faq_path,
            docs_dir=tech_dir
        )
        report = {
            "mode": "full",
            "faq_chunks": len(faq_chunks),
            "doc_chunks": len(doc_chunks),
            "total_chunks": db_manager.document_collection.count() + db_manager.faq_collection.count()
        }
        return report

    elif mode == "document":
        if not document_target:
            print("[ERROR] '--document' argument is required when mode is 'document'.")
            sys.exit(1)
        print(f"[SINGLE DOCUMENT MODE] Re-indexing target: '{document_target}'...")
        db_manager.reindex_document(document_target, docs_dirs=docs_dirs, base_dir=documents_base_dir)
        return {"mode": "document", "target": document_target}

    else:
        # Default: Incremental Mode
        print("[INCREMENTAL MODE] Synchronizing document state manifest...")
        report = db_manager.sync_incremental(
            docs_dirs=docs_dirs,
            faq_path=faq_path,
            base_dir=documents_base_dir
        )
        return report


def main():
    parser = argparse.ArgumentParser(description="SmartLogi RAG Knowledge Base Indexer CLI")
    parser.add_argument(
        "--mode",
        choices=["incremental", "full", "document"],
        default="incremental",
        help="Indexing mode: 'incremental' (default), 'full' (rebuild), or 'document' (re-index target document)"
    )
    parser.add_argument(
        "--document",
        type=str,
        default=None,
        help="Target document ID or path when --mode is 'document'"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Custom ChromaDB directory path"
    )

    args = parser.parse_args()
    run_indexing(mode=args.mode, document_target=args.document, db_path=args.db_path)


if __name__ == "__main__":
    main()
