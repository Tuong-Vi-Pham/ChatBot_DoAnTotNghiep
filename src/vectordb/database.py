import os
import json
import hashlib
from typing import List, Dict, Any, Tuple
import chromadb
from src.loaders.document import Document
from src.embeddings.embedder import Embedder

class VectorDBManager:
    """
    Manages ChromaDB collections (faq_collection and document_collection).
    Ensures data persistence and implements incremental rebuild logic based on file changes.
    """
    def __init__(self, db_path: str = "chroma_db", embedder: Embedder = None):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize Persistent Client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Initialize Embedder
        self.embedder = embedder if embedder is not None else Embedder()
        
        # Get or create collections
        # We use Cosine distance for text similarity searches
        self.faq_collection = self.client.get_or_create_collection(
            name="faq_collection", 
            metadata={"hnsw:space": "cosine"}
        )
        self.document_collection = self.client.get_or_create_collection(
            name="document_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.state_file = os.path.join(self.db_path, "db_state.json")

    def get_source_state_signature(self, faq_path: str, docs_dir: str) -> Dict[str, Any]:
        """
        Scans all files in faq_path and docs_dir to generate a signature 
        based on file paths, modification times, and sizes.
        """
        files_state = {}
        
        # Check FAQ file
        if os.path.exists(faq_path):
            stat = os.stat(faq_path)
            files_state[os.path.basename(faq_path)] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size
            }
            
        # Check Tech_Team documents
        if os.path.exists(docs_dir):
            for root, _, files in os.walk(docs_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    stat = os.stat(file_path)
                    rel_path = os.path.relpath(file_path, docs_dir)
                    files_state[rel_path] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size
                    }
                    
        # Compute combined hash of the files state
        state_str = json.dumps(files_state, sort_keys=True)
        state_hash = hashlib.md5(state_str.encode('utf-8')).hexdigest()
        
        return {
            "hash": state_hash,
            "files": files_state,
            "faq_count": self.faq_collection.count(),
            "doc_count": self.document_collection.count()
        }

    def needs_rebuild(self, faq_path: str, docs_dir: str) -> bool:
        """
        Determines if the database needs to be rebuilt by comparing
        the current state signature with the persisted state.
        """
        if not os.path.exists(self.state_file):
            return True
            
        # Check if collections are empty
        if self.faq_collection.count() == 0 and self.document_collection.count() == 0:
            return True
            
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                saved_state = json.load(f)
                
            current_state = self.get_source_state_signature(faq_path, docs_dir)
            
            # Check hash and count match
            if saved_state.get("hash") != current_state["hash"]:
                return True
            if saved_state.get("faq_count") != current_state["faq_count"]:
                return True
            if saved_state.get("doc_count") != current_state["doc_count"]:
                return True
                
            return False
        except Exception:
            return True  # Rebuild if state file is corrupted

    def rebuild_database(self, faq_documents: List[Document], doc_chunks: List[Document], faq_path: str, docs_dir: str):
        """
        Clears existing data and populates collections with new documents and embeddings.
        """
        print("Rebuilding ChromaDB collections...")
        
        # 1. Clear old collections by recreating them
        try:
            self.client.delete_collection(name="faq_collection")
        except Exception:
            pass
        try:
            self.client.delete_collection(name="document_collection")
        except Exception:
            pass
            
        self.faq_collection = self.client.get_or_create_collection(
            name="faq_collection", 
            metadata={"hnsw:space": "cosine"}
        )
        self.document_collection = self.client.get_or_create_collection(
            name="document_collection",
            metadata={"hnsw:space": "cosine"}
        )

        # 2. Add FAQ Documents
        if faq_documents:
            print(f"Adding {len(faq_documents)} FAQ entries...")
            faq_texts = [d.page_content for d in faq_documents]
            faq_embeddings = self.embedder.embed_texts(faq_texts)
            faq_ids = [f"faq_{i}" for i in range(len(faq_documents))]
            faq_metadatas = [d.metadata for d in faq_documents]
            
            # ChromaDB expects primitive types for metadatas, clean/verify types
            self._add_to_collection_in_batches(
                collection=self.faq_collection,
                ids=faq_ids,
                embeddings=faq_embeddings,
                metadatas=faq_metadatas,
                documents=faq_texts
            )

        # 3. Add Document Chunks
        if doc_chunks:
            print(f"Adding {len(doc_chunks)} Document chunks...")
            doc_texts = [d.page_content for d in doc_chunks]
            doc_embeddings = self.embedder.embed_texts(doc_texts)
            doc_ids = [f"doc_{i}" for i in range(len(doc_chunks))]
            doc_metadatas = [d.metadata for d in doc_chunks]
            
            self._add_to_collection_in_batches(
                collection=self.document_collection,
                ids=doc_ids,
                embeddings=doc_embeddings,
                metadatas=doc_metadatas,
                documents=doc_texts
            )

        # 4. Persist and save state file
        new_state = self.get_source_state_signature(faq_path, docs_dir)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(new_state, f, indent=2)
        print("Database rebuilt and state signature saved.")

    def _add_to_collection_in_batches(self, collection, ids, embeddings, metadatas, documents, batch_size=100):
        """
        Helper to batch inserts to prevent memory spikes or API limit issues.
        """
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            
            # Clean metadatas to ensure all keys/values are primitive types
            batch_metadatas = []
            for meta in metadatas[i:end_idx]:
                clean_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)
                batch_metadatas.append(clean_meta)

            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=batch_metadatas,
                documents=documents[i:end_idx]
            )
            
    def count(self) -> Tuple[int, int]:
        """Returns (faq_count, doc_count)"""
        return self.faq_collection.count(), self.document_collection.count()
