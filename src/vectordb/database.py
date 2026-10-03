import os
import json
import hashlib
import re
import datetime
from typing import List, Dict, Any, Tuple, Optional
import chromadb

from src.loaders.document import Document
from src.loaders.document_loader import DocumentLoader
from src.loaders.faq_loader import FAQLoader
from src.chunking.splitter import DocumentSplitter
from src.embeddings.embedder import Embedder


def compute_file_hash(file_path: str) -> str:
    """Computes MD5 hash of raw file content for change detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_document_id_from_relpath(rel_path: str) -> str:
    """Generates a stable document_id based on relative file path."""
    clean_path = rel_path.replace('\\', '/')
    clean_slug = re.sub(r'[^a-zA-Z0-9_]', '_', clean_path)
    clean_slug = re.sub(r'_+', '_', clean_slug).strip('_')
    return f"doc_{clean_slug}"


class VectorDBManager:
    """
    Manages ChromaDB collections (faq_collection and document_collection).
    Supports per-file incremental indexing (NEW, MODIFIED, UNCHANGED, DELETED)
    with a persistent document manifest registry stored at chroma_db/db_state.json.
    """
    CHUNKING_VERSION = "v1_Recursive_800_150"
    EMBEDDING_VERSION = "v1"

    def __init__(
        self, 
        db_path: str = "chroma_db", 
        embedder: Optional[Embedder] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize Persistent Client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Initialize Embedder
        self.embedder = embedder if embedder is not None else Embedder()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Get or create collections with Cosine distance
        self.faq_collection = self.client.get_or_create_collection(
            name="faq_collection", 
            metadata={"hnsw:space": "cosine"}
        )
        self.document_collection = self.client.get_or_create_collection(
            name="document_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.state_file = os.path.join(self.db_path, "db_state.json")
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Loads persistent document manifest registry from db_state.json."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "documents" in data:
                        return data
            except Exception as e:
                print(f"[VectorDBManager Warning] Failed to parse manifest state file: {e}")

        # Default empty manifest schema
        return {
            "version": "2.0",
            "chunking_version": self.CHUNKING_VERSION,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": getattr(self.embedder, "model_name", "BAAI/bge-m3"),
            "embedding_version": self.EMBEDDING_VERSION,
            "indexed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "faq_hash": "",
            "documents": {}
        }

    def _save_manifest(self) -> None:
        """Saves current manifest state to db_state.json."""
        self.manifest["indexed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.manifest["chunking_version"] = self.CHUNKING_VERSION
        self.manifest["chunk_size"] = self.chunk_size
        self.manifest["chunk_overlap"] = self.chunk_overlap
        self.manifest["embedding_model"] = getattr(self.embedder, "model_name", "BAAI/bge-m3")
        self.manifest["embedding_version"] = self.EMBEDDING_VERSION
        
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2)

    def sync_faq(self, faq_path: str) -> bool:
        """
        Indexes FAQ Excel file into faq_collection if modified or empty.
        Keeps FAQ collection separate from document_collection.
        """
        if not os.path.exists(faq_path):
            return False

        current_faq_hash = compute_file_hash(faq_path)
        saved_faq_hash = self.manifest.get("faq_hash", "")

        if current_faq_hash == saved_faq_hash and self.faq_collection.count() > 0:
            return False  # FAQ unchanged

        print(f"[VectorDBManager] Indexing FAQ dataset from {faq_path}...")
        faq_loader = FAQLoader(faq_path)
        faq_docs = faq_loader.load()

        # Clear existing FAQ collection
        try:
            self.client.delete_collection("faq_collection")
        except Exception:
            pass
        self.faq_collection = self.client.get_or_create_collection(
            name="faq_collection",
            metadata={"hnsw:space": "cosine"}
        )

        splitter = DocumentSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        faq_chunks = splitter.split_documents(faq_docs, print_stats=False)

        faq_texts = [d.page_content for d in faq_chunks]
        faq_embeddings = self.embedder.embed_texts(faq_texts)
        faq_ids = [d.metadata.get("chunk_id", f"faq_{i}") for i, d in enumerate(faq_chunks)]
        faq_metadatas = [d.metadata for d in faq_chunks]

        self._add_to_collection_in_batches(
            collection=self.faq_collection,
            ids=faq_ids,
            embeddings=faq_embeddings,
            metadatas=faq_metadatas,
            documents=faq_texts
        )

        self.manifest["faq_hash"] = current_faq_hash
        self._save_manifest()
        print(f"[VectorDBManager] Indexed {len(faq_chunks)} FAQ entries successfully.")
        return True

    def sync_incremental(
        self, 
        docs_dirs: List[str], 
        faq_path: Optional[str] = None,
        base_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Performs file-level incremental vector indexing across all provided document directories.
        Classifies every file as NEW, MODIFIED, UNCHANGED, or DELETED based on file content hashing.
        """
        if faq_path:
            self.sync_faq(faq_path)

        # Ensure docs_dirs is a list
        if isinstance(docs_dirs, str):
            docs_dirs = [docs_dirs]

        doc_loader = DocumentLoader()
        splitter = DocumentSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        # 1. Discover all files currently present on disk
        discovered_files: Dict[str, Dict[str, Any]] = {}
        
        for ddir in docs_dirs:
            if not os.path.exists(ddir):
                continue
            
            effective_base = base_dir if base_dir is not None else os.path.dirname(os.path.abspath(ddir))

            for root, _, files in os.walk(ddir):
                for fname in sorted(files):
                    if fname.startswith('~$') or fname.startswith('.'):
                        continue
                    
                    full_path = os.path.abspath(os.path.join(root, fname))
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in {'.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg'}:
                        continue

                    try:
                        rel_path = os.path.relpath(full_path, effective_base).replace('\\', '/')
                    except ValueError:
                        rel_path = full_path.replace('\\', '/')

                    doc_id = get_document_id_from_relpath(rel_path)
                    f_hash = compute_file_hash(full_path)

                    discovered_files[doc_id] = {
                        "document_id": doc_id,
                        "full_path": full_path,
                        "relative_path": rel_path,
                        "file_name": fname,
                        "file_type": ext.lstrip('.'),
                        "file_hash": f_hash,
                        "base_dir": effective_base
                    }

        # 2. Check for global chunking or embedding model version changes
        config_mismatch = (
            self.manifest.get("chunking_version") != self.CHUNKING_VERSION or
            self.manifest.get("embedding_model") != getattr(self.embedder, "model_name", "BAAI/bge-m3") or
            self.manifest.get("embedding_version") != self.EMBEDDING_VERSION
        )

        saved_docs: Dict[str, Dict[str, Any]] = self.manifest.get("documents", {})

        # 3. Categorize document lifecycle states
        new_docs: List[Dict[str, Any]] = []
        modified_docs: List[Dict[str, Any]] = []
        unchanged_docs: List[Dict[str, Any]] = []
        deleted_doc_ids: List[str] = []

        # Check existing saved documents for DELETED
        for doc_id, s_info in saved_docs.items():
            if s_info.get("status") == "DELETED":
                continue
            if doc_id not in discovered_files:
                deleted_doc_ids.append(doc_id)

        # Check discovered files for NEW, MODIFIED, or UNCHANGED
        for doc_id, d_info in discovered_files.items():
            if doc_id not in saved_docs or saved_docs[doc_id].get("status") == "DELETED":
                new_docs.append(d_info)
            else:
                s_info = saved_docs[doc_id]
                if d_info["file_hash"] != s_info.get("file_hash") or config_mismatch:
                    modified_docs.append(d_info)
                else:
                    unchanged_docs.append(d_info)

        # Counters for report
        chunks_deleted = 0
        chunks_created = 0
        errors: List[str] = []

        # 4. Handle DELETED Documents
        for doc_id in deleted_doc_ids:
            s_info = saved_docs[doc_id]
            old_chunk_ids = s_info.get("chunk_ids", [])
            if old_chunk_ids:
                try:
                    self.document_collection.delete(ids=old_chunk_ids)
                    chunks_deleted += len(old_chunk_ids)
                except Exception as e:
                    errors.append(f"Failed to delete chunks for deleted doc {doc_id}: {e}")

            saved_docs[doc_id]["status"] = "DELETED"
            saved_docs[doc_id]["chunk_count"] = 0
            saved_docs[doc_id]["chunk_ids"] = []

        # 5. Handle MODIFIED Documents
        for d_info in modified_docs:
            doc_id = d_info["document_id"]
            if doc_id in saved_docs:
                old_chunk_ids = saved_docs[doc_id].get("chunk_ids", [])
                if old_chunk_ids:
                    try:
                        self.document_collection.delete(ids=old_chunk_ids)
                        chunks_deleted += len(old_chunk_ids)
                    except Exception as e:
                        errors.append(f"Failed to delete old chunks for modified doc {doc_id}: {e}")

            # Re-index document
            created_count, new_ids = self._index_single_document(d_info, doc_loader, splitter)
            chunks_created += created_count

            saved_docs[doc_id] = {
                "document_id": doc_id,
                "source_path": d_info["relative_path"],
                "file_name": d_info["file_name"],
                "file_type": d_info["file_type"],
                "file_hash": d_info["file_hash"],
                "chunk_count": created_count,
                "chunk_ids": new_ids,
                "chunking_version": self.CHUNKING_VERSION,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": getattr(self.embedder, "model_name", "BAAI/bge-m3"),
                "embedding_version": self.EMBEDDING_VERSION,
                "indexed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": "INDEXED"
            }

        # 6. Handle NEW Documents
        for d_info in new_docs:
            doc_id = d_info["document_id"]
            created_count, new_ids = self._index_single_document(d_info, doc_loader, splitter)
            chunks_created += created_count

            saved_docs[doc_id] = {
                "document_id": doc_id,
                "source_path": d_info["relative_path"],
                "file_name": d_info["file_name"],
                "file_type": d_info["file_type"],
                "file_hash": d_info["file_hash"],
                "chunk_count": created_count,
                "chunk_ids": new_ids,
                "chunking_version": self.CHUNKING_VERSION,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": getattr(self.embedder, "model_name", "BAAI/bge-m3"),
                "embedding_version": self.EMBEDDING_VERSION,
                "indexed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": "INDEXED"
            }

        # Save manifest update
        self.manifest["documents"] = saved_docs
        self._save_manifest()

        # Compute summary totals
        total_indexed_docs = sum(1 for d in saved_docs.values() if d.get("status") == "INDEXED")
        total_indexed_chunks = self.document_collection.count()

        report = {
            "scanned_documents": len(discovered_files),
            "new": len(new_docs),
            "modified": len(modified_docs),
            "unchanged": len(unchanged_docs),
            "deleted": len(deleted_doc_ids),
            "chunks_deleted": chunks_deleted,
            "chunks_created": chunks_created,
            "total_indexed_documents": total_indexed_docs,
            "total_indexed_chunks": total_indexed_chunks,
            "errors": errors
        }

        self._print_indexing_report(report)
        return report

    def _index_single_document(
        self, 
        d_info: Dict[str, Any], 
        doc_loader: DocumentLoader, 
        splitter: DocumentSplitter
    ) -> Tuple[int, List[str]]:
        """Helper to load, split, embed, and store a single document."""
        full_path = d_info["full_path"]
        base_dir = d_info["base_dir"]
        doc = doc_loader.load_file(full_path, base_dir=base_dir)
        if not doc:
            return 0, []

        doc.metadata["document_id"] = d_info["document_id"]
        chunks = splitter.split_documents([doc], print_stats=False)
        if not chunks:
            return 0, []

        chunk_texts = [c.page_content for c in chunks]
        chunk_embeddings = self.embedder.embed_texts(chunk_texts)
        chunk_ids = [c.metadata.get("chunk_id", f"{d_info['document_id']}_chunk_{i}") for i, c in enumerate(chunks)]
        chunk_metadatas = [c.metadata for c in chunks]

        self._add_to_collection_in_batches(
            collection=self.document_collection,
            ids=chunk_ids,
            embeddings=chunk_embeddings,
            metadatas=chunk_metadatas,
            documents=chunk_texts
        )

        return len(chunks), chunk_ids

    def reindex_document(self, document_id_or_path: str, docs_dirs: List[str], base_dir: Optional[str] = None) -> bool:
        """
        Forces re-indexing of a single specified document.
        """
        saved_docs = self.manifest.get("documents", {})
        target_doc_id = None

        if document_id_or_path in saved_docs:
            target_doc_id = document_id_or_path
        else:
            # Check by relative path or file name
            for did, dinfo in saved_docs.items():
                if dinfo.get("source_path") == document_id_or_path or dinfo.get("file_name") == document_id_or_path:
                    target_doc_id = did
                    break

        if not target_doc_id:
            # Check if it's a new file path
            rel_path = document_id_or_path.replace('\\', '/')
            target_doc_id = get_document_id_from_relpath(rel_path)

        if target_doc_id in saved_docs:
            # Mark file_hash invalid to trigger MODIFIED status
            saved_docs[target_doc_id]["file_hash"] = "FORCE_REINDEX"

        self._save_manifest()
        self.sync_incremental(docs_dirs=docs_dirs, base_dir=base_dir)
        return True

    def rebuild_database(
        self, 
        faq_documents: List[Document], 
        doc_chunks: List[Document], 
        faq_path: str, 
        docs_dir: str
    ):
        """
        Full rebuild mode: clears all collections and populates from scratch.
        """
        print("[VectorDBManager] Full rebuild of ChromaDB collections...")
        
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

        if faq_documents:
            print(f"Adding {len(faq_documents)} FAQ entries...")
            faq_texts = [d.page_content for d in faq_documents]
            faq_embeddings = self.embedder.embed_texts(faq_texts)
            faq_ids = [d.metadata.get("chunk_id", f"faq_{i}") for i, d in enumerate(faq_documents)]
            faq_metadatas = [d.metadata for d in faq_documents]
            
            self._add_to_collection_in_batches(
                collection=self.faq_collection,
                ids=faq_ids,
                embeddings=faq_embeddings,
                metadatas=faq_metadatas,
                documents=faq_texts
            )

        saved_docs = {}
        if doc_chunks:
            print(f"Adding {len(doc_chunks)} Document chunks...")
            doc_texts = [d.page_content for d in doc_chunks]
            doc_embeddings = self.embedder.embed_texts(doc_texts)
            doc_ids = [d.metadata.get("chunk_id", f"doc_{i}") for i, d in enumerate(doc_chunks)]
            doc_metadatas = [d.metadata for d in doc_chunks]
            
            self._add_to_collection_in_batches(
                collection=self.document_collection,
                ids=doc_ids,
                embeddings=doc_embeddings,
                metadatas=doc_metadatas,
                documents=doc_texts
            )

            # Group chunks by document_id to populate manifest
            for chunk in doc_chunks:
                did = chunk.metadata.get("document_id", "doc_unknown")
                if did not in saved_docs:
                    saved_docs[did] = {
                        "document_id": did,
                        "source_path": chunk.metadata.get("source_path", chunk.metadata.get("filename", "")),
                        "file_name": chunk.metadata.get("file_name", chunk.metadata.get("filename", "")),
                        "file_type": chunk.metadata.get("document_type", "docx"),
                        "file_hash": "full_rebuild",
                        "chunk_count": 0,
                        "chunk_ids": [],
                        "chunking_version": self.CHUNKING_VERSION,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "embedding_model": getattr(self.embedder, "model_name", "BAAI/bge-m3"),
                        "embedding_version": self.EMBEDDING_VERSION,
                        "indexed_at": datetime.datetime.utcnow().isoformat() + "Z",
                        "status": "INDEXED"
                    }
                saved_docs[did]["chunk_count"] += 1
                saved_docs[did]["chunk_ids"].append(chunk.metadata.get("chunk_id"))

        self.manifest = {
            "version": "2.0",
            "chunking_version": self.CHUNKING_VERSION,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": getattr(self.embedder, "model_name", "BAAI/bge-m3"),
            "embedding_version": self.EMBEDDING_VERSION,
            "indexed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "faq_hash": compute_file_hash(faq_path) if os.path.exists(faq_path) else "",
            "documents": saved_docs
        }
        self._save_manifest()
        print("Database full rebuild completed.")

    def needs_rebuild(self, faq_path: str, docs_dir: str) -> bool:
        """Determines if any collection is empty or state manifest missing."""
        if not os.path.exists(self.state_file):
            return True
        if self.faq_collection.count() == 0 or self.document_collection.count() == 0:
            return True
        return False

    def _add_to_collection_in_batches(self, collection, ids, embeddings, metadatas, documents, batch_size=100):
        """Helper to batch inserts to prevent memory spikes or API limit issues."""
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            
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

    def _print_indexing_report(self, report: Dict[str, Any]):
        """Prints formatted indexing summary report."""
        print("=========================================")
        print("INCREMENTAL INDEXING REPORT")
        print("=========================================")
        print(f"Documents Scanned:        {report['scanned_documents']}")
        print(f"  NEW:                    {report['new']}")
        print(f"  MODIFIED:               {report['modified']}")
        print(f"  UNCHANGED:              {report['unchanged']}")
        print(f"  DELETED:                {report['deleted']}")
        print("-----------------------------------------")
        print(f"Chunks Deleted:           {report['chunks_deleted']}")
        print(f"Chunks Created:           {report['chunks_created']}")
        print(f"Total Indexed Documents:  {report['total_indexed_documents']}")
        print(f"Total Indexed Chunks:     {report['total_indexed_chunks']}")
        if report['errors']:
            print("-----------------------------------------")
            print("Errors Encountered:")
            for err in report['errors']:
                print(f"  - {err}")
        print("=========================================")

    def count(self) -> Tuple[int, int]:
        """Returns (faq_count, doc_count)."""
        return self.faq_collection.count(), self.document_collection.count()

    def get_indexed_documents(self) -> List[str]:
        """
        Returns a sorted list of unique filenames for all indexed documents.
        """
        if hasattr(self, "manifest") and isinstance(self.manifest, dict) and "documents" in self.manifest:
            docs = [d.get("file_name") for d in self.manifest["documents"].values() if d.get("file_name")]
            if docs:
                return sorted(list(set(docs)))
        try:
            res = self.document_collection.get(include=["metadatas"])
            if res and res.get("metadatas"):
                docs = set(m.get("source") or m.get("filename") for m in res["metadatas"] if m.get("source") or m.get("filename"))
                return sorted(list(docs))
        except Exception:
            pass
        return []

    def get_document_chunks(self, target_document: str) -> List[Dict[str, Any]]:
        """
        Retrieves all chunks strictly belonging to target_document from document_collection.
        Chunks are sorted by chunk_index in ascending natural document order.
        """
        if not target_document:
            return []

        # Try matching by source metadata
        res = self.document_collection.get(
            where={"source": target_document},
            include=["documents", "metadatas"]
        )

        # Fallback to filename metadata if no chunks matched source
        if not res or not res.get("ids"):
            res = self.document_collection.get(
                where={"filename": target_document},
                include=["documents", "metadatas"]
            )

        if not res or not res.get("ids"):
            return []

        chunks = []
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            c_idx = meta.get("chunk_index", 0)
            chunks.append({
                "chunk_id": cid,
                "content": doc,
                "metadata": meta,
                "score": 1.0,
                "retrieval_score": 1.0,
                "rerank_score": None,
                "chunk_index": c_idx,
                "type": "document"
            })

        # Sort in natural document order by chunk_index
        chunks.sort(key=lambda x: x["chunk_index"])
        return chunks

