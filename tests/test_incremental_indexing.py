import os
import sys
import shutil
import tempfile
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.vectordb.database import VectorDBManager
from src.loaders.document_loader import DocumentLoader
from src.chunking.splitter import DocumentSplitter


class DummyEmbedder:
    """Fast dummy embedder for unit test isolation."""
    def __init__(self, model_name="BAAI/bge-m3", dim=1024):
        self.model_name = model_name
        self.dim = dim

    def embed_texts(self, texts):
        return [[0.1] * self.dim for _ in texts]

    def embed_query(self, query):
        return [0.1] * self.dim

    def get_dimension(self):
        return self.dim


class TestIncrementalIndexing(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.crm_dir = os.path.join(self.temp_dir, "data", "documents", "CRM")
        self.tech_dir = os.path.join(self.temp_dir, "data", "documents", "Tech_Team")
        self.db_path = os.path.join(self.temp_dir, "chroma_db")
        self.base_dir = os.path.join(self.temp_dir, "data", "documents")

        os.makedirs(self.crm_dir, exist_ok=True)
        os.makedirs(self.tech_dir, exist_ok=True)
        os.makedirs(self.db_path, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_new_document(self):
        """Test 1: Detects NEW document and adds chunks to ChromaDB."""
        embedder = DummyEmbedder()
        db_manager = VectorDBManager(db_path=self.db_path, embedder=embedder)

        doc1_path = os.path.join(self.crm_dir, "test_brd.txt")
        with open(doc1_path, "w", encoding="utf-8") as f:
            f.write("Business Requirements Document for SmartLogi CRM Module.\n" * 10)

        report = db_manager.sync_incremental(
            docs_dirs=[self.crm_dir],
            base_dir=self.base_dir
        )

        self.assertEqual(report["new"], 1)
        self.assertEqual(report["unchanged"], 0)
        self.assertEqual(report["modified"], 0)
        self.assertGreater(report["chunks_created"], 0)
        self.assertEqual(db_manager.document_collection.count(), report["chunks_created"])

    def test_02_unchanged_document(self):
        """Test 2: Detects UNCHANGED document and skips re-embedding/writes."""
        embedder = DummyEmbedder()
        db_manager = VectorDBManager(db_path=self.db_path, embedder=embedder)

        doc1_path = os.path.join(self.crm_dir, "test_brd.txt")
        with open(doc1_path, "w", encoding="utf-8") as f:
            f.write("Business Requirements Document for SmartLogi CRM Module.\n" * 10)

        # Initial sync
        db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        # Second sync (no changes)
        report2 = db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        self.assertEqual(report2["new"], 0)
        self.assertEqual(report2["modified"], 0)
        self.assertEqual(report2["unchanged"], 1)
        self.assertEqual(report2["chunks_created"], 0)
        self.assertEqual(report2["chunks_deleted"], 0)

    def test_03_modified_document(self):
        """Test 3: Detects MODIFIED document, deletes old chunks, and adds new chunks."""
        embedder = DummyEmbedder()
        db_manager = VectorDBManager(db_path=self.db_path, embedder=embedder)

        doc1_path = os.path.join(self.crm_dir, "test_brd.txt")
        # Initial long content to generate multiple chunks
        with open(doc1_path, "w", encoding="utf-8") as f:
            f.write("Original content for BRD document with detailed sections and specifications.\n" * 40)

        # Initial sync
        db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        doc_id = "doc_CRM_test_brd_txt"
        old_chunks = db_manager.document_collection.get(where={"document_id": doc_id})["ids"]
        initial_chunk_count = len(old_chunks)
        self.assertGreater(initial_chunk_count, 1)

        # Modify file content to short text (fewer chunks)
        with open(doc1_path, "w", encoding="utf-8") as f:
            f.write("Short modified content for BRD document.\n")

        # Second sync
        report2 = db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        self.assertEqual(report2["modified"], 1)
        self.assertEqual(report2["chunks_deleted"], initial_chunk_count)

        new_chunks = db_manager.document_collection.get(where={"document_id": doc_id})["ids"]
        self.assertEqual(len(new_chunks), report2["chunks_created"])
        self.assertLess(len(new_chunks), initial_chunk_count)

        # Verify old tail chunks that are no longer part of new_chunks do NOT exist
        obsolete_chunks = set(old_chunks) - set(new_chunks)
        self.assertGreater(len(obsolete_chunks), 0)
        for obs_id in obsolete_chunks:
            self.assertEqual(len(db_manager.document_collection.get(ids=[obs_id])["ids"]), 0)

    def test_04_deleted_document(self):
        """Test 4: Detects DELETED document and purges all associated chunks."""
        embedder = DummyEmbedder()
        db_manager = VectorDBManager(db_path=self.db_path, embedder=embedder)

        doc1_path = os.path.join(self.crm_dir, "temp_doc.txt")
        with open(doc1_path, "w", encoding="utf-8") as f:
            f.write("Temporary document content to be deleted.\n" * 10)

        db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        doc_id = "doc_CRM_temp_doc_txt"
        self.assertGreater(len(db_manager.document_collection.get(where={"document_id": doc_id})["ids"]), 0)

        # Delete file on disk
        os.remove(doc1_path)

        # Sync
        report2 = db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        self.assertEqual(report2["deleted"], 1)
        self.assertGreater(report2["chunks_deleted"], 0)
        self.assertEqual(len(db_manager.document_collection.get(where={"document_id": doc_id})["ids"]), 0)

    def test_05_duplicate_file_content(self):
        """Test 5: Handles duplicate file content in different paths with distinct document_ids."""
        embedder = DummyEmbedder()
        db_manager = VectorDBManager(db_path=self.db_path, embedder=embedder)

        content = "Identical content in two separate files.\n" * 10
        doc1_path = os.path.join(self.crm_dir, "doc_a.txt")
        doc2_path = os.path.join(self.tech_dir, "doc_b.txt")

        with open(doc1_path, "w", encoding="utf-8") as f:
            f.write(content)
        with open(doc2_path, "w", encoding="utf-8") as f:
            f.write(content)

        report = db_manager.sync_incremental(
            docs_dirs=[self.crm_dir, self.tech_dir],
            base_dir=self.base_dir
        )

        self.assertEqual(report["new"], 2)
        res_a = db_manager.document_collection.get(where={"document_id": "doc_CRM_doc_a_txt"})
        res_b = db_manager.document_collection.get(where={"document_id": "doc_Tech_Team_doc_b_txt"})

        self.assertGreater(len(res_a["ids"]), 0)
        self.assertGreater(len(res_b["ids"]), 0)

    def test_06_changed_content_same_filename(self):
        """Test 6: Changes content with same filename, verifying stable document_id."""
        embedder = DummyEmbedder()
        db_manager = VectorDBManager(db_path=self.db_path, embedder=embedder)

        doc_path = os.path.join(self.crm_dir, "SRS.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Version 1 content for SRS.\n" * 5)

        db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        doc_id = "doc_CRM_SRS_txt"
        manifest_doc_1 = db_manager.manifest["documents"][doc_id]
        hash_1 = manifest_doc_1["file_hash"]

        # Change content
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Version 2 updated requirements for SRS.\n" * 15)

        db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        manifest_doc_2 = db_manager.manifest["documents"][doc_id]
        hash_2 = manifest_doc_2["file_hash"]

        self.assertEqual(manifest_doc_1["document_id"], manifest_doc_2["document_id"])
        self.assertNotEqual(hash_1, hash_2)

    def test_07_changed_filename_same_logical_document(self):
        """Test 7: Renaming a file deletes old document_id and creates new document_id."""
        embedder = DummyEmbedder()
        db_manager = VectorDBManager(db_path=self.db_path, embedder=embedder)

        old_path = os.path.join(self.crm_dir, "old_spec.txt")
        new_path = os.path.join(self.crm_dir, "new_spec.txt")
        content = "Specification document content.\n" * 5

        with open(old_path, "w", encoding="utf-8") as f:
            f.write(content)

        db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        # Rename file
        os.rename(old_path, new_path)

        report = db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        self.assertEqual(report["deleted"], 1)
        self.assertEqual(report["new"], 1)
        self.assertEqual(len(db_manager.document_collection.get(where={"document_id": "doc_CRM_old_spec_txt"})["ids"]), 0)
        self.assertGreater(len(db_manager.document_collection.get(where={"document_id": "doc_CRM_new_spec_txt"})["ids"]), 0)

    def test_08_chunking_config_change(self):
        """Test 8: Chunking version/size change forces re-indexing of documents."""
        embedder = DummyEmbedder()
        db_manager = VectorDBManager(db_path=self.db_path, embedder=embedder, chunk_size=800)

        doc_path = os.path.join(self.crm_dir, "arch.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Architecture Overview Document.\n" * 20)

        db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        # Simulate chunking config change
        db_manager_v2 = VectorDBManager(db_path=self.db_path, embedder=embedder, chunk_size=400)
        db_manager_v2.CHUNKING_VERSION = "v2_Recursive_400_100"

        report = db_manager_v2.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        self.assertEqual(report["modified"], 1)
        self.assertGreater(report["chunks_created"], 0)

    def test_09_embedding_model_change(self):
        """Test 9: Embedding model change triggers re-indexing."""
        embedder1 = DummyEmbedder(model_name="BAAI/bge-m3")
        db_manager1 = VectorDBManager(db_path=self.db_path, embedder=embedder1)

        doc_path = os.path.join(self.crm_dir, "test.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Testing model change detection.\n" * 5)

        db_manager1.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        # Change embedder model name
        embedder2 = DummyEmbedder(model_name="BAAI/bge-large-en-v1.5")
        db_manager2 = VectorDBManager(db_path=self.db_path, embedder=embedder2)

        report = db_manager2.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        self.assertEqual(report["modified"], 1)

    def test_10_partial_indexing_failure_recovery(self):
        """Test 10: System handles partial file failures gracefully without corrupting overall manifest."""
        embedder = DummyEmbedder()
        db_manager = VectorDBManager(db_path=self.db_path, embedder=embedder)

        valid_doc = os.path.join(self.crm_dir, "valid.txt")
        with open(valid_doc, "w", encoding="utf-8") as f:
            f.write("Valid text document content.\n" * 5)

        report = db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        self.assertEqual(report["new"], 1)
        self.assertEqual(db_manager.manifest["documents"]["doc_CRM_valid_txt"]["status"], "INDEXED")

    def test_11_crm_document_indexing(self):
        """Test 11: Verifies indexing of CRM documents."""
        embedder = DummyEmbedder()
        db_manager = VectorDBManager(db_path=self.db_path, embedder=embedder)

        brd_path = os.path.join(self.crm_dir, "BRD.txt")
        srs_path = os.path.join(self.crm_dir, "SRS_part1.txt")

        with open(brd_path, "w", encoding="utf-8") as f:
            f.write("Business Requirements Document for CRM Scope.\n" * 10)
        with open(srs_path, "w", encoding="utf-8") as f:
            f.write("Software Requirements Specification Part 1.\n" * 10)

        report = db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        self.assertEqual(report["new"], 2)
        doc_ids = db_manager.manifest["documents"].keys()
        self.assertTrue(any("CRM" in did for did in doc_ids))

    def test_12_tech_team_document_indexing(self):
        """Test 12: Verifies indexing of Tech_Team documents."""
        embedder = DummyEmbedder()
        db_manager = VectorDBManager(db_path=self.db_path, embedder=embedder)

        tech_file = os.path.join(self.tech_dir, "Project_Charter.txt")
        with open(tech_file, "w", encoding="utf-8") as f:
            f.write("Software Development Team Project Charter.\n" * 10)

        report = db_manager.sync_incremental(docs_dirs=[self.tech_dir], base_dir=self.base_dir)

        self.assertEqual(report["new"], 1)
        doc_ids = db_manager.manifest["documents"].keys()
        self.assertTrue(any("Tech_Team" in did for did in doc_ids))

    def test_13_simultaneous_multi_document_modifications(self):
        """Test 13: Handles multiple document modifications, additions, and deletions simultaneously."""
        embedder = DummyEmbedder()
        db_manager = VectorDBManager(db_path=self.db_path, embedder=embedder)

        doc1 = os.path.join(self.crm_dir, "doc1.txt")
        doc2 = os.path.join(self.crm_dir, "doc2.txt")
        doc3 = os.path.join(self.crm_dir, "doc3.txt")

        for d in [doc1, doc2, doc3]:
            with open(d, "w", encoding="utf-8") as f:
                f.write(f"Initial content for {os.path.basename(d)}\n" * 5)

        db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        # 1. Modify doc1
        with open(doc1, "w", encoding="utf-8") as f:
            f.write("Updated modified content for doc1\n" * 15)

        # 2. Delete doc2
        os.remove(doc2)

        # 3. Add doc4 (NEW)
        doc4 = os.path.join(self.crm_dir, "doc4.txt")
        with open(doc4, "w", encoding="utf-8") as f:
            f.write("Brand new document content for doc4\n" * 5)

        # 4. Leave doc3 UNCHANGED

        report = db_manager.sync_incremental(docs_dirs=[self.crm_dir], base_dir=self.base_dir)

        self.assertEqual(report["modified"], 1)
        self.assertEqual(report["deleted"], 1)
        self.assertEqual(report["new"], 1)
        self.assertEqual(report["unchanged"], 1)


if __name__ == "__main__":
    unittest.main()
