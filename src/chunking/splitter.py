import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.loaders.document import Document
from src.preprocessing.cleaning import clean_text

class DocumentSplitter:
    """
    Splitter that handles document chunking.
    Curated FAQ Documents are preserved intact, whereas Tech_Team documents
    are split into heading-aware semantic chunks with configurable overlap and enriched metadata.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
        )

    def _extract_section_heading(self, text: str) -> str:
        """Helper method to detect section/heading line if present in text chunk."""
        match = re.search(r'^(?:#+|\d+\.|\bSection\b)\s*(.+)$', text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[0][:60] if lines else "General"

    def split_documents(self, documents: List[Document], print_stats: bool = True) -> List[Document]:
        processed_chunks = []
        
        doc_count = 0
        faq_count = 0
        chunk_lengths = []
        
        for doc in documents:
            # Clean/normalize text first
            cleaned = clean_text(doc.page_content)
            if not cleaned:
                continue
                
            # Check if this is an FAQ document
            source_type = doc.metadata.get("source_type", "")
            source_filename = doc.metadata.get("source", doc.metadata.get("file_name", "unknown"))
            rel_path = doc.metadata.get("relative_path", source_filename)
            
            if source_type == "faq":
                faq_count += 1
                metadata = doc.metadata.copy()
                metadata.setdefault("filename", source_filename)
                metadata.setdefault("file_name", source_filename)
                metadata.setdefault("document_type", "excel")
                metadata.setdefault("title", "FAQ Spreadsheet")
                metadata.setdefault("section", metadata.get("category", "General"))
                
                doc_id = metadata.get("document_id")
                if not doc_id:
                    clean_slug = re.sub(r'[^a-zA-Z0-9_]', '_', rel_path)
                    clean_slug = re.sub(r'_+', '_', clean_slug).strip('_')
                    doc_id = f"faq_{clean_slug}"
                metadata["document_id"] = doc_id
                metadata.setdefault("chunk_id", f"{doc_id}_chunk_{doc.metadata.get('faq_no', len(processed_chunks))}")
                metadata["chunk_index"] = 0
                metadata["source_path"] = metadata.get("full_path", rel_path)
                
                faq_doc = Document(page_content=cleaned, metadata=metadata)
                processed_chunks.append(faq_doc)
                chunk_lengths.append(len(cleaned))
            else:
                doc_count += 1
                # Split documents (CRM or Tech_Team)
                split_texts = self.splitter.split_text(cleaned)
                doc_title = doc.metadata.get("title", source_filename.replace("_", " ").split(".")[0])
                
                doc_id = doc.metadata.get("document_id")
                if not doc_id:
                    clean_slug = re.sub(r'[^a-zA-Z0-9_]', '_', rel_path)
                    clean_slug = re.sub(r'_+', '_', clean_slug).strip('_')
                    doc_id = f"doc_{clean_slug}"
                
                for i, text in enumerate(split_texts):
                    text = text.strip()
                    if not text:
                        continue
                        
                    # Copy and enrich metadata for citation and precision
                    metadata = doc.metadata.copy()
                    metadata["document_id"] = doc_id
                    metadata["filename"] = source_filename
                    metadata["file_name"] = source_filename
                    metadata["source_path"] = metadata.get("full_path", rel_path)
                    metadata["document_type"] = metadata.get("document_type", source_filename.split(".")[-1] if "." in source_filename else "text")
                    metadata["title"] = doc_title
                    metadata["section"] = self._extract_section_heading(text)
                    metadata["chunk_index"] = i
                    metadata["total_chunks"] = len(split_texts)
                    metadata["chunk_id"] = f"{doc_id}_chunk_{i}"
                    
                    chunk_doc = Document(page_content=text, metadata=metadata)
                    processed_chunks.append(chunk_doc)
                    chunk_lengths.append(len(text))
                    
        # Calculate and print statistics if requested
        if print_stats and chunk_lengths:
            total_chunks = len(processed_chunks)
            avg_len = sum(chunk_lengths) / total_chunks
            min_len = min(chunk_lengths)
            max_len = max(chunk_lengths)
            
            print("=========================================")
            print("CHUNKING STATISTICS")
            print("=========================================")
            print(f"Source FAQs (Intact):        {faq_count}")
            print(f"Source Documents (Split):    {doc_count}")
            print(f"Total Chunks Generated:      {total_chunks}")
            print(f"Average Chunk Length (char): {avg_len:.2f}")
            print(f"Minimum Chunk Length (char): {min_len}")
            print(f"Maximum Chunk Length (char): {max_len}")
            print("=========================================")
            
        return processed_chunks
