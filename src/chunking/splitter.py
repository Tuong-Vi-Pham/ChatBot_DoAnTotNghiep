from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.loaders.document import Document
from src.preprocessing.cleaning import clean_text

class DocumentSplitter:
    """
    Splitter that handles document chunking.
    Curated FAQ Documents are preserved intact, whereas Tech_Team documents
    are split into smaller chunks with configurable overlap.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

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
            
            if source_type == "faq":
                faq_count += 1
                # Preserve FAQ intact
                faq_doc = Document(page_content=cleaned, metadata=doc.metadata.copy())
                processed_chunks.append(faq_doc)
                chunk_lengths.append(len(cleaned))
            else:
                doc_count += 1
                # Split Tech_Team documents
                split_texts = self.splitter.split_text(cleaned)
                
                for i, text in enumerate(split_texts):
                    text = text.strip()
                    if not text:
                        continue
                        
                    # Copy and enrich metadata
                    metadata = doc.metadata.copy()
                    metadata["chunk_index"] = i
                    metadata["total_chunks"] = len(split_texts)
                    
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
