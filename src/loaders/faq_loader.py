import os
import pandas as pd
from typing import List
from src.loaders.document import Document

class FAQLoader:
    """
    Loader for the curated Q&A FAQ Excel spreadsheet.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"FAQ file not found at: {self.file_path}")
        
        # Read the Excel file
        df = pd.read_excel(self.file_path)
        
        # Verify essential columns exist
        required = {'Question', 'Answer'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"FAQ Excel file is missing required columns: {missing}")
            
        documents = []
        for idx, row in df.iterrows():
            question = str(row['Question']).strip() if pd.notna(row['Question']) else ""
            answer = str(row['Answer']).strip() if pd.notna(row['Answer']) else ""
            
            if not question and not answer:
                continue  # skip empty rows
                
            # Content formatted exactly as specified: "Question: ...\nAnswer: ..."
            page_content = f"Question: {question}\nAnswer: {answer}"
            
            # Map category from 'Question type' if present, otherwise 'Category' or 'General'
            category = 'General'
            if 'Question type' in df.columns and pd.notna(row['Question type']):
                category = str(row['Question type']).strip()
            elif 'Category' in df.columns and pd.notna(row['Category']):
                category = str(row['Category']).strip()
                
            metadata = {
                "source_type": "faq",
                "category": category,
                "source": os.path.basename(self.file_path),
                "question": question,
                "answer": answer
            }
            
            if 'No.' in df.columns and pd.notna(row['No.']):
                metadata["faq_no"] = int(row['No.'])
                
            documents.append(Document(page_content=page_content, metadata=metadata))
            
        return documents
