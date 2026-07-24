class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        truncated_content = self.page_content[:50].replace('\n', ' ')
        return f"Document(page_content='{truncated_content}...', metadata={self.metadata})"
