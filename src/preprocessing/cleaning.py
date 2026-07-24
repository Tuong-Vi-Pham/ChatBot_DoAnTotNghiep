import re

def clean_text(text: str) -> str:
    """
    Normalize whitespaces, line breaks, and trim edges.
    """
    if not text:
        return ""
        
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace three or more newlines with double newlines to maintain paragraph separation
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip whitespace from start and end
    return text.strip()
