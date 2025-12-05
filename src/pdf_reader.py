from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        return full_text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
