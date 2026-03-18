import pdfplumber
import PyPDF2
import io
import logging

log = logging.getLogger("PDFHandler")

def extract_text_from_pdf(file_bytes):
    """Extracts raw text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        log.error(f"pdfplumber failed: {e}. Falling back to PyPDF2.")
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e2:
            log.error(f"PyPDF2 also failed: {e2}")
    
    return text.strip()

def parse_chuka_document(file_name, file_bytes):
    """
    Parses a Chuka University document (Timetable or Fee Statement).
    Returns a formatted context string.
    """
    raw_text = extract_text_from_pdf(file_bytes)
    if not raw_text:
        return "No text could be extracted from the uploaded PDF."

    # Identify document type
    doc_type = "Unknown Document"
    if "timetable" in file_name.lower() or "schedule" in file_name.lower() or "venue" in raw_text.lower():
        doc_type = "Personal Timetable"
    elif "fee" in file_name.lower() or "statement" in file_name.lower() or "balance" in raw_text.lower() or "kes" in raw_text.lower():
        doc_type = "Fee Statement"

    context = f"=== Uploaded Document: {doc_type} ({file_name}) ===\n"
    context += raw_text
    return context
