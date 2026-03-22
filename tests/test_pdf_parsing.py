import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from reportlab.pdfgen import canvas
from src.pdf_handler import extract_text_from_pdf, parse_chuka_document
import io
import os

import pytest

@pytest.mark.unit
def test_pdf_parsing():
    # 1. Create a dummy PDF in memory
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, "Chuka University Personal Timetable")
    c.drawString(100, 730, "Monday: COSC 121 - Computer Science - Room: B1")
    c.drawString(100, 710, "Tuesday: COSC 222 - Data Structures - Room: LH1")
    c.save()
    
    pdf_bytes = buf.getvalue()
    
    # 2. Test extraction
    text = extract_text_from_pdf(pdf_bytes)
    print("Extracted Text:")
    print(text)
    
    assert "COSC 121" in text
    assert "COSC 222" in text
    
    # 3. Test Chuka parsing logic
    context = parse_chuka_document("my_timetable.pdf", pdf_bytes)
    print("\nParsed Context:")
    print(context)
    
    assert "Personal Timetable" in context
    assert "my_timetable.pdf" in context
    
    print("\nPDF Parsing Verification: SUCCESS")

if __name__ == "__main__":
    test_pdf_parsing()
