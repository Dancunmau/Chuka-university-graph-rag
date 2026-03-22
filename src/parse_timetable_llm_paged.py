import os
import time
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Configuration
load_dotenv('d:/Jupyter notebook/Graph rag/.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
genai.configure(api_key=GEMINI_API_KEY)

PDF_PATH = r'd:/Jupyter notebook/Graph rag/reports/1st-Draft-Teaching-TT-Jan-April.-2026._compressed-1.pdf'
OUTPUT_CSV = r'd:/Jupyter notebook/Graph rag/data/timetable.csv'

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    print("Waiting for file processing...")
    for name in (f.name for f in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")

def extract_timetable_data():
    # 2. Upload the file
    pdf_file = upload_to_gemini(PDF_PATH, mime_type="application/pdf")
    wait_for_files_active([pdf_file])

    # 3. Prompt Construction
    prompt = """
    Extract all academic timetable entries from this Chuka University Timetable PDF.
    
    The PDF contains a grid/table layout with days (Monday to Sunday) and time slots.
    Each cell contains a unit code (e.g., COSC 121), a room (e.g., S104), and potentially a group/programme label.
    
    Task:
    Return a JSON list of objects, one for each "slot" where a unit is assigned.
    Each object MUST have these keys:
    - "day": The day of the week (e.g., "Monday")
    - "time": The time slot (e.g., "7:00-10:00 AM")
    - "room": The room code (e.g., "S104")
    - "unit_code": The unit code (e.g., "COSC 121")
    - "semester": The semester (e.g., 1) - If not explicitly stated per row, assume 1 unless the PDF says otherwise.
    
    Return ONLY a valid JSON list.
    """

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={"response_mime_type": "application/json"}
    )

    print("Requesting extraction from Gemini (full PDF)")
    try:
        response = model.generate_content([pdf_file, prompt])
        results = json.loads(response.text)
        
        # 4. Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Extraction successful Saved {len(df)} records to {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
    finally:
        # Cleanup
        genai.delete_file(pdf_file.name)
        print(f"Deleted remote file {pdf_file.name}")

if __name__ == "__main__":
    extract_timetable_data()
