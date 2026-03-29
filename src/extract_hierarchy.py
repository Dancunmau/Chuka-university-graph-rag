"""Extracts the academic hierarchy from raw PDF text using Gemini. Outputs a structured CSV mapping.
"""

import json
import os
import re
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY missing")

genai.configure(api_key=API_KEY)

# Enforcing application/json MIME type at the API level guarantees the LLM 
# returns valid JSON, bypassing the need for complex string parsing/regex.
model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})

def extract_hierarchy():
    with open('extracted_pdf_text.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    advert_text = data.get('CU-Advert-Jan-2024.pdf', '')
    handbook_text = data.get('Student-handbook-August-2024.docx.pdf', '')

    prompt = f"""
    You are an expert data extractor. I need to map Chuka University's academic hierarchy.
    Below are texts from the University Advert and Student Handbook.

    Task:
    Extract every single academic programme mentioned, and map it to its corresponding Faculty/School, and if available in the text, its Department.
    If the Department is not explicitly stated for a programme, output null for Department.
    The Levels are: Certificate, Diploma, Bachelor, Master, PhD.

    Output format MUST be a list of JSON objects exactly like this:
    [
      {{
        "Faculty": "Faculty of Science & Technology",
        "Department": "Department of Computer Sciences",
        "Programme": "Bachelor of Science in Computer Science",
        "Level": "Bachelor"
      }}
    ]

    Important rules:
    - Include ALL faculties (Business, Science, Education, Agriculture, Environment, Humanities, Nursing/Public Health, Law, Engineering).
    - Get all bachelor degrees, master's, phds, diplomas, certificates.
    - Be precise with programme names.
    - Ensure EVERY programme has a Faculty.
    
    Advert Text (use for full list of programmes and their faculties):
    {advert_text}
    
    Handbook Text (use to find which Departments belong to which Faculties, and map programmes to them):
    {handbook_text}
    """
    
    print("Calling Gemini to extract hierarchy...")
    response = model.generate_content(prompt)
    
    try:
        results = json.loads(response.text)
        df = pd.DataFrame(results)
        df.to_csv('hierarchy_mapping.csv', index=False)
        print(f"Successfully extracted {len(df)} programmes to hierarchy_mapping.csv")
        print(df.head())
    except Exception as e:
        print("Error parsing response:", e)
        print("Raw response:", response.text)

if __name__ == "__main__":
    extract_hierarchy()
