import pdfplumber
import csv
import re

pdf_path = r'D:\OS DESIGN\CU-Advert-October-2024-F.pdf'
output_csv = r'd:\Jupyter notebook\Graph rag\programs.csv'

def clean_text(text):
    if not text: return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^\d+[\.\)]\s*', '', text)
    return text

def extract_programs(pdf_path, output_path):
    programs = []
    current_level = "Bachelors"
    current_faculty = ""
    
    # Levels and keywords
    levels = {
        "POSTGRADUATE DIPLOMA": "Postgraduate Diploma",
        "POST-GRADUATE DIPLOMA": "Postgraduate Diploma",
        "PGDE": "Postgraduate Diploma",
        "PH.D.": "Doctorate",
        "DOCTOR": "Doctorate",
        "MASTER OF": "Masters",
        "M.SC.": "Masters",
        "BACHELOR": "Bachelors",
        "B.SC.": "Bachelors",
        "B.A.": "Bachelors",
        "DIPLOMA IN": "Diploma",
        "CERTIFICATE IN": "Certificate"
    }

    last_dur = ""
    last_fee = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables: continue
            
            for table in tables:
                p_name = ""
                for row in table:
                    if not any(row): continue
                    row_text = " ".join([str(c) for c in row if c]).upper()
                    
                    # Faculty Header
                    if "FACULTY OF" in row_text or "SCHOOL OF" in row_text:
                        current_faculty = clean_text(row[0]) if row[0] else current_faculty
                        continue
                    
                    # Level Header detection
                    if "DIPLOMA PROGRAMMES" in row_text: current_level = "Diploma"
                    elif "CERTIFICATE PROGRAMMES" in row_text: current_level = "Certificate"
                    elif "BACHELORS DEGREE" in row_text: current_level = "Bachelors"
                    elif "MASTERS DEGREE" in row_text: current_level = "Masters"
                    elif "PHD DEGREE" in row_text: current_level = "Doctorate"
                    elif "POSTGRADUATE" in row_text: current_level = "Postgraduate Diploma"

                    # Check for lists in PhD/Masters (Special layout)
                    is_list = any(re.search(r'\d+[\.\)]\s', str(c)) for c in row if c)
                    if is_list and current_level in ["Masters", "Doctorate"]:
                        pot = []
                        r_dur = ""
                        r_fee = ""
                        for c in row:
                            if not c: continue
                            val = str(c).strip()
                            if re.search(r'\d+[\.\)]\s', val):
                                sub = re.split(r'\d+[\.\)]\s', val)
                                pot.extend([clean_text(n) for n in sub if n.strip()])
                            elif "/=" in val: r_fee = clean_text(val)
                            elif "YEAR" in val.upper(): r_dur = clean_text(val)
                        
                        for n in pot:
                            if len(n) < 10: continue
                            programs.append({"name": n, "level": current_level, "duration": r_dur, "fee": r_fee, "faculty": current_faculty})
                        continue

                    # Standard Row
                    n_col = str(row[0]) if row[0] else ""
                    if n_col:
                        # Detect level change within name
                        for k, v in levels.items():
                            if k in n_col.upper():
                                current_level = v
                                break
                        
                        # Gather metadata
                        dur = ""
                        fee = ""
                        for c in row[1:]:
                            if not c: continue
                            cv = str(c).upper()
                            if any(k in cv for k in ["YEAR", "SEM", "BLOCK"]): dur = clean_text(c)
                            if "/=" in cv: fee = clean_text(c)
                        
                        if dur: last_dur = dur
                        if fee: last_fee = fee
                        
                        if len(n_col) > 10 and not "PROGRAMME" in n_col.upper():
                            programs.append({"name": clean_text(n_col), "level": current_level, "duration": dur if dur else last_dur, "fee": fee if fee else last_fee, "faculty": current_faculty})

    # Deduplicate and Filter noise
    unique = []
    seen = set()
    for p in programs:
        p['name'] = re.sub(r'\(.*?\)', '', p['name']).strip() # Normalize
        p['name'] = re.sub(r'\s+', ' ', p['name']).strip()
        nu = p['name'].upper()
        if any(k in nu for k in ["BASED;", "/=", "SEM 1", "TUITION"]): continue
        if len(p['name']) < 12 and current_level not in ["Diploma", "Certificate"]: continue
        
        key = (p['name'].lower(), p['level'])
        if key not in seen:
            unique.append(p)
            seen.add(key)

    from collections import Counter
    counts = Counter(p['level'] for p in unique)
    print(f"Final Count: {len(unique)}")
    print("Counts by level:", dict(counts))

    keys = ["name", "level", "duration", "fee", "faculty"]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(unique)

if __name__ == "__main__":
    extract_programs(pdf_path, output_csv)
