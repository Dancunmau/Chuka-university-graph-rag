import pandas as pd
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})

def map_programmes():
    # Load the 35 old programmes
    old_df = pd.read_csv(r'd:/Jupyter notebook/Graph rag/data/curricular_mapping.csv')
    old_programmes = sorted(old_df['programme'].dropna().unique().tolist())
    
    print(f"Found {len(old_programmes)} unique old programmes.")
    
    # Load the 252 new official programmes
    new_df = pd.read_csv(r'd:/Jupyter notebook/Graph rag/data/hierarchy_mapping.csv')
    new_programmes = sorted(new_df['Programme'].dropna().unique().tolist())
    
    prompt = f"""
    You are an AI tasked with matching a list of 35 "old" academic programme names to a list of official "new" programme names.
    
    Here is the list of 35 old programme names:
    {json.dumps(old_programmes)}
    
    Here is the allowed list of official new programme names:
    {json.dumps(new_programmes)}
    
    Your task:
    For EVERY SINGLE ONE of the 35 old programme names, find its exact match in the official new list.
    Output MUST be a JSON list of objects, exactly like this:
    [
      {{
         "old_name": "Bachelor of Science in Computer Science",
         "new_name": "Bachelor of Science (Computer Science)"
      }}
    ]
    
    Rules:
    - You must output exactly {len(old_programmes)} objects.
    - "new_name" MUST be an exact string from the allowed new list. Do not make up names.
    - If you are absolutely unsure, pick the closest semantic match.
    """
    
    print("Asking Gemini to map the old programme names to the new official ones...")
    response = model.generate_content(prompt)
    
    try:
        mapping = json.loads(response.text)
        print(f"Received mapping for {len(mapping)} programmes.")
        
        # Build mapping dictionary
        name_map = {item['old_name']: item['new_name'] for item in mapping}
        
        # Merge the old data with the new hierarchy
        old_df['Official_Programme'] = old_df['programme'].map(name_map)
        
        # Merge to get faculty and department
        merged_df = pd.merge(old_df, new_df, left_on='Official_Programme', right_on='Programme', how='left')
        
        # Overwrite the old corrupted faculty/department columns with the pristine ones
        # Original columns: 'course_code', 'course_name', 'department', 'faculty', 'level', 'is_jan_april_2026_offering', 'programme', 'semester'
        # Drop the old ones
        merged_df = merged_df.drop(columns=['faculty', 'department', 'programme'])
        
        # Rename new ones to match the standard
        merged_df = merged_df.rename(columns={
            'Faculty': 'faculty',
            'Department': 'department',
            'Programme': 'programme'
        })
        
        # Reorder to standard mapping
        cols = ['course_code', 'course_name', 'faculty', 'department', 'Level', 'programme', 'level', 'semester', 'is_jan_april_2026_offering']
        available_cols = [c for c in cols if c in merged_df.columns]
        merged_df = merged_df[available_cols]
        
        # Rename lower case 'level' back to standard
        merged_df = merged_df.rename(columns={'Level': 'Academic_Level', 'level': 'year'})
        
        # Export as the new pristine map
        merged_df.to_csv(r'd:/Jupyter notebook/Graph rag/data/master_curricular_mapping.csv', index=False)
        print("Successfully generated master_curricular_mapping.csv")
        
    except Exception as e:
        print("Error during mapping:", e)
        print("Raw response:", response.text)

if __name__ == "__main__":
    map_programmes()
