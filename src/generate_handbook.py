
import pandas as pd
import os

def generate_handbook(csv_path: str, output_md_path: str):
    """
    Reads curricular_mapping.csv and generates a structured Markdown handbook.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # Sort the data logically to ensure structured output
    df = df.sort_values(by=['faculty', 'department', 'programme', 'level', 'semester', 'course_code'])

    # Fill NaNs with a placeholder to avoid grouping errors
    df = df.fillna("Unknown")

    print("Generating handbook")
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("# Chuka University - Course Handbook\n\n")
        f.write("*Auto-generated from curricular mapping data.*\n\n---\n\n")

        # Group by Faculty > Dept > Prog > Level > Semester
        grouped = df.groupby(['faculty', 'department', 'programme', 'level', 'semester'])

        current_faculty = None
        current_dept = None
        current_prog = None
        current_level = None
        current_sem = None

        for name, group in grouped:
            faculty, dept, prog, level, sem = name

            if faculty != current_faculty:
                f.write(f"\n## {faculty}\n\n")
                current_faculty = faculty
                current_dept = None
                current_prog = None

            if dept != current_dept:
                f.write(f"###  {dept}\n\n")
                current_dept = dept
                current_prog = None
            
            if prog != current_prog:
                f.write(f"####  {prog}\n\n")
                current_prog = prog
                current_level = None

            if level != current_level or sem != current_sem:
                 f.write(f"##### {level} — Semester {sem}\n\n")
                 current_level = level
                 current_sem = sem

            for _, row in group.iterrows():
                current_badge = " *(Currently Offered)*" if str(row.get('is_jan_april_2026_offering')) == '1' else ""
                f.write(f"- **{row['course_code']}**: {row['course_name']}{current_badge}\n")
            
            f.write("\n") # Blank line after units in a semester

    print(f"Done! Handbook saved to {output_md_path}")

if __name__ == "__main__":
    base_dir = "d:/Jupyter notebook/Graph rag"
    input_csv = os.path.join(base_dir, "curricular_mapping.csv")
    output_md = os.path.join(base_dir, "Course_Handbook.md")
    generate_handbook(input_csv, output_md)
