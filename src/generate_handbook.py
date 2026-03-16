
import pandas as pd
import os

def generate_handbook(csv_path: str, output_md_path: str):
    """
    Reads curricular_mapping.csv and generates a structured Markdown handbook.
    Groups by: Faculty -> Department -> Programme -> Level -> Semester -> Units
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

    print("Generating handbook...")
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("# Chuka University - Course Handbook\n\n")
        f.write("*Auto-generated from curricular mapping data.*\n\n---\n\n")

        # Group by Faculty -> Dept -> Prog -> Level -> Semester
        grouped = df.groupby(['faculty', 'department', 'programme', 'level', 'semester'])

        current_faculty = None
        current_dept = None
        current_prog = None
        current_level = None
        current_sem = None

        for name, group in grouped:
            faculty, dept, prog, level, sem = name

            # 1. Faculty Header (Level 2)
            if faculty != current_faculty:
                f.write(f"\n## 🏛️ {faculty}\n\n")
                current_faculty = faculty
                current_dept = None # Reset child tracking
                current_prog = None

            # 2. Department Header (Level 3)
            if dept != current_dept:
                f.write(f"### 🏢 {dept}\n\n")
                current_dept = dept
                current_prog = None # Reset child tracking
            
            # 3. Programme Header (Level 4)
            if prog != current_prog:
                f.write(f"#### 🎓 {prog}\n\n")
                current_prog = prog
                current_level = None # Reset child tracking

            # 4. Level & Semester Header (Level 5)
            # Combine level and semester to avoid too many tiny headings if possible,
            # or keep them separate. Let's combine for now: "Year 1 - Semester 1"
            if level != current_level or sem != current_sem:
                 f.write(f"##### {level} — Semester {sem}\n\n")
                 current_level = level
                 current_sem = sem

            # 5. List the units (bullets)
            for _, row in group.iterrows():
                # Indicate if it's currently offered
                current_badge = " *(Currently Offered)*" if str(row.get('is_jan_april_2026_offering')) == '1' else ""
                f.write(f"- **{row['course_code']}**: {row['course_name']}{current_badge}\n")
            
            f.write("\n") # Blank line after units in a semester

    print(f"Done! Handbook saved to {output_md_path}")

if __name__ == "__main__":
    base_dir = "d:/Jupyter notebook/Graph rag"
    input_csv = os.path.join(base_dir, "curricular_mapping.csv")
    output_md = os.path.join(base_dir, "Course_Handbook.md")
    generate_handbook(input_csv, output_md)
