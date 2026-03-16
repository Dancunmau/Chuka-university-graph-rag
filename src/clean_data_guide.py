"""
Data Cleaning Guide for Chuka Exam Papers Metadata
====================================================

This script will help you clean the chuka_exam_papers_metadata.csv file.

Current state:
- Total rows: 26,942
- Unique papers: 20,042
- Issues: Duplicates, possibly empty course codes, inconsistent dates
"""

import pandas as pd
import numpy as np

# Load the data
print("Loading data...")
df = pd.read_csv('d:/Jupyter notebook/chuka_exam_papers_metadata.csv')
print(f"Initial rows: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================
# STEP 1: Remove Duplicates
# ============================================
print("\n" + "="*50)
print("STEP 1: Removing Duplicates")
print("="*50)

print(f"\nBefore deduplication: {len(df)} rows")

# Remove exact duplicates based on repository_link (the unique identifier)
df_clean = df.drop_duplicates(subset=['repository_link'], keep='first')
print(f"After removing duplicate links: {len(df_clean)} rows")
print(f"Removed: {len(df) - len(df_clean)} duplicate rows")

# ============================================
# STEP 2: Handle Missing/Empty Values
# ============================================
print("\n" + "="*50)
print("STEP 2: Checking Missing Values")
print("="*50)

print("\nMissing values per column:")
print(df_clean.isnull().sum())

print("\nEmpty strings per column:")
for col in df_clean.columns:
    empty_count = (df_clean[col] == '').sum()
    if empty_count > 0:
        print(f"{col}: {empty_count} empty strings")

# ============================================
# STEP 3: Clean Course Codes
# ============================================
print("\n" + "="*50)
print("STEP 3: Analyzing Course Codes")
print("="*50)

# Count papers with and without course codes
has_code = df_clean['course_code'].notna() & (df_clean['course_code'] != '')
print(f"\nPapers WITH course codes: {has_code.sum()}")
print(f"Papers WITHOUT course codes: {(~has_code).sum()}")

# Show sample of papers without course codes
print("\nSample of papers without course codes:")
print(df_clean[~has_code][['title', 'year']].head(10))

# DECISION POINT: Do you want to keep papers without course codes?
# Option A: Keep all papers
# Option B: Keep only papers with course codes (exam papers)

print("\n>>> DECISION NEEDED <<<")
print("Do you want to:")
print("A) Keep ALL papers (including non-exam papers)")
print("B) Keep ONLY papers with course codes (exam papers only)")
print("\nUncomment the line below for your choice:")

# df_clean = df_clean[has_code]  # Uncomment for Option B

# ============================================
# STEP 4: Clean Year Column
# ============================================
print("\n" + "="*50)
print("STEP 4: Analyzing Year Data")
print("="*50)

print("\nUnique year values (first 20):")
print(df_clean['year'].value_counts().head(20))

# Check for invalid years
print("\nYear data types and samples:")
print(df_clean['year'].describe())

# ============================================
# STEP 5: Clean Title Column
# ============================================
print("\n" + "="*50)
print("STEP 5: Analyzing Titles")
print("="*50)

# Check for very long or very short titles
df_clean['title_length'] = df_clean['title'].str.len()
print(f"\nTitle length statistics:")
print(df_clean['title_length'].describe())

# Show titles that might be problematic
print("\nVery short titles (< 10 chars):")
print(df_clean[df_clean['title_length'] < 10][['title', 'course_code']].head())

print("\nVery long titles (> 200 chars):")
print(df_clean[df_clean['title_length'] > 200][['title', 'course_code']].head())

# ============================================
# STEP 6: Sort and Organize
# ============================================
print("\n" + "="*50)
print("STEP 6: Sorting Data")
print("="*50)

# Sort by course code, then year
df_clean = df_clean.sort_values(['course_code', 'year'], ascending=[True, False])
print("Data sorted by course_code and year (descending)")

# ============================================
# STEP 7: Save Cleaned Data
# ============================================
print("\n" + "="*50)
print("STEP 7: Saving Cleaned Data")
print("="*50)

# Remove temporary columns
df_clean = df_clean.drop(columns=['title_length'], errors='ignore')

# Save to new file
output_file = 'd:/Jupyter notebook/chuka_exam_papers_CLEANED.csv'
df_clean.to_csv(output_file, index=False)
print(f"\nCleaned data saved to: {output_file}")
print(f"Final row count: {len(df_clean)}")

# ============================================
# STEP 8: Summary Statistics
# ============================================
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)

print(f"\nTotal papers: {len(df_clean)}")
print(f"Unique course codes: {df_clean['course_code'].nunique()}")
print(f"Date range: {df_clean['year'].min()} to {df_clean['year'].max()}")

print("\nTop 10 courses by paper count:")
print(df_clean['course_code'].value_counts().head(10))

print("\nPapers per year:")
print(df_clean['year'].value_counts().sort_index(ascending=False).head(10))

print("\n" + "="*50)
print("CLEANING COMPLETE!")
print("="*50)
print(f"\nReview the cleaned file: {output_file}")
print("You can now use this for your analysis.")
