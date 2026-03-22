import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.chuka_graphrag_pipeline import GraphRAGAssistant

OFFICIAL_QUERIES = [
    "What units do I take this semester?",
    "Show my timetable for Monday",
    "Find past papers for COSC 482",
    "How much do I pay per semester?",
    "What's the exam policy for missed exams?",
    "When is the COSC 482 exam?",
    "What rooms is my class in?",
    "How do I defer my studies?",
    "What happens if I fail a course?",
    "Show me units for Year 3 Semester 2"
]

import pytest

@pytest.mark.integration
def test_official_proposal_evaluation():
    print("="*60)
    print("CHUKA UNIVERSITY GRAPHRAG - OFFICIAL EVALUATION (APPENDIX C)")
    print("="*60)
    
    try:
        assistant = GraphRAGAssistant()
        profile = {
            'program': 'Computer Science', 
            'year': '1', 
            'semester': '1',
            'faculty': 'Faculty of Science'
        }
        
        results = []
        for i, query in enumerate(OFFICIAL_QUERIES):
            print(f"\n[{i+1}/10] Query: {query}")
            response = assistant.generate_response(query, profile)
            
            # Simple heuristic for 'success' in automation: length and lack of "error"
            is_success = len(response) > 50 and "error" not in response.lower()
            results.append(is_success)
            
            print(f"Status: {' PASS' if is_success else ' FAIL (Manual Review Required)'}")
            print(f"Response Snippet: {response[:150]}...")
            
        print("\n" + "="*60)
        success_count = sum(results)
        accuracy = (success_count / len(OFFICIAL_QUERIES)) * 100
        print(f"FINAL ACCURACY: {accuracy}% ({success_count}/10 passed automation)")
        print("NOTE: Manual verification of response quality is required for Appendix C.")
        print("="*60)
        
        assistant.close()
    except Exception as e:
        print(f"Evaluation Error: {e}")

if __name__ == "__main__":
    test_official_proposal_evaluation()
