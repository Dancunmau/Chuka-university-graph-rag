# Integration Test for Chuka GraphRAG Pipeline
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.chuka_graphrag_pipeline import GraphRAGAssistant
    print(" Successfully imported GraphRAGAssistant from src")
except ImportError as e:
    print(f" Import failed: {e}")
    sys.exit(1)

import pytest

@pytest.mark.integration
def test_initialization():
    try:
        assistant = GraphRAGAssistant()
        print(" Assistant initialized successfully")
        
        # Simple test query
        profile = {'program': 'Computer Science', 'year': '1', 'semester': '1'}
        response = assistant.generate_response("What are my units?", profile)
        
        if response and len(response) > 20:
            print(" Response generation successful")
            print(f"Sample response: {response[:100]}...")
        else:
            print(" Response seems empty or too short")
            
        assistant.close()
        print(" Assistant closed cleanly")
    except Exception as e:
        print(f" Test failed: {e}")

if __name__ == "__main__":
    test_initialization()
