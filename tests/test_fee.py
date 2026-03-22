import sys
import os
# Add root directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.chuka_graphrag_pipeline import GraphRAGAssistant

import pytest

@pytest.mark.integration
def test_fee_calculation():
    assistant = GraphRAGAssistant()
    profile = {
        "faculty": "Science and Technology",
        "program": "BSc Computer Science",
        "year": "1",
        "semester": "1"
    }
    
    query = "How much are the tuition fees for a Bachelor of Science in Computer Science? Please calculate the total cost for the entire programme."
    print("TESTING FEE QUERY:")
    print(f"Q: {query}")
    ans = assistant.generate_response(query, profile)
    print("\n" + "="*50)
    print(ans)
    print("="*50)
    assert len(ans) > 50
    assistant.close()

if __name__ == "__main__":
    test_fee_calculation()
