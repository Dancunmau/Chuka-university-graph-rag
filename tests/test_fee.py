import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chuka_graphrag_pipeline import GraphRAGAssistant

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
assistant.close()
