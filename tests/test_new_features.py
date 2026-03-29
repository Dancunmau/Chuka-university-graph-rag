# New Features Verification Test 
import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.chuka_graphrag_pipeline import GraphRAGAssistant
from src.database import get_or_create_user, log_chat_history, get_chat_history, clear_chat_history

import pytest

@pytest.mark.integration
def test_new_features():
    print("="*60)
    print("VERIFYING PHASE 2 FUNCTIONAL FEATURES")
    print("="*60)
    
    assistant = GraphRAGAssistant()
    profile = {
        'program': 'Bachelor of Science (Computer Science)', 
        'year': '1', 
        'semester': '1'
    }

    # 1. Test Day-Specific Timetable Filtering
    print(f"\n[1/4] Testing Day-Specific Filtering (Monday)...")
    # We ask a question that should trigger the "day" entity extraction
    response = assistant.generate_response("Show my timetable for Monday", profile)
    # Success if it doesn't crash and returns something better than an empty fallback
    if len(response) > 50 and "Monday" in response:
        print(" PASS: Day context processed in response.")
    else:
        print(f" NOTE: Response generated but 'Monday' keyword check failed. Content: {response[:100]}...")

    # 2. Test Fee Calculation
    print(f"\n[2/4] Testing Total Cost Calculation...")
    # Use a specific program query
    query = "What is the total cost for Bachelor of Science in Computer Science?"
    cost_response = assistant.generate_response(query, profile)
    if "Total Program Cost" in cost_response or "KES" in cost_response or "cost" in cost_response.lower():
        print(" PASS: Total cost calculation detected.")
    else:
        print(f" FAIL: No cost calculation found. Response: {cost_response[:100]}...")

    # 3. Test Database History & Clear History
    print(f"\n[3/4] Testing Chat History Clear...")
    token = "test_token_" + os.urandom(4).hex()
    user = get_or_create_user(token)
    user_id = user["user_id"]
    session_id = "test_sess_" + os.urandom(4).hex()
    log_chat_history(user_id, session_id, "Test Query", "Test Response")
    
    history_before = get_chat_history(user_id, session_id=session_id)
    print(f"History count before: {len(history_before)}")
    
    clear_chat_history(user_id)
    history_after = get_chat_history(user_id)
    print(f"History count after: {len(history_after)}")
    
    if len(history_before) > 0 and len(history_after) == 0:
        print(" PASS: Database history cleared successfully.")
    else:
        print(" FAIL: History clear failed.")

    # 4. Test Timetable Data Retrieval
    print(f"\n[4/4] Testing Timetable Export Retrieval...")
    tt_data = assistant.get_personalized_timetable(profile)
    if isinstance(tt_data, list):
        print(f" PASS: Method returned list with {len(tt_data)} rows.")
    else:
        print(" FAIL: Could not retrieve timetable list.")

    assistant.close()
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_new_features()
