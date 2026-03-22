import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Mock environment variables BEFORE importing pipeline
os.environ["GEMINI_API_KEY"] = "mock_key"
os.environ["NEO4J_URI"] = "bolt://mock:7687"
os.environ["NEO4J_USERNAME"] = "mock"
os.environ["NEO4J_PASSWORD"] = "mock"

from src.chuka_graphrag_pipeline import extract_entities, synthesise_response

import pytest

@pytest.mark.unit
class TestUnitPipeline(unittest.TestCase):
    """
    Isolated Mock Unit Tests
    These tests deliberately bypass live Neo4j, FAISS, and Gemini endpoints using unittest.mock.
    This guarantees execution in milliseconds during CI/CD without timeouts.
    """

    @patch('src.chuka_graphrag_pipeline.genai.GenerativeModel')
    def test_extract_entities_llm_success(self, MockModel):
        # Setup mock Gemini response
        mock_instance = MockModel.return_value
        mock_response = MagicMock()
        mock_response.text = '{"course_code": "COSC 121", "programme": "Computer Science", "year": "1", "semester": "1", "day": "Monday", "topic": null}'
        mock_instance.generate_content.return_value = mock_response

        query = "What time is COSC 121 on Monday?"
        profile = {"program": "Computer Science"}
        
        entities = extract_entities(query, profile)
        
        self.assertEqual(entities["course_code"], "COSC 121")
        self.assertEqual(entities["day"], "Monday")
        self.assertEqual(entities["programme"], "Computer Science")

    @patch('src.chuka_graphrag_pipeline.genai.GenerativeModel')
    def test_extract_entities_regex_fallback(self, MockModel):
        # Force LLM JSON error to trigger regex fallback
        mock_instance = MockModel.return_value
        mock_response = MagicMock()
        mock_response.text = 'I am an AI, I cannot help with that.' # Invalid JSON
        mock_instance.generate_content.return_value = mock_response

        # Regex fallback should catch COSC 121 and 'Monday'
        query = "Show my Monday timetable for COSC 121"
        profile = {"program": "Nursing", "year": "2", "semester": "2"}
        
        entities = extract_entities(query, profile)
        
        self.assertEqual(entities["course_code"], "COSC 121")
        self.assertEqual(entities["day"], "Monday")
        # Should fallback to profile 
        self.assertEqual(entities["programme"], "Nursing")
        self.assertEqual(entities["year"], "2")

    @patch('src.chuka_graphrag_pipeline.genai.GenerativeModel')
    def test_synthesise_response_with_mocks(self, MockModel):
        # Setup mock Gemini final answer
        mock_instance = MockModel.return_value
        mock_response = MagicMock()
        mock_response.text = 'Based on the context, COSC 121 is on Monday.'
        mock_instance.generate_content.return_value = mock_response

        graph_ctx = "COSC 121: Intro to Computers | Timetable: Monday 7AM-10AM"
        faiss_ctx = ""
        profile = {"program": "Computer Science"}

        final_answer = synthesise_response("When is my computer class?", graph_ctx, faiss_ctx, profile)
        
        self.assertIn("COSC 121 is on Monday", final_answer)
        mock_instance.generate_content.assert_called_once()

if __name__ == '__main__':
    unittest.main()
