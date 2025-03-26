"""Tests for utility functions."""
import unittest
from unittest.mock import patch, MagicMock
import sys

from app.utils import get_lm


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    @patch('dspy.LM')
    def test_get_lm_regular_model(self, mock_lm):
        """Test get_lm with a regular model."""
        model_name = "anthropic/claude-3-7-sonnet-20250219"
        get_lm(model_name)
        mock_lm.assert_called_once_with(model_name, **{})
        
    @patch('dspy.LM')
    def test_get_lm_o3_mini_model(self, mock_lm):
        """Test get_lm with o3-mini model."""
        model_name = "o3-mini"
        get_lm(model_name)
        mock_lm.assert_called_once_with(model_name, **{
            "temperature": 1.0,
            "max_tokens": 1000
        })
        
    @patch('dspy.LM')
    def test_get_lm_with_kwargs(self, mock_lm):
        """Test get_lm with custom kwargs."""
        model_name = "o3-mini"
        get_lm(model_name, temperature=0.8, max_tokens=2000)
        mock_lm.assert_called_once_with(model_name, **{
            "temperature": 0.8,
            "max_tokens": 2000
        })


if __name__ == '__main__':
    unittest.main()