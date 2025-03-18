"""Tests for the Evaluation module."""

import pytest
from unittest.mock import patch, MagicMock, ANY

from app.services.evaluation import Evaluator

class MockProgram:
    """Mock DSPy program for testing."""
    
    def __init__(self):
        self.predictor = self
    
    def __call__(self, **kwargs):
        # Return a simple prediction for testing
        return {"output": "This is a mock prediction."}

@pytest.fixture
def mock_program():
    """Fixture for a mock DSPy program."""
    return MockProgram()

@pytest.fixture
def evaluator(app_state, sample_manager):
    """Fixture for an Evaluator instance."""
    return Evaluator(app_state, sample_manager)

@patch("metrics.judge_metric", return_value=(0.8, "Good response"))
def test_run_evaluation_basic(mock_judge, evaluator, app_state, sample_manager, basic_signature, sample_data, mock_program):
    """Test basic evaluation functionality."""
    # Add signature and set current
    app_state.add_signature(basic_signature)
    app_state.set_current_signature(basic_signature["name"])
    
    # Mock sample loading
    with patch.object(sample_manager, "load_samples", return_value=sample_data):
        # Mock program loading
        with patch.object(app_state, "get_current_program", return_value=mock_program):
            # Run evaluation
            results = evaluator.run_evaluation("gpt-4")
            
            # Check results
            assert "samples" in results
            assert len(results["samples"]) == len(sample_data)
            assert "total_score" in results
            assert 0 <= results["total_score"] <= 1
            assert "average_score" in results
            assert 0 <= results["average_score"] <= 1
            
            # Check individual sample results
            for sample_result in results["samples"]:
                assert "input" in sample_result
                assert "expected" in sample_result
                assert "predicted" in sample_result
                assert "score" in sample_result
                assert "explanation" in sample_result

@patch("metrics.judge_metric", return_value=(0.8, "Good response"))
def test_run_evaluation_without_current_signature(mock_judge, evaluator, app_state, sample_manager):
    """Test evaluation without a current signature."""
    # No signature set
    app_state.current_signature = None
    
    # Run evaluation should raise exception
    with pytest.raises(Exception):
        evaluator.run_evaluation("gpt-4")

@patch("metrics.judge_metric", return_value=(0.8, "Good response"))
def test_run_evaluation_without_samples(mock_judge, evaluator, app_state, sample_manager, basic_signature, mock_program):
    """Test evaluation without samples."""
    # Add signature and set current
    app_state.add_signature(basic_signature)
    app_state.set_current_signature(basic_signature["name"])
    
    # Mock empty sample loading
    with patch.object(sample_manager, "load_samples", return_value=[]):
        # Mock program loading
        with patch.object(app_state, "get_current_program", return_value=mock_program):
            # Run evaluation should raise exception
            with pytest.raises(Exception):
                evaluator.run_evaluation("gpt-4")

@patch("metrics.judge_metric")
def test_run_evaluation_with_model_error(mock_judge, evaluator, app_state, sample_manager, basic_signature, sample_data):
    """Test evaluation when model errors occur."""
    # Add signature and set current
    app_state.add_signature(basic_signature)
    app_state.set_current_signature(basic_signature["name"])
    
    # Create mock program that raises exception
    error_program = MagicMock()
    error_program.predictor = MagicMock(side_effect=Exception("Model error"))
    
    # Mock sample loading
    with patch.object(sample_manager, "load_samples", return_value=sample_data):
        # Mock program loading
        with patch.object(app_state, "get_current_program", return_value=error_program):
            # Run evaluation
            results = evaluator.run_evaluation("gpt-4")
            
            # Check results
            assert "samples" in results
            assert len(results["samples"]) == len(sample_data)
            assert "total_score" in results
            assert results["total_score"] == 0  # Should be 0 due to errors
            assert "average_score" in results
            assert results["average_score"] == 0  # Should be 0 due to errors
            
            # Check that judge_metric was not called (since prediction failed)
            mock_judge.assert_not_called()