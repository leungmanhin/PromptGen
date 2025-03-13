"""Tests for the Optimization module."""

import pytest
from unittest.mock import patch, MagicMock, ANY

from app.services.optimization import Optimizer
import dspy

class MockDSPyCompiler:
    """Mock DSPy compiler for testing."""
    
    def __init__(self, **kwargs):
        pass
    
    def compile(self, program, **kwargs):
        # Just return the program without changes
        return program

class MockSignatureClass:
    """Mock DSPy Signature class for testing."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

@pytest.fixture
def optimizer(app_state, sample_manager):
    """Fixture for an Optimizer instance."""
    return Optimizer(app_state, sample_manager)

@pytest.fixture
def dspy_mocks():
    """Fixture for DSPy mocks."""
    with patch("dspy.teleprompt.MIPRO", return_value=MockDSPyCompiler()) as mock_mipro, \
         patch("dspy.Signature", side_effect=MockSignatureClass) as mock_signature, \
         patch("dspy.Example") as mock_example, \
         patch("dspy.Module") as mock_module:
        yield {
            "mipro": mock_mipro,
            "signature": mock_signature,
            "example": mock_example,
            "module": mock_module
        }

def test_optimizer_init(optimizer, app_state, sample_manager):
    """Test Optimizer initialization."""
    assert optimizer.app_state is app_state
    assert optimizer.sample_manager is sample_manager

@patch("optimization.Optimizer._prepare_training_data", return_value=[])
@patch("dspy.Module")
def test_run_optimization_without_signature(mock_module, mock_prepare, optimizer, app_state):
    """Test running optimization without a signature."""
    # No signature set
    app_state.current_signature = None
    
    # Run optimization should raise exception
    with pytest.raises(Exception):
        optimizer.run_optimization("gpt-4")

@patch("optimization.Optimizer._prepare_training_data", return_value=[])
def test_run_optimization_without_samples(mock_prepare, optimizer, app_state, sample_manager, basic_signature, dspy_mocks):
    """Test running optimization without samples."""
    # Add signature and set current
    app_state.add_signature(basic_signature)
    app_state.set_current_signature(basic_signature["name"])
    
    # Mock empty sample loading
    with patch.object(sample_manager, "load_samples", return_value=[]):
        # Run optimization should raise exception
        with pytest.raises(Exception):
            optimizer.run_optimization("gpt-4")

@patch("optimization.Optimizer._optimization_metric", return_value=0.8)
def test_run_optimization_basic(mock_metric, optimizer, app_state, sample_manager, basic_signature, sample_data, dspy_mocks):
    """Test basic optimization functionality."""
    # Add signature and set current
    app_state.add_signature(basic_signature)
    app_state.set_current_signature(basic_signature["name"])
    
    # Create mock DSPy program
    mock_program = MagicMock()
    
    # Mock sample loading
    with patch.object(sample_manager, "load_samples", return_value=sample_data):
        # Mock program creation/loading
        with patch.object(app_state, "get_or_create_program", return_value=mock_program):
            # Mock program saving
            with patch.object(app_state, "save_program") as mock_save:
                # Run optimization
                optimizer.run_optimization("gpt-4")
                
                # Check if DSPy components were called correctly
                dspy_mocks["mipro"].assert_called_once()
                dspy_mocks["signature"].assert_called_once()
                
                # Check if program was saved
                mock_save.assert_called_once()

def test_prepare_training_data(optimizer, app_state, sample_manager, basic_signature, sample_data, dspy_mocks):
    """Test preparation of training data."""
    # Add signature and set current
    app_state.add_signature(basic_signature)
    app_state.set_current_signature(basic_signature["name"])
    
    # Prepare training data
    with patch.object(dspy_mocks["example"], "from_dict", return_value="example_obj"):
        examples = optimizer._prepare_training_data(sample_data, basic_signature)
        
        # Check results
        assert len(examples) == len(sample_data)
        assert all(example == "example_obj" for example in examples)
        
        # Check if dspy.Example.from_dict was called for each sample
        assert dspy_mocks["example"].from_dict.call_count == len(sample_data)

def test_optimization_metric(optimizer):
    """Test optimization metric function."""
    # Create example and prediction
    example = {"input": "What is the capital of France?", "output": "The capital of France is Paris."}
    pred = {"output": "Paris is the capital of France."}
    
    # Test with valid inputs
    with patch("metrics.judge_metric", return_value=(0.8, "Good response")):
        score = optimizer._optimization_metric(example, pred)
        assert score == 0.8
    
    # Test with exception
    with patch("metrics.judge_metric", side_effect=Exception("Test error")):
        score = optimizer._optimization_metric(example, pred)
        assert score == 0  # Should return 0 on error