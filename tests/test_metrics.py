"""Tests for the metrics module."""

import pytest
from unittest.mock import patch, MagicMock

from app.utils import metrics
from app.utils.metrics import clean_pln_list as actual_clean_pln_list

@pytest.fixture
def example_prediction():
    """Fixture for example prediction data."""
    return {
        "input": "What is the capital of France?",
        "expected": "The capital of France is Paris.",
        "predicted": "Paris is the capital of France."
    }

@pytest.fixture
def pln_example():
    """Fixture for PLN example data."""
    return {
        "input": "Generate PLN statements for birds can fly",
        "expected": "Birds can fly.\nMost birds can fly.\nSome birds cannot fly.",
        "predicted": "Birds can fly.\nMost birds have the ability to fly.\nPenguins are birds that cannot fly."
    }

@patch("app.utils.metrics.create_dynamic_judge")
@patch("dspy.ChainOfThought")
def test_judge_generic_metric(mock_chain, mock_create_judge, example_prediction, app_state):
    """Test the generic judge metric."""
    # Create example and prediction as dspy-like objects
    class Example:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
                
    example = Example(input=example_prediction["input"], output=example_prediction["expected"])
    pred = Example(output=example_prediction["predicted"])
    
    # Mock the judge
    mock_judge = MagicMock()
    mock_judge.return_value.similarity = 0.8
    mock_judge.return_value.explanation = "The predictions are similar"
    mock_chain.return_value = mock_judge
    
    # Mock the get_signature method to return a proper signature
    from app.models.signature import SignatureDefinition
    mock_signature = SignatureDefinition(
        name="TestSignature",
        signature_class_def="",
        description="Test",
        input_fields=["input"],
        output_fields=["output"],
        field_processors={}
    )
    
    # Create a dynamic judge class
    mock_judge_class = MagicMock()
    mock_create_judge.return_value = mock_judge_class
    
    # Add a signature to app_state
    app_state.signatures["TestSignature"] = mock_signature
    app_state.current_signature_name = "TestSignature"
    
    # Set a mock for app_state.get_signature
    with patch.object(app_state, "get_signature", return_value=mock_signature):
        # Call judge_generic_metric
        score, explanation = metrics.judge_generic_metric(example, pred)
        
        # Check the score and explanation
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert isinstance(explanation, str)
        assert len(explanation) > 0

@patch("app.utils.metrics.clean_pln_list")
def test_process_field_pln(mock_clean_pln, pln_example):
    """Test processing PLN field."""
    # Mock clean_pln_list to return a known value
    mock_clean_pln.return_value = (["Birds can fly.", "Most birds can fly."], 0.9)
    
    # Create field and signature
    from app.models.signature import SignatureDefinition
    
    field_name = "pln_statements"
    field_value = pln_example["predicted"]
    signature = SignatureDefinition(
        name="PLNTask",
        signature_class_def="",
        description="Convert English to PLN",
        input_fields=["english"],
        output_fields=["pln_types", "pln_statements", "pln_query"],
        field_processors={"pln_statements": "clean_pln_list"}
    )
    
    # Process field
    result, score = metrics.process_field(field_name, field_value, signature)
    
    # Check if clean_pln_list was called
    mock_clean_pln.assert_called_once()
    
    # Check result and score
    assert result == "Birds can fly.\nMost birds can fly."
    assert score == 0.9

def test_process_field_text():
    """Test processing text field."""
    # Create field and signature
    from app.models.signature import SignatureDefinition
    
    field_name = "response"
    field_value = "This is a test."
    signature = SignatureDefinition(
        name="TestSignature",
        signature_class_def="",
        description="A test signature",
        input_fields=["query"],
        output_fields=["response"],
        field_processors={}  # No processor for the text field
    )
    
    # Process field
    result, score = metrics.process_field(field_name, field_value, signature)
    
    # Check result and score
    assert result == field_value
    assert score == 1.0

def test_clean_pln_list_integration():
    """Test the clean_pln_list function with actual data."""
    # Sample PLN statements
    pln_statements = [
        "Birds can fly.",
        "Most birds have wings.",
        "Some birds are flightless.",
        ""  # Empty line to test filtering
    ]
    
    # Call clean_pln_list
    cleaned_statements, score = actual_clean_pln_list(pln_statements)
    
    # Check results
    assert len(cleaned_statements) == 3  # Empty line should be filtered out
    assert 0 <= score <= 1
    assert "Birds can fly." in cleaned_statements
    assert "Most birds have wings." in cleaned_statements
    assert "Some birds are flightless." in cleaned_statements

@patch("dspy.ChainOfThought")
def test_create_dynamic_judge(mock_chain):
    """Test creating a dynamic judge class."""
    from app.models.signature import SignatureDefinition
    
    # Create signature
    signature = SignatureDefinition(
        name="TestSignature",
        signature_class_def="",
        description="A test signature",
        input_fields=["query"],
        output_fields=["response"],
        field_processors={}
    )
    
    # Mock the judge class behavior
    mock_judge_instance = MagicMock()
    mock_chain.return_value = mock_judge_instance
    
    # Create dynamic judge class
    judge_class = metrics.create_dynamic_judge(signature)
    
    # We should have created a dynamic class with the right name
    assert judge_class.__name__ == "TestSignatureJudgeSignature"
    
    # The signature definition should be available from the class
    assert hasattr(judge_class, "__module__")