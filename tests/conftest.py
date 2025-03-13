"""pytest configurations and fixtures."""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.state import AppState
from app.services.samples import SampleManager

@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def app_state(temp_dir):
    """Fixture to create a AppState instance for testing."""
    # Import Config for patching
    from app.config import Config
    
    # Save original values
    original_program_dir = Config.PROGRAM_DIR
    original_samples_dir = Config.SAMPLES_DIR
    original_signatures_dir = Config.SIGNATURES_DIR
    
    # Update Config paths to use temp directory
    temp_path = Path(temp_dir)
    Config.PROGRAM_DIR = temp_path / "programs"
    Config.SAMPLES_DIR = temp_path / "samples"
    Config.SIGNATURES_DIR = temp_path / "signatures"
    
    # Clear the singleton instance if it exists
    AppState._instance = None
    
    # Create app state (will initialize directories)
    state = AppState()
    
    yield state
    
    # Reset the Config paths
    Config.PROGRAM_DIR = original_program_dir
    Config.SAMPLES_DIR = original_samples_dir
    Config.SIGNATURES_DIR = original_signatures_dir
    
    # Reset the singleton
    AppState._instance = None

@pytest.fixture
def sample_manager(app_state):
    """Fixture to create a SampleManager instance for testing."""
    return SampleManager(app_state)

@pytest.fixture
def basic_signature():
    """Fixture for a basic signature definition."""
    from app.models.signature import SignatureDefinition
    
    return SignatureDefinition(
        name="TestSignature",
        signature_class_def="""class TestSignature(dspy.Signature):
    \"\"\"A test signature for simple question answering\"\"\"
    query = dspy.InputField(desc="The input query")
    response = dspy.OutputField(desc="The output response")
""",
        description="A test signature",
        input_fields=["query"],
        output_fields=["response"],
        field_processors={}
    )

@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    return [
        {
            "query": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "query": "Who wrote Romeo and Juliet?",
            "response": "William Shakespeare wrote Romeo and Juliet."
        }
    ]

@pytest.fixture
def mock_signature_instance(monkeypatch, basic_signature):
    """Mock the SignatureDefinition class methods needed in tests."""
    # First, store the original methods
    from app.models.signature import SignatureDefinition
    
    # Create a mock instance with the necessary attributes
    mock_instance = basic_signature
    
    return mock_instance