"""Tests for the SampleManager class."""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from app.services.samples import SampleManager
from app.config import Config

def test_create_empty_sample(sample_manager, app_state, basic_signature):
    """Test creating an empty sample."""
    # Add signature to app state
    app_state.add_signature(basic_signature)
    app_state.set_current_signature(basic_signature.name)
    
    # Create empty sample
    sample = sample_manager.create_empty_sample()
    
    # Check if sample has expected fields
    assert "query" in sample
    assert "response" in sample
    assert sample["query"] == ""
    assert sample["response"] == ""

def test_validate_sample_valid(sample_manager, app_state, basic_signature, sample_data):
    """Test validating a valid sample."""
    # Add signature to app state
    app_state.add_signature(basic_signature)
    app_state.set_current_signature(basic_signature.name)
    
    # Validate sample
    result = sample_manager.validate_sample(sample_data[0])
    
    # Sample should be valid
    assert result is True

def test_validate_sample_invalid(sample_manager, app_state, basic_signature):
    """Test validating an invalid sample."""
    # Add signature to app state
    app_state.add_signature(basic_signature)
    app_state.set_current_signature(basic_signature.name)
    
    # Create invalid sample (missing required field)
    invalid_sample = {"query": "What is the capital of France?"}
    
    # Validate sample
    result = sample_manager.validate_sample(invalid_sample)
    
    # Sample should be invalid
    assert result is False

@patch("os.path.exists", return_value=True) 
@patch("os.path.isfile", return_value=True)
def test_save_samples_integration(mock_isfile, mock_exists, temp_dir, sample_manager, app_state, basic_signature, sample_data):
    """Test saving samples to file integration test."""
    # Add signature to app state
    app_state.add_signature(basic_signature)
    app_state.set_current_signature(basic_signature.name)

    # Create a real temporary file path
    from pathlib import Path
    import tempfile
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    
    # Replace the path in Config to use our test path
    original_samples_dir = Config.SAMPLES_DIR
    Config.SAMPLES_DIR = Path(temp_dir)
    
    try:
        # Save samples
        with patch("pathlib.Path.open", return_value=open(temp_path, "w")):
            sample_manager.save_samples(basic_signature.name, sample_data)
            
        # The file should have been accessed
        assert temp_path.exists()
    finally:
        # Clean up
        Config.SAMPLES_DIR = original_samples_dir
        if temp_path.exists():
            temp_path.unlink()

@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='[{"query": "What is the capital of France?", "response": "The capital of France is Paris."}, {"query": "Who wrote Romeo and Juliet?", "response": "William Shakespeare wrote Romeo and Juliet."}]')
def test_load_samples(mock_file, mock_exists, sample_manager, app_state, basic_signature):
    """Test loading samples from file."""
    # Add signature to app state
    app_state.add_signature(basic_signature)
    
    # Load samples
    samples = sample_manager.load_samples(basic_signature.name)
    
    # Check if samples were loaded correctly
    assert len(samples) == 2
    assert samples[0]["query"] == "What is the capital of France?"
    assert samples[0]["response"] == "The capital of France is Paris."
    assert samples[1]["query"] == "Who wrote Romeo and Juliet?"
    assert samples[1]["response"] == "William Shakespeare wrote Romeo and Juliet."

@patch("os.path.exists", return_value=False)
def test_load_samples_nonexistent(mock_exists, sample_manager, app_state, basic_signature):
    """Test loading samples from a non-existent file."""
    # Add signature to app state
    app_state.add_signature(basic_signature)
    
    # Load samples from non-existent file
    samples = sample_manager.load_samples(basic_signature.name)
    
    # Should return an empty list
    assert samples == []