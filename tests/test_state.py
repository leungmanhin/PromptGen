"""Tests for the AppState class."""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from app.services.state import AppState

def test_singleton_pattern(app_state):
    """Test that AppState follows the singleton pattern."""
    # Get another instance of AppState
    another_state = AppState()
    
    # Both instances should be the same object
    assert app_state is another_state
    assert id(app_state) == id(another_state)

def test_ensure_directories(temp_dir):
    """Test directory initialization."""
    # Import Config for testing
    from app.config import Config
    
    # Configure temp directories
    temp_path = Path(temp_dir)
    Config.PROGRAM_DIR = temp_path / "programs"
    Config.SAMPLES_DIR = temp_path / "samples"
    Config.SIGNATURES_DIR = temp_path / "signatures"
    
    # Reset singleton
    AppState._instance = None
    
    # Create new instance (should call _ensure_directories internally)
    state = AppState()
    
    # Check if directories were created
    assert Config.PROGRAM_DIR.exists()
    assert Config.SAMPLES_DIR.exists()
    assert Config.SIGNATURES_DIR.exists()
    
    # Reset singleton for other tests
    AppState._instance = None

def test_add_get_signature(app_state, basic_signature):
    """Test adding and retrieving signatures."""
    # Add signature
    app_state.add_signature(basic_signature)
    
    # Get signature
    signature = app_state.get_signature(basic_signature.name)
    
    # Check if signature is the same
    assert signature.name == basic_signature.name
    assert signature.description == basic_signature.description
    assert len(signature.input_fields) == len(basic_signature.input_fields)
    assert len(signature.output_fields) == len(basic_signature.output_fields)

def test_set_current_signature(app_state, basic_signature):
    """Test setting current signature."""
    # Add signature
    app_state.add_signature(basic_signature)
    
    # Set current signature
    app_state.set_current_signature(basic_signature.name)
    
    # Check if current signature is set
    current_sig = app_state.get_current_signature()
    assert current_sig.name == basic_signature.name
    
    # Test with non-existent signature
    result = app_state.set_current_signature("non_existent_signature")
    # Should return False for non-existent signatures
    assert result is False

@patch("builtins.open", new_callable=mock_open, read_data="{}")
@patch("json.dump")
def test_save_signature(mock_json_dump, mock_file, app_state, basic_signature):
    """Test saving a signature to file."""
    # The _save_signature method is private, but it's called by add_signature
    # So we'll test add_signature instead
    
    # Add signature
    result = app_state.add_signature(basic_signature)
    
    # Check result
    assert result is True
    
    # Check if file was opened with the right path
    from app.config import Config
    expected_path = Config.SIGNATURES_DIR / f"{basic_signature.name}.json"
    mock_file.assert_called_with(expected_path, "w")
    
    # Check if json.dump was called with the right arguments
    mock_json_dump.assert_called()
    # The first argument to json.dump should be the signature dict
    signature_dict = mock_json_dump.call_args[0][0]
    assert signature_dict["name"] == basic_signature.name

def test_load_default_signatures(app_state):
    """Test loading default signatures."""
    # This test leverages the fixture which already loads default signatures
    
    # Since we're using a fresh app_state, it should have loaded the default PLN signature
    assert len(app_state.signatures) > 0
    
    # The PLNTask signature should be among the loaded signatures
    assert "PLNTask" in app_state.signatures
    
    # Check that the current signature name is set
    assert app_state.current_signature_name is not None