"""
Utility functions for program management
"""
import time
import json
from typing import Optional

from ..config import Config

def save_program(
    task,
    model_name: str,
    signature_name: str,
    signature_description: str,
    base_program_id: Optional[str] = None,
) -> str:
    """Save a DSPy program to disk and return its ID
    
    Args:
        task: The DSPy program/task to save
        model_name (str): The model used for the program
        signature_name (str): The signature name used
        signature_description (str): Description of the signature
        base_program_id (Optional[str]): ID of the base program, if any
        
    Returns:
        str: The ID of the created program
    """
    # Generate a unique ID for the program
    program_id = f"program_{int(time.time())}"
    program_dir = Config.PROGRAM_DIR / program_id
    
    # Create directory if it doesn't exist
    program_dir.mkdir(exist_ok=True)
    
    # Save program to the directory
    task.save(str(program_dir), save_program=True)
    
    # Add additional metadata
    metadata_path = program_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Add additional metadata
        metadata["model"] = model_name
        metadata["created_at"] = time.time()
        metadata["task_name"] = signature_description
        metadata["base_program_id"] = base_program_id
        metadata["signature_name"] = signature_name
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    return program_id