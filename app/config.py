import os
from pathlib import Path

class Config:
    # Flask config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-for-promptgen'
    
    # Application paths
    BASE_DIR = Path(__file__).parent.parent
    PROGRAM_DIR = BASE_DIR / 'programs'
    SAMPLES_DIR = BASE_DIR / 'samples'
    SIGNATURES_DIR = BASE_DIR / 'signatures'
    
    # Ensure directories exist
    PROGRAM_DIR.mkdir(exist_ok=True)
    SAMPLES_DIR.mkdir(exist_ok=True)
    SIGNATURES_DIR.mkdir(exist_ok=True)
    
    # Default model
    DEFAULT_MODEL = 'anthropic/claude-3-7-sonnet-20250219'
