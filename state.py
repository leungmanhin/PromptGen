"""
Module for managing application state
"""
import threading
import os
import json
import time

class AppState:
    """Singleton class for managing application state"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AppState, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initialize the application state"""
        self.current_model = "anthropic/claude-3-5-sonnet-20241022"
        self.optimization_running = False
        self.evaluation_results = {"metrics": {}, "results": []}
        
        # Available models
        self.AVAILABLE_MODELS = [
            'openrouter/anthropic/claude-3.7-sonnet',
            'deepseek/deepseek-reasoner',
            'anthropic/claude-3-7-sonnet-20250219',
            'anthropic/claude-3-5-sonnet-20240620',
            'openai/gpt-4o'
        ]
        
        # Program management
        self.programs = {}  # Dictionary of program_id -> metadata
        self.current_program_id = None
        self._ensure_programs_directory()
        self.load_available_programs()
    
    def _ensure_programs_directory(self):
        """Create programs directory if it doesn't exist"""
        os.makedirs("./programs", exist_ok=True)
    
    def load_available_programs(self):
        """Load metadata for all available programs"""
        programs_dir = "./programs/"
        self.programs = {}
        
        # Find all program directories
        if os.path.exists(programs_dir):
            for program_id in os.listdir(programs_dir):
                program_path = os.path.join(programs_dir, program_id)
                if os.path.isdir(program_path):
                    metadata_path = os.path.join(program_path, "metadata.json")
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, "r") as f:
                                metadata = json.load(f)
                            # Add creation time from file if not in metadata
                            if "created_at" not in metadata:
                                metadata["created_at"] = os.path.getctime(metadata_path)
                            # Add program ID to metadata
                            metadata["id"] = program_id
                            # Add to programs dictionary
                            self.programs[program_id] = metadata
                        except Exception as e:
                            print(f"Error loading program {program_id}: {e}")
        
        # Set current program to the most recent if none is selected
        if not self.current_program_id and self.programs:
            # Sort by creation time (newest first)
            sorted_programs = sorted(self.programs.items(), 
                                    key=lambda x: x[1].get("created_at", 0), 
                                    reverse=True)
            self.current_program_id = sorted_programs[0][0]
    
    def set_current_program(self, program_id):
        """Set the current program"""
        if program_id in self.programs:
            self.current_program_id = program_id
            return True
        return False
