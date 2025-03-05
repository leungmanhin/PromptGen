import json
from typing import List, Dict
import os
from .state import AppState

class SampleManager:
    """Handles loading and saving of sample data"""
    
    def __init__(self, app_state: AppState = None, sample_file: str = "samples/generated_samples.json"):
        self.app_state = app_state if app_state else AppState()
        self.sample_file = sample_file
        os.makedirs(os.path.dirname(sample_file), exist_ok=True)

    def load_samples(self) -> List[Dict]:
        """Load samples from JSON file"""
        try:
            with open(self.sample_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_samples(self, samples: List[Dict]) -> None:
        """Save samples to JSON file"""
        with open(self.sample_file, "w") as f:
            json.dump(samples, f, indent=2)

    def validate_sample(self, sample: Dict) -> bool:
        """Validate sample structure based on task definition"""
        task_def = self.app_state.task_definition
        if not task_def:
            return False
            
        # Check that all input fields are present
        for field in task_def.input_fields:
            if field["name"] not in sample:
                return False
                
        # Check that all output fields are present
        for field in task_def.output_fields:
            if field["name"] not in sample:
                return False
                
        return True
        
    def create_empty_sample(self) -> Dict:
        """Create an empty sample based on the current task definition"""
        task_def = self.app_state.task_definition
        if not task_def:
            return {}
            
        sample = {}
        
        # Add input fields
        for field in task_def.input_fields:
            sample[field["name"]] = ""
            
        # Add output fields
        for field in task_def.output_fields:
            sample[field["name"]] = ""
            
        return sample
