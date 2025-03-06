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
        """Validate sample structure for PLN task"""
        required_fields = ["english", "pln_types", "pln_statements", "pln_questions"]
        return all(field in sample for field in required_fields)
        
    def create_empty_sample(self) -> Dict:
        """Create an empty sample for PLN task"""
        return {
            "english": "",
            "pln_types": "",
            "pln_statements": "",
            "pln_questions": ""
        }
