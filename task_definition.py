"""
Module for managing task definitions
"""
import json
import os
from typing import Dict, Any, Optional, List

class TaskDefinition:
    """Manages the definition of DSPy tasks"""
    
    def __init__(self, name: str, input_fields: List[Dict[str, str]], output_fields: List[Dict[str, str]], 
                 description: str = ""):
        self.name = name
        self.input_fields = input_fields  # List of dicts with 'name' and 'desc' keys
        self.output_fields = output_fields  # List of dicts with 'name' and 'desc' keys
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "input_fields": self.input_fields,
            "output_fields": self.output_fields,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskDefinition':
        return cls(
            name=data.get("name", "CustomTask"),
            input_fields=data.get("input_fields", []),
            output_fields=data.get("output_fields", []),
            description=data.get("description", "")
        )
    
    def save(self, directory: str = "./task_definitions") -> None:
        """Save the task definition to disk"""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "task_definition.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, directory: str = "./task_definitions") -> Optional['TaskDefinition']:
        """Load the task definition from disk"""
        try:
            with open(os.path.join(directory, "task_definition.json"), "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Failed to load task definition: {e}")
            return None
    
    def create_dspy_signature(self):
        """Create a DSPy Signature class from this task definition"""
        import dspy
        
        # Create a new class dynamically
        attrs = {
            "__doc__": self.description,
        }
        
        # Add input fields
        for field in self.input_fields:
            attrs[field["name"]] = dspy.InputField(desc=field["desc"])
        
        # Add output fields
        for field in self.output_fields:
            attrs[field["name"]] = dspy.OutputField(desc=field["desc"])
        
        # Create the class
        signature_class = type(self.name, (dspy.Signature,), attrs)
        return signature_class
