"""
Module for signature definition model
"""
from typing import Dict, List, Optional

class SignatureDefinition:
    """Class for managing signature definitions and their implementations"""
    
    def __init__(self, name: str, signature_class_def: str, description: str = "", input_fields: List[str] = None, output_fields: List[str] = None):
        self.name = name
        self.signature_class_def = signature_class_def
        self.description = description
        self.input_fields = input_fields or []
        self.output_fields = output_fields or []
    
    def to_dict(self) -> Dict:
        """Convert signature definition to dictionary"""
        return {
            "name": self.name,
            "signature_class_def": self.signature_class_def,
            "description": self.description,
            "input_fields": self.input_fields,
            "output_fields": self.output_fields
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SignatureDefinition':
        """Create signature definition from dictionary"""
        return cls(
            name=data.get("name", ""),
            signature_class_def=data.get("signature_class_def", ""),
            description=data.get("description", ""),
            input_fields=data.get("input_fields", []),
            output_fields=data.get("output_fields", [])
        )