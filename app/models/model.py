"""
Module for model definition
"""
from typing import Dict, Optional

class ModelDefinition:
    """Class for managing model definitions and their configurations"""
    
    def __init__(self, 
                name: str, 
                provider: str,
                description: str = "", 
                parameters: Optional[Dict[str, any]] = None):
        """
        Initialize a model definition
        
        Args:
            name: Name of the model
            provider: Provider of the model (e.g., 'anthropic', 'openai')
            description: Description of the model
            parameters: Dictionary of model parameters (e.g., temperature, max_tokens)
        """
        self.name = name
        self.provider = provider
        self.description = description
        self.parameters = parameters or {}
    
    def to_dict(self) -> Dict:
        """Convert model definition to dictionary"""
        return {
            "name": self.name,
            "provider": self.provider,
            "description": self.description,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelDefinition':
        """Create model definition from dictionary"""
        return cls(
            name=data.get("name", ""),
            provider=data.get("provider", ""),
            description=data.get("description", ""),
            parameters=data.get("parameters", {})
        )
        
    @property
    def full_name(self) -> str:
        """Get full model name as provider/name"""
        return f"{self.provider}/{self.name}"