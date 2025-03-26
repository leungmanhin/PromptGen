"""
Module for managing samples for different signatures
"""
import json
import time
import importlib
from typing import List, Dict, Any, Optional
from pathlib import Path

import dspy

from ..models.signature import SignatureDefinition
from ..config import Config
from .state import AppState

class SampleManager:
    """Handles loading and saving of sample data for different signatures"""
    
    def __init__(self, app_state: AppState = None):
        self.app_state = app_state if app_state else AppState()
        Config.SAMPLES_DIR.mkdir(exist_ok=True)

    def get_sample_file_for_signature(self, signature_name: str) -> Path:
        """Get the sample file path for a specific signature"""
        return Config.SAMPLES_DIR / f"{signature_name}_samples.json"

    def load_samples(self, signature_name: Optional[str] = None) -> List[Dict]:
        """Load samples from JSON file for a specific signature
        
        Args:
            signature_name (str, optional): Name of the signature to load samples for.
                                         If None, uses the current signature.
        
        Returns:
            List[Dict]: List of samples for the signature
        """
        sig_name = signature_name or self.app_state.current_signature_name
        if not sig_name:
            return []
        
        sample_file = self.get_sample_file_for_signature(sig_name)
        try:
            with open(sample_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_samples(self, samples: List[Dict], signature_name: Optional[str] = None) -> None:
        """Save samples to JSON file for a specific signature
        
        Args:
            samples (List[Dict]): List of samples to save
            signature_name (str, optional): Name of the signature to save samples for.
                                         If None, uses the current signature.
        """
        sig_name = signature_name or self.app_state.current_signature_name
        if not sig_name:
            return
        
        sample_file = self.get_sample_file_for_signature(sig_name)
        with open(sample_file, "w") as f:
            json.dump(samples, f, indent=2)

    def validate_sample(self, sample: Dict, signature_name: Optional[str] = None) -> bool:
        """Validate sample structure for a specific signature
        
        Args:
            sample (Dict): The sample to validate
            signature_name (str, optional): Name of the signature to validate against.
                                         If None, uses the current signature.
        
        Returns:
            bool: True if the sample is valid, False otherwise
        """
        sig_name = signature_name or self.app_state.current_signature_name
        if not sig_name:
            return False
        
        signature = self.app_state.get_signature(sig_name)
        if not signature:
            return False
        
        # All input fields must be present as keys in the sample
        for field in signature.input_fields:
            if field not in sample:
                return False
        
        # All output fields must be present as keys in the sample
        for field in signature.output_fields:
            if field not in sample:
                return False
                
        return True
        
    def create_empty_sample(self, signature_name: Optional[str] = None) -> Dict:
        """Create an empty sample for a specific signature
        
        Args:
            signature_name (str, optional): Name of the signature to create sample for.
                                         If None, uses the current signature.
        
        Returns:
            Dict: An empty sample with all required fields
        """
        sig_name = signature_name or self.app_state.current_signature_name
        if not sig_name:
            return {}
        
        signature = self.app_state.get_signature(sig_name)
        if not signature:
            return {}
        
        sample = {}
        # Initialize all input and output fields to empty strings
        for field in signature.input_fields:
            sample[field] = ""
        
        for field in signature.output_fields:
            sample[field] = ""
            
        return sample
    
    def _create_sample_generator_for_signature(self, signature: SignatureDefinition, eval_result: Dict) -> type:
        """Create a dynamic sample generator class for a signature
        
        Args:
            signature (SignatureDefinition): The signature to create a generator for
            eval_result (Dict): The evaluation result to base field types on
            
        Returns:
            type: A dynamic dspy.Signature class for generating samples
        """
        # Create a dynamic module for the generator
        module_name = f"dynamic_generator_{int(time.time())}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        module.dspy = dspy
        
        # Build class definition with dynamic fields based on signature
        class_def = f"""class {signature.name}SampleGenerator(dspy.Signature):
    \"\"\"
    You are helping to create a new sample for training a system.
    
    Based on the evaluation of a previous sample, generate a COMPLETELY NEW sample that addresses similar concepts 
    but is different in content. The new sample should:
    1. Have BOTH new inputs and new outputs that are different from the original sample
    2. Cover similar concepts but with different content, examples, and scenarios
    3. Address the weaknesses identified in the evaluation
    4. Have clear inputs and outputs that follow the format shown in the examples
    5. Be well-formed and complete
    \"\"\"
"""
        
        # Add input fields for original sample and evaluation
        for field in signature.input_fields:
            # Check if the field is a list in the original sample
            original_value = eval_result.get(f"input_{field}", "")
            is_list = isinstance(original_value, list)
            type_annotation = "list[str]" if is_list else "str"
            
            class_def += f"    input_{field}: {type_annotation} = dspy.InputField(desc=\"Original {field} input\")\n"
        
        for field in signature.output_fields:
            # Check if the field is a list in the original sample
            expected_value = eval_result.get(f"expected_{field}", "")
            is_list = isinstance(expected_value, list)
            type_annotation = "list[str]" if is_list else "str"
            
            class_def += f"    expected_{field}: {type_annotation} = dspy.InputField(desc=\"Expected {field} from the original sample\")\n"
            class_def += f"    predicted_{field}: {type_annotation} = dspy.InputField(desc=\"Model's predicted {field}\")\n"
        
        class_def += "    similarity_explanation: str = dspy.InputField(desc=\"Explanation of the similarity between expected and predicted outputs\")\n"
        class_def += "    overall_score: str = dspy.InputField(desc=\"Overall similarity score\")\n"
        class_def += "    program_instructions: str = dspy.InputField(desc=\"Instructions for the program\")\n"
        
        # Add output fields for new sample (both input and output fields)
        for field in signature.input_fields:
            # Check if the field is a list in the original sample
            original_value = eval_result.get(f"input_{field}", "")
            is_list = isinstance(original_value, list)
            type_annotation = "list[str]" if is_list else "str"
            
            class_def += f"    {field}: {type_annotation} = dspy.OutputField(desc=\"New {field} input that is different from the original but covers similar concepts\")\n"
            
        for field in signature.output_fields:
            # Check if the field is a list in the original sample
            expected_value = eval_result.get(f"expected_{field}", "")
            is_list = isinstance(expected_value, list) 
            type_annotation = "list[str]" if is_list else "str"
            
            class_def += f"    {field}: {type_annotation} = dspy.OutputField(desc=\"New {field} output appropriate for the new input\")\n"
        
        # Execute the class definition in the module's context
        exec(class_def, module.__dict__)
        
        # Get the class from the module
        generator_class = getattr(module, f"{signature.name}SampleGenerator")
        return generator_class
        
    def generate_new_sample_from_evaluation(self, eval_result: Dict, model_name: str, program_instructions: str = "", signature_name: Optional[str] = None) -> Dict:
        """Generate a new sample using LLM based on evaluation results
        
        Args:
            eval_result (Dict): Evaluation results to base the new sample on
            model_name (str): Name of the model to use for generation
            program_instructions (str, optional): Instructions for the program
            signature_name (str, optional): Name of the signature to generate for.
                                         If None, uses the current signature.
        
        Returns:
            Dict: A new sample based on the evaluation results
        """
        sig_name = signature_name or self.app_state.current_signature_name
        if not sig_name:
            return {}
        
        signature = self.app_state.get_signature(sig_name)
        if not signature:
            return {}
        
        try:
            # Create a sample generator for the signature
            generator_class = self._create_sample_generator_for_signature(signature, eval_result)
            sample_generator = dspy.ChainOfThought(generator_class)
            
            # Get a model instance
            from ..utils import get_lm
            sample_lm = get_lm(model_name)
            
            # Prepare kwargs for the generator
            kwargs = {
                "program_instructions": program_instructions,
                "similarity_explanation": eval_result.get("similarity_result", {}).get("explanation", ""),
                "overall_score": str(eval_result.get("overall_score", 0))
            }
            
            # Add input fields
            for field in signature.input_fields:
                kwargs[f"input_{field}"] = eval_result.get(f"input_{field}", "")
            
            # Add expected and predicted output fields
            for field in signature.output_fields:
                kwargs[f"expected_{field}"] = eval_result.get(f"expected_{field}", "")
                kwargs[f"predicted_{field}"] = eval_result.get(f"predicted_{field}", "")
            
            # Call the LLM to generate a new sample
            with dspy.context(lm=sample_lm):
                pred = sample_generator(**kwargs)
            
            # Create a new sample from the predicted values
            new_sample = {}
            
            # Get input fields from the prediction
            for field in signature.input_fields:
                field_value = getattr(pred, field, "")
                # Fall back to original input if the prediction is empty
                if not field_value:
                    field_value = eval_result.get(f"input_{field}", "")
                new_sample[field] = field_value
            
            # Get output fields from the prediction
            for field in signature.output_fields:
                field_value = getattr(pred, field, "")
                new_sample[field] = field_value
            
            dspy.inspect_history(n=1)
            
            return new_sample
            
        except Exception as e:
            print(f"Error generating new sample with LLM: {e}")
            # Fall back to using the evaluation data directly
            new_sample = {}
            for field in signature.input_fields:
                original_input = eval_result.get(f"input_{field}", "")
                # Add a note to encourage the user to modify the input
                if original_input:
                    new_sample[field] = f"[PLEASE MODIFY THIS TO MAKE A NEW SAMPLE]\n\n{original_input}"
                else:
                    new_sample[field] = ""
            
            for field in signature.output_fields:
                new_sample[field] = eval_result.get(f"predicted_{field}", "")
            
            return new_sample
