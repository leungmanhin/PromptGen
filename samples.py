import json
from typing import List, Dict
import os
import dspy
from .state import AppState

class NewSampleGenerator(dspy.Signature):
    """
    You are helping to create a new sample for training a Natural Language to PLN (Probabilistic Logic Network) system.
    
    Based on the evaluation of a previous sample, generate a NEW sample that addresses similar concepts 
    but is different in content. The new sample should:
    1. Cover similar concepts but with different content
    2. Address the weaknesses identified in the evaluation
    3. Have clear PLN types, statements, and query that follow the format shown in the examples
    """
    input_english = dspy.InputField(desc="Original English input")
    expected_pln_types = dspy.InputField(desc="Expected PLN types from the original sample")
    expected_pln_statements = dspy.InputField(desc="Expected PLN statements from the original sample")
    expected_pln_query = dspy.InputField(desc="Expected PLN query from the original sample")
    predicted_pln_types = dspy.InputField(desc="Model's predicted PLN types")
    predicted_pln_statements = dspy.InputField(desc="Model's predicted PLN statements")
    predicted_pln_query = dspy.InputField(desc="Model's predicted PLN query")
    similarity_explanation = dspy.InputField(desc="Explanation of the similarity between expected and predicted outputs")
    overall_score = dspy.InputField(desc="Overall similarity score")
    program_instructions = dspy.InputField(desc="Instructions for the program generating PLN from English")
    
    english = dspy.OutputField(desc="New English input for the sample")
    pln_types = dspy.OutputField(desc="New PLN types")
    pln_statements = dspy.OutputField(desc="New PLN statements")
    pln_query = dspy.OutputField(desc="New PLN query")

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
        required_fields = ["english", "pln_types", "pln_statements", "pln_query"]
        return all(field in sample for field in required_fields)
        
    def create_empty_sample(self) -> Dict:
        """Create an empty sample for PLN task"""
        return {
            "english": "",
            "pln_types": "",
            "pln_statements": "",
            "pln_query": ""
        }
        
    def generate_new_sample_from_evaluation(self, eval_result: Dict, model_name: str, program_instructions: str = "") -> Dict:
        """Generate a new sample using LLM based on evaluation results"""
        try:
            # Create a ChainOfThought module with the NewSampleGenerator signature
            sample_generator = dspy.ChainOfThought(NewSampleGenerator)
            
            # Get a model instance
            sample_lm = dspy.LM(model_name)
            
            # Call the LLM to generate a new sample
            with dspy.context(lm=sample_lm):
                pred = sample_generator(
                    input_english=eval_result.get("input_english", ""),
                    expected_pln_types=eval_result.get("expected_pln_types", ""),
                    expected_pln_statements=eval_result.get("expected_pln_statements", ""),
                    expected_pln_query=eval_result.get("expected_pln_query", ""),
                    predicted_pln_types=eval_result.get("predicted_pln_types", ""),
                    predicted_pln_statements=eval_result.get("predicted_pln_statements", ""),
                    predicted_pln_query=eval_result.get("predicted_pln_query", ""),
                    similarity_explanation=eval_result.get("similarity_result", {}).get("explanation", ""),
                    overall_score=str(eval_result.get("overall_score", 0)),
                    program_instructions=program_instructions
                )
            
            # Create a new sample from the predicted values
            new_sample = {
                "english": pred.english,
                "pln_types": pred.pln_types,
                "pln_statements": pred.pln_statements,
                "pln_query": pred.pln_query
            }
            
            return new_sample
            
        except Exception as e:
            print(f"Error generating new sample with LLM: {e}")
            # Fall back to using the evaluation data directly
            return {
                "english": eval_result.get("input_english", ""),
                "pln_types": eval_result.get("predicted_pln_types", ""),
                "pln_statements": eval_result.get("predicted_pln_statements", ""),
                "pln_query": eval_result.get("predicted_pln_query", "")
            }
