"""
Evaluation service for DSPy programs
"""
import dspy
from typing import Dict, List, Any, Optional
import time

from ..config import Config
from .state import AppState
from .samples import SampleManager
from ..utils.metrics import judge_metric

class Evaluator:
    """Class for evaluating DSPy programs"""
    
    def __init__(self, app_state: AppState, sample_manager: SampleManager):
        self.app_state = app_state
        self.sample_manager = sample_manager
        self.running = False
    
    def run_evaluation(self, model_name: str, similarity_model_name: Optional[str] = None) -> Dict:
        """Run evaluation on current program
        
        Args:
            model_name (str): Name of model to use for evaluation
            similarity_model_name (str, optional): Name of model to use for similarity scoring
                                                If None, uses the same model as evaluation
        
        Returns:
            Dict: Evaluation results
        """
        self.running = True
        
        try:
            # Validate that we have all we need
            if not self.app_state.current_program_id:
                return {
                    "status": "error",
                    "message": "No program selected. Please select a program first."
                }
            
            # Get the signature for the current program
            program_metadata = self.app_state.programs.get(self.app_state.current_program_id, {})
            signature_name = program_metadata.get("signature_name")
            if not signature_name:
                return {
                    "status": "error",
                    "message": "Program has no associated signature."
                }
                
            signature = self.app_state.get_signature(signature_name)
            if not signature:
                return {
                    "status": "error",
                    "message": f"Signature '{signature_name}' not found."
                }
                
            # Load samples
            samples = self.sample_manager.load_samples(signature_name)
            if not samples:
                return {
                    "status": "error",
                    "message": f"No samples found for signature '{signature_name}'."
                }
                
            # Load the program
            program_path = Config.PROGRAM_DIR / self.app_state.current_program_id
            try:
                program = dspy.load(str(program_path))
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to load program: {e}"
                }
            
            # Configure models
            eval_lm = dspy.LM(model_name)
            sim_lm = dspy.LM(similarity_model_name) if similarity_model_name else eval_lm
            
            # Prepare results dictionary
            results = []
            metrics = {
                "avg_score": 0.0,
                "num_samples": len(samples),
            }
            total_score = 0.0
            
            # Evaluate each sample
            with dspy.context(lm=eval_lm):
                for i, sample in enumerate(samples):
                    print(i)
                    try:
                        # Prepare input data
                        input_data = {}
                        for field in signature.input_fields:
                            input_data[field] = sample.get(field, "")
                        
                        # Run the prediction
                        start_time = time.time()
                        pred = program(**input_data)
                        end_time = time.time()
                        
                        # Create a prediction result
                        pred_result = {
                            "sample_id": i,
                            "time_taken": end_time - start_time
                        }
                        
                        # Add input fields
                        for field in signature.input_fields:
                            pred_result[f"input_{field}"] = sample.get(field, "")
                        
                        # Add expected output fields
                        for field in signature.output_fields:
                            pred_result[f"expected_{field}"] = sample.get(field, "")
                            pred_result[f"predicted_{field}"] = getattr(pred, field, "")
                        
                        # Create Example objects for metric evaluation
                        example_data = {}
                        for field in signature.input_fields:
                            example_data[field] = sample.get(field, "")
                        for field in signature.output_fields:
                            example_data[field] = sample.get(field, "")
                        
                        example = dspy.Example(**example_data)
                        
                        # Evaluate the prediction using our metric
                        with dspy.context(lm=sim_lm):
                            score, explanation = judge_metric(example, pred)
                            
                        pred_result["overall_score"] = score
                        pred_result["similarity_result"] = {
                            "explanation": explanation
                        }
                        
                        # Add the result to our results list
                        results.append(pred_result)
                        
                        # Update total score
                        total_score += score
                        
                    except Exception as e:
                        # Handle errors for individual samples
                        results.append({
                            "sample_id": i,
                            "error": str(e),
                            "overall_score": 0.0
                        })
            
            # Calculate average score
            if len(samples) > 0:
                metrics["avg_score"] = total_score / len(samples)
            
            # Return the results
            return {
                "status": "success",
                "metrics": metrics,
                "results": results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            self.running = False
