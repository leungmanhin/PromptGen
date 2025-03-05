import dspy
from typing import List, Dict, Any, Optional
from .models import ModelManager
from .samples import SampleManager
from .state import AppState

class Evaluator:
    """Handles evaluation of the optimized model against sample data"""
    
    def __init__(self, app_state: AppState, model_manager: ModelManager, sample_manager: SampleManager):
        self.app_state = app_state
        self.model_manager = model_manager
        self.sample_manager = sample_manager
    
    def load_optimized_task(self):
        """Load the optimized task from disk"""
        try:
            optimized_task = dspy.load("./program/")
            print(f"Successfully loaded optimized task: {type(optimized_task)}")
            return optimized_task
        except Exception as e:
            print(f"Failed to load optimized task: {e}")
            return None
    
    def evaluate_similarity_with_llm(self, expected: str, predicted: str, lm: dspy.LM) -> Dict[str, Any]:
        """Use LLM to evaluate similarity between expected and predicted outputs"""
        
        class SimilarityEvaluator(dspy.Signature):
            """Evaluate the similarity between expected and predicted outputs."""
            expected = dspy.InputField(desc="The expected output")
            predicted = dspy.InputField(desc="The predicted output")
            explanation = dspy.OutputField(desc="Explanation of the similarity score")
            similarity_score = dspy.OutputField(desc="Similarity score from 0-100")
        
        evaluator = dspy.Predict(SimilarityEvaluator)
        
        try:
            with dspy.context(lm=lm):
                result = evaluator(expected=expected, predicted=predicted)
                
            # Try to convert score to int, default to 0 if not possible
            try:
                score = int(result.similarity_score)
                # Ensure score is within 0-100 range
                score = max(0, min(100, score))
            except (ValueError, TypeError):
                score = 0
                
            return {
                "score": score,
                "explanation": result.explanation
            }
        except Exception as e:
            print(f"Error in LLM similarity evaluation: {e}")
            return {
                "score": 0,
                "explanation": f"Error: {str(e)}"
            }
    
    def run_evaluation(self, model_name: str, similarity_model_name: Optional[str] = None) -> Dict[str, Any]:
        """Run the evaluation on the optimized model
        
        Args:
            model_name: Model to use for running the optimized task
            similarity_model_name: Model to use for similarity scoring (defaults to model_name if None)
        """
        try:
            # Get model instances
            eval_lm = self.model_manager.get_lm_instance(model_name)
            if eval_lm is None:
                return {
                    "error": f"Failed to initialize evaluation model {model_name}",
                    "metrics": {},
                    "full_output": f"Error initializing evaluation model {model_name}"
                }
            
            # Use the same model for similarity if not specified
            if similarity_model_name is None or similarity_model_name == "":
                similarity_lm = eval_lm
            else:
                similarity_lm = self.model_manager.get_lm_instance(similarity_model_name)
                if similarity_lm is None:
                    return {
                        "error": f"Failed to initialize similarity model {similarity_model_name}",
                        "metrics": {},
                        "full_output": f"Error initializing similarity model {similarity_model_name}"
                    }
                
            # Load optimized task
            optimized_task = self.load_optimized_task()
            if optimized_task is None:
                return {
                    "error": "Failed to load optimized task",
                    "metrics": {},
                    "full_output": "Error loading optimized task"
                }
            
            # Load samples
            samples = self.sample_manager.load_samples()
            if not samples:
                return {
                    "error": "No samples to evaluate",
                    "metrics": {},
                    "full_output": "Error: No samples found"
                }
                
            # Evaluate model on samples
            results = []
            for i, sample in enumerate(samples):
                try:
                    print(f"Evaluating sample {i+1}/{len(samples)}: {sample['input'][:50]}...")
                    english = sample["input"]
                    expected_types = sample["types"]
                    expected_statements = sample["statements"]
                    expected_questions = sample.get("questions", "")
                    
                    # Run the model on the input with the specific LM instance
                    with dspy.context(lm=eval_lm):
                        prediction = optimized_task(english=english)
                    
                    # Calculate exact match metrics
                    types_match = expected_types.strip() == prediction.pln_types.strip()
                    statements_match = expected_statements.strip() == prediction.pln_statements.strip()
                    questions_match = expected_questions.strip() == prediction.pln_questions.strip()
                    
                    # Calculate LLM-based similarity scores
                    types_similarity = self.evaluate_similarity_with_llm(
                        expected_types, prediction.pln_types, similarity_lm
                    )
                    
                    statements_similarity = self.evaluate_similarity_with_llm(
                        expected_statements, prediction.pln_statements, similarity_lm
                    )
                    
                    questions_similarity = self.evaluate_similarity_with_llm(
                        expected_questions, prediction.pln_questions, similarity_lm
                    )
                    
                    # Calculate overall similarity score
                    overall_score = (
                        types_similarity["score"] + 
                        statements_similarity["score"] + 
                        questions_similarity["score"]
                    ) / 3
                    
                    # Store the results
                    results.append({
                        "sample_id": i,
                        "input": english,
                        "types_match": types_match,
                        "statements_match": statements_match,
                        "questions_match": questions_match,
                        "expected_types": expected_types,
                        "predicted_types": prediction.pln_types,
                        "expected_statements": expected_statements,
                        "predicted_statements": prediction.pln_statements,
                        "expected_questions": expected_questions,
                        "predicted_questions": prediction.pln_questions,
                        "types_similarity": types_similarity,
                        "statements_similarity": statements_similarity,
                        "questions_similarity": questions_similarity,
                        "overall_score": overall_score
                    })
                except Exception as e:
                    print(f"Error evaluating sample {i+1}: {e}")
                    results.append({
                        "sample_id": i,
                        "input": sample["input"],
                        "types_match": False,
                        "statements_match": False,
                        "questions_match": False,
                        "expected_types": sample["types"],
                        "predicted_types": "ERROR",
                        "expected_statements": sample["statements"],
                        "predicted_statements": "ERROR",
                        "expected_questions": sample.get("questions", ""),
                        "predicted_questions": "ERROR",
                        "types_similarity": {"score": 0, "explanation": "Error"},
                        "statements_similarity": {"score": 0, "explanation": "Error"},
                        "questions_similarity": {"score": 0, "explanation": "Error"},
                        "overall_score": 0,
                        "error": str(e)
                    })
                    
            # Calculate overall metrics
            total = len(results)
            types_correct = sum(1 for r in results if r.get("types_match", False))
            statements_correct = sum(1 for r in results if r.get("statements_match", False))
            questions_correct = sum(1 for r in results if r.get("questions_match", False))
            all_correct = sum(1 for r in results if r.get("types_match", False) and r.get("statements_match", False) and r.get("questions_match", False))
            errors = sum(1 for r in results if "error" in r)
            
            # Calculate average similarity scores
            avg_types_similarity = sum(r.get("types_similarity", {}).get("score", 0) for r in results) / total if total > 0 else 0
            avg_statements_similarity = sum(r.get("statements_similarity", {}).get("score", 0) for r in results) / total if total > 0 else 0
            avg_questions_similarity = sum(r.get("questions_similarity", {}).get("score", 0) for r in results) / total if total > 0 else 0
            avg_overall_score = sum(r.get("overall_score", 0) for r in results) / total if total > 0 else 0
            
            # Generate metrics
            metrics = {
                "Types Correct": f"{types_correct}/{total} ({types_correct/total:.2%})",
                "Statements Correct": f"{statements_correct}/{total} ({statements_correct/total:.2%})",
                "Questions Correct": f"{questions_correct}/{total} ({questions_correct/total:.2%})",
                "All Components Correct": f"{all_correct}/{total} ({all_correct/total:.2%})",
                "Errors": f"{errors}/{total} ({errors/total:.2%})",
                "Avg Types Similarity": f"{avg_types_similarity:.1f}/100",
                "Avg Statements Similarity": f"{avg_statements_similarity:.1f}/100",
                "Avg Questions Similarity": f"{avg_questions_similarity:.1f}/100",
                "Avg Overall Score": f"{avg_overall_score:.1f}/100"
            }
            
            # Return the evaluation results
            return {
                "metrics": metrics,
                "results": results
            }
        except Exception as e:
            return {
                "error": f"Evaluation failed: {str(e)}",
                "metrics": {},
                "full_output": f"Error: {str(e)}"
            }
