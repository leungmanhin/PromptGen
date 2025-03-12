"""
Metrics used for evaluating DSPy programs
"""
import dspy
import importlib
import time
from typing import Tuple, List

from ..models.signature import SignatureDefinition
from ..services.state import AppState

class PLNJudgeSignature(dspy.Signature):
    """You are a Judge for the following task:
    Convert English text to Programming Language for Thought (PLN).
    Slight differences in naming or the structure of the true output and the predicted output are allowed.
    But the predicted output should not contain anything more or less than the true output.
    """
    task_input = dspy.InputField(desc="Input to the PLN task")
    true_pln_types = dspy.InputField(desc="True PLN type definitions from the example")
    true_pln_statements = dspy.InputField(desc="True PLN statements from the example")
    true_pln_query = dspy.InputField(desc="True PLN query from the example")
    pred_pln_types = dspy.InputField(desc="Predicted PLN type definitions from the model")
    pred_pln_statements = dspy.InputField(desc="Predicted PLN statements from the model")
    pred_pln_query = dspy.InputField(desc="Predicted PLN query from the model")
    similarity: float = dspy.OutputField(desc="Similarity score between true and predicted outputs (0.0 to 1.0)")

class GenericJudgeSignature(dspy.Signature):
    """You are a Judge for a task.
    Your job is to compare the true output with the predicted output and determine their similarity.
    Slight differences in formatting or wording are allowed,
    but the predicted output should capture the same meaning as the true output.
    
    For each output field, carefully compare the true value with the predicted value and
    determine a similarity score (0.0 to 1.0) where:
    - 1.0 means the outputs are semantically identical
    - 0.0 means the outputs are completely different or unrelated
    - Values in between represent partial matches
    
    Provide an overall similarity score that takes into account all the output fields.
    """
    # These will be dynamically populated with actual field names when created
    task_description = dspy.InputField(desc="Description of the task")
    explanation = dspy.OutputField(desc="Detailed explanation of the similarity score")
    similarity: float = dspy.OutputField(desc="Overall similarity score between true and predicted outputs (0.0 to 1.0)")

def clean_pln_list(pln_text: str) -> Tuple[List[str], float]:
    """Clean a string of PLN statements or questions and return the cleaned list and minimum score."""
    from ..utils import cleanAndScore
    
    cleaned_list = []
    score_sum = 0.0
    cnt = 0.0
    
    # Split the text into individual lines
    lines = pln_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            cleaned_item, score = cleanAndScore(line)
            cleaned_list.append(cleaned_item)
            score_sum += score
            cnt += 1.0
    
    return cleaned_list, score_sum / cnt if cnt > 0 else 1.0

def judge_metric(example, pred, trace=None) -> Tuple[float, str]:
    """Judge metric that handles different signature types
    
    For PLN tasks, uses the specific PLN judge with cleaning.
    For other tasks, uses a generic judge.
    
    Args:
        example: The example (reference) to compare against
        pred: The prediction to evaluate
        trace: Optional trace information
        
    Returns:
        Tuple[float, str]: A tuple of (score, explanation)
    """
    # Check if this is a PLN task by looking for specific fields
    if (hasattr(example, 'english') and 
        hasattr(example, 'pln_types') and 
        hasattr(example, 'pln_statements') and 
        hasattr(example, 'pln_query')):
        return judge_pln_metric(example, pred, trace)
    else:
        return judge_generic_metric(example, pred, trace)

def judge_pln_metric(example, pred, trace=None) -> Tuple[float, str]:
    """Judge metric specifically for PLN tasks"""
    # Clean and score the predicted PLN statements and questions
    cleaned_pred_statements, stmt_score = clean_pln_list(pred.pln_statements)
    cleaned_pred_questions, ques_score = clean_pln_list(pred.pln_query)
    
    # Create a ChainOfThought module with the signature
    judge = dspy.ChainOfThought(PLNJudgeSignature)

    res = judge(
        task_input=example.english,
        true_pln_types=example.pln_types,
        true_pln_statements=example.pln_statements,
        true_pln_query=example.pln_query,
        pred_pln_types=pred.pln_types,
        pred_pln_statements='\n'.join(cleaned_pred_statements),
        pred_pln_query='\n'.join(cleaned_pred_questions),
    )
    
    # Adjust the similarity score based on the cleaning scores
    cleaning_score = (stmt_score + ques_score) / 2
    adjusted_similarity = res.similarity * cleaning_score
    
    # Return the score and reasoning
    return adjusted_similarity, getattr(res, 'reasoning', '')

def create_dynamic_judge(signature: SignatureDefinition) -> type:
    """Create a dynamic judge signature for a specific task signature
    
    Args:
        signature (SignatureDefinition): The signature to create a judge for
        
    Returns:
        type: A dynamic dspy.Signature class for judging
    """
    # Create a dynamic module for the judge
    module_name = f"dynamic_judge_{int(time.time())}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.dspy = dspy
    
    # Build class definition with dynamic fields based on signature
    class_def = f"""class {signature.name}JudgeSignature(dspy.Signature):
    \"\"\"
    You are a Judge for the {signature.name} task.
    Your job is to compare the true output with the predicted output and determine their similarity.
    Slight differences in formatting or wording are allowed,
    but the predicted output should capture the same meaning as the true output.
    
    For each output field, carefully compare the true value with the predicted value and
    determine a similarity score (0.0 to 1.0) where:
    - 1.0 means the outputs are semantically identical
    - 0.0 means the outputs are completely different or unrelated
    - Values in between represent partial matches
    
    Provide an overall similarity score that takes into account all the output fields.
    \"\"\"
    # Task description
    task_description = dspy.InputField(desc="Description of the {signature.name} task")
"""
    
    # Add input fields for the task
    for field in signature.input_fields:
        class_def += f"    {field} = dspy.InputField(desc=\"Input {field} for the task\")\n"
    
    # Add true output fields
    for field in signature.output_fields:
        class_def += f"    true_{field} = dspy.InputField(desc=\"True {field} from the example\")\n"
    
    # Add predicted output fields
    for field in signature.output_fields:
        class_def += f"    pred_{field} = dspy.InputField(desc=\"Predicted {field} from the model\")\n"
    
    # Add output fields for the judge
    class_def += "    explanation = dspy.OutputField(desc=\"Detailed explanation of the similarity score\")\n"
    class_def += "    similarity: float = dspy.OutputField(desc=\"Overall similarity score between true and predicted outputs (0.0 to 1.0)\")\n"
    
    # Execute the class definition in the module's context
    exec(class_def, module.__dict__)
    
    # Get the class from the module
    judge_class = getattr(module, f"{signature.name}JudgeSignature")
    return judge_class

def judge_generic_metric(example, pred, trace=None) -> Tuple[float, str]:
    """Judge metric for any signature type
    
    Dynamically creates a judge for the signature and uses it to evaluate.
    
    Args:
        example: The example (reference) to compare against
        pred: The prediction to evaluate
        trace: Optional trace information
        
    Returns:
        Tuple[float, str]: A tuple of (score, explanation)
    """
    # Get the app state
    app_state = AppState()
    
    # Determine which signature this is for
    signature_name = None
    for sig_name, signature in app_state.signatures.items():
        # Check if this example matches the signature's fields
        if all(hasattr(example, field) for field in signature.input_fields + signature.output_fields):
            signature_name = sig_name
            break
    
    if not signature_name:
        # Fall back to a simple similarity score if we can't determine the signature
        return 0.5, "Could not determine signature type for evaluation"
    
    # Get the signature
    signature = app_state.get_signature(signature_name)
    
    # Create a dynamic judge for this signature
    judge_class = create_dynamic_judge(signature)
    judge = dspy.ChainOfThought(judge_class)
    
    # Prepare the judge inputs
    judge_inputs = {
        "task_description": signature.description
    }
    
    # Add input fields
    for field in signature.input_fields:
        judge_inputs[field] = getattr(example, field, "")
    
    # Add true and predicted output fields
    for field in signature.output_fields:
        judge_inputs[f"true_{field}"] = getattr(example, field, "")
        judge_inputs[f"pred_{field}"] = getattr(pred, field, "")
    
    # Run the judge
    res = judge(**judge_inputs)
    
    # Return the score and explanation
    return res.similarity, res.explanation
