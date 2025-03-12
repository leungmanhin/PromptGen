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

def clean_pln_list(pln_text: List[str]) -> Tuple[List[str], float]:
    """Clean a string of PLN statements or questions and return the cleaned list and minimum score."""
    from ..utils import cleanAndScore
    
    cleaned_list = []
    score_sum = 0.0
    cnt = 0.0
    
    for line in pln_text:
        line = line.strip()
        if line:  # Skip empty lines
            cleaned_item, score = cleanAndScore(line)
            cleaned_list.append(cleaned_item)
            score_sum += score
            cnt += 1.0
    
    return cleaned_list, score_sum / cnt if cnt > 0 else 1.0

def judge_metric(example, pred, trace=None) -> Tuple[float, str]:
    """Judge metric that handles different signature types
    
    Uses a generic judge for all signature types.
    Field processors are applied based on signature configuration.
    
    Args:
        example: The example (reference) to compare against
        pred: The prediction to evaluate
        trace: Optional trace information
        
    Returns:
        Tuple[float, str]: A tuple of (score, explanation)
    """
    # For backward compatibility with PLN tasks
    if (hasattr(example, 'english') and 
        hasattr(example, 'pln_types') and 
        hasattr(example, 'pln_statements') and 
        hasattr(example, 'pln_query')):
        
        # Get the app state
        app_state = AppState()
        
        # Check if PLN signature exists, otherwise create a temporary one
        if 'PLN' not in app_state.signatures:
            # Create a temporary PLN signature with field processors
            app_state.signatures['PLN'] = SignatureDefinition(
                name="PLN",
                signature_class_def="",
                description="Convert English text to Programming Language for Thought (PLN).",
                input_fields=["english"],
                output_fields=["pln_types", "pln_statements", "pln_query"],
                field_processors={
                    "pln_statements": "clean_pln_list",
                    "pln_query": "clean_pln_list"
                }
            )
    
    # Use the generic judge for all signatures
    return judge_generic_metric(example, pred, trace)

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

def get_processor_by_name(processor_name: str):
    """Get a processor function by name
    
    Args:
        processor_name: Name of the processor function
        
    Returns:
        Callable: The processor function, or None if not found
    """
    # Import possible processor modules
    from ..utils import cleanpln
    
    # Map of processor names to functions
    processors = {
        "clean_pln_list": clean_pln_list,
        "cleanAndScore": cleanpln.cleanAndScore,
        "cleanPLN": cleanpln.cleanPLN,
        "balance_parentheses": cleanpln.balance_parentheses,
        "checkStmt": cleanpln.checkStmt,
    }
    
    return processors.get(processor_name)

def process_field(field_name: str, field_value: str, signature: SignatureDefinition) -> Tuple[str, float]:
    """Process a field value according to the signature's field processor
    
    Args:
        field_name: Name of the field
        field_value: Value of the field
        signature: Signature definition
        
    Returns:
        Tuple[str, float]: Processed value and score
    """
    # If no field processor is defined, return the original value with a perfect score
    if not signature.field_processors or field_name not in signature.field_processors:
        return field_value, 1.0
    
    # Get the processor function
    processor_name = signature.field_processors[field_name]
    processor_fn = get_processor_by_name(processor_name)
    
    # If processor not found, return the original value with a perfect score
    if not processor_fn:
        return field_value, 1.0
    
    # Apply the processor
    if processor_name == "clean_pln_list":
        # Clean PLN list returns a tuple of (list, score)
        if not isinstance(field_value, list):
            processed_list, score = processor_fn(field_value.split('\n'))
            return '\n'.join(processed_list), score
        return processor_fn(field_value)
    elif processor_name in ["cleanAndScore", "balance_parentheses"]:
        # These processors return a tuple of (value, score)
        processed_value, score = processor_fn(field_value)
        return processed_value, score
    else:
        # Other processors just return a value (no score)
        return processor_fn(field_value), 1.0

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
    
    # Add true output fields
    for field in signature.output_fields:
        judge_inputs[f"true_{field}"] = getattr(example, field, "")
    
    # Add predicted output fields with processing if configured
    processing_scores = []
    for field in signature.output_fields:
        field_value = getattr(pred, field, "")
        processed_value, score = process_field(field, field_value, signature)
        judge_inputs[f"pred_{field}"] = processed_value
        processing_scores.append(score)
    
    # Run the judge
    res = judge(**judge_inputs)
    
    # Calculate average processing score
    avg_processing_score = sum(processing_scores) / len(processing_scores) if processing_scores else 1.0
    
    # Adjust similarity by processing score
    adjusted_similarity = res.similarity * avg_processing_score
    
    # Return the score and explanation
    return adjusted_similarity, res.explanation
