import dspy
from typing import Dict, Any

def create_judge_signature(task_def):
    """Create a DSPy Signature class for the judge based on task definition"""
    
    # Create a new class dynamically
    attrs = {
        "__doc__": f"""You are a Judge for the following task:
        {task_def.description}
        Slight differences in naming or the structure of the true output and the predicted output are allowed.
        But the predicted output should not contain anything more or less than the true output.
        """
    }

    attrs["task_input"] = dspy.InputField(
        desc=f"Input to the {task_def.name} task"
    )
    
    # Add input fields for true values
    for field in task_def.output_fields:
        field_name = field["name"]
        attrs[f"true_{field_name}"] = dspy.InputField(
            desc=f"True {field_name} from the example"
        )
    
    # Add input fields for predicted values
    for field in task_def.output_fields:
        field_name = field["name"]
        attrs[f"pred_{field_name}"] = dspy.InputField(
            desc=f"Predicted {field_name} from the model"
        )
    
    # Add output field for similarity score
    attrs["similarity"] = dspy.OutputField(
        desc="Similarity score between true and predicted outputs (0.0 to 1.0)"
    )
    
    # Create the class
    signature_class = type(f"{task_def.name}JudgeSignature", (dspy.Signature,), attrs)
    return signature_class

def judge_metric(example, pred, task_def, trace=None) -> float:
    # Create a dynamic judge signature class based on task definition
    judge_signature = create_judge_signature(task_def)
    
    # Create a ChainOfThought module with the signature
    judge = dspy.ChainOfThought(judge_signature)
    
    # Create arguments for the judge
    judge_args = {}
    
    # Add true values from example
    for field in task_def.output_fields:
        field_name = field["name"]
        judge_args[f"true_{field_name}"] = getattr(example, field_name, "")
    
    # Add predicted values
    for field in task_def.output_fields:
        field_name = field["name"]
        judge_args[f"pred_{field_name}"] = getattr(pred, field_name, "")
    
    # Run the judge
    return judge(**judge_args)
