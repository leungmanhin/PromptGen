import dspy
from typing import Dict, Any, Tuple

class PLNJudgeSignature(dspy.Signature):
    """You are a Judge for the following task:
    Convert English text to Programming Language for Thought (PLN).
    Slight differences in naming or the structure of the true output and the predicted output are allowed.
    But the predicted output should not contain anything more or less than the true output.
    """
    task_input = dspy.InputField(desc="Input to the PLN task")
    true_pln_types = dspy.InputField(desc="True PLN type definitions from the example")
    true_pln_statements = dspy.InputField(desc="True PLN statements from the example")
    true_pln_questions = dspy.InputField(desc="True PLN questions from the example")
    pred_pln_types = dspy.InputField(desc="Predicted PLN type definitions from the model")
    pred_pln_statements = dspy.InputField(desc="Predicted PLN statements from the model")
    pred_pln_questions = dspy.InputField(desc="Predicted PLN questions from the model")
    similarity = dspy.OutputField(desc="Similarity score between true and predicted outputs (0.0 to 1.0)")

def judge_metric(example, pred, trace=None) -> Tuple[float, str]:
    # Create a ChainOfThought module with the signature
    judge = dspy.ChainOfThought(PLNJudgeSignature)

    res = judge(
        task_input=example.english,
        true_pln_types=example.pln_types,
        true_pln_statements=example.pln_statements,
        true_pln_questions=example.pln_questions,
        pred_pln_types=pred.pln_types,
        pred_pln_statements=pred.pln_statements,
        pred_pln_questions=pred.pln_questions,
    )
    # Run the judge
    return res.similarity , res.reasoning 
