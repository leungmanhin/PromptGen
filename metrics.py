import dspy
from typing import Dict, Any, Tuple, List
from cleanpln import cleanAndScore

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

def clean_pln_list(pln_text: str) -> Tuple[List[str], float]:
    """Clean a string of PLN statements or questions and return the cleaned list and minimum score."""
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
    
    # Run the judge
    return adjusted_similarity, res.reasoning
