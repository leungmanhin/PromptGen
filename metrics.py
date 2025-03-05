import dspy

def judge_metric(example, pred, task_def, trace=None) -> float:
        # Create a dynamic judge signature based on task definition
        judge_signature = 'true_'
        judge_signature += ', true_'.join([f["name"] for f in task_def.output_fields])
        judge_signature += ', pred_'
        judge_signature += ', pred_'.join([f["name"] for f in task_def.output_fields])
        judge_signature += ' -> similarity: float'
        
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
