from typing import List, Dict
import dspy
import json
import os
from .samples import SampleManager
from .state import AppState

class Optimizer:
    """Handles optimization process for the language model"""
    
    def __init__(self, app_state: AppState, sample_manager: SampleManager):
        self.app_state = app_state
        self.sample_manager = sample_manager
        self.running = False

    def _prepare_training_data(self, samples: List[Dict]) -> List[dspy.Example]:
        """Prepare DSPy examples from loaded samples based on task definition"""
        task_def = self.app_state.task_definition
        
        examples = []
        for d in samples:
            # Create example with dynamic fields based on task definition
            example_data = {}
            
            # Add input fields
            for field in task_def.input_fields:
                field_name = field["name"]
                if field_name in d:
                    example_data[field_name] = d[field_name]
            
            # Add output fields
            for field in task_def.output_fields:
                field_name = field["name"]
                if field_name in d:
                    example_data[field_name] = d[field_name]
            
            # Create example with the first input field as the input
            if task_def.input_fields:
                input_field = task_def.input_fields[0]["name"]
                if input_field in example_
                    example = dspy.Example(**example_data).with_inputs(input_field)
                    examples.append(example)
        
        return examples

    def run_optimization(self, model_name: str) -> None:
        """Run the optimization process with thread safety"""
        if self.running:
            return

        self.running = True
        try:
            thread_lm = dspy.LM(model_name)
            if not thread_lm:
                return

            # Get the task definition
            task_def = self.app_state.task_definition
            if task_def is None:
                raise ValueError("No task definition found")
            
            # Create the DSPy signature from the task definition
            signature_class = task_def.create_dspy_signature()
            
            samples = self.sample_manager.load_samples()
            training_data = self._prepare_training_data(samples)
            
            if not training_
                raise ValueError("No valid training data found")

            # Create task using the custom signature
            task = dspy.ChainOfThought(signature_class)
            
            optimized_task = dspy.MIPROv2(
                metric=self._optimization_metric,
                auto="light"
            ).compile(task, trainset=training_data, requires_permission_to_run=False)
            
            optimized_task.save("./program/", save_program=True)
        except Exception as e:
            print(f"Optimization error: {e}")
        finally:
            self.running = False

    def _optimization_metric(self, example, pred, trace=None) -> float:
        """Metric for optimization process using dynamic fields from task definition"""
        task_def = self.app_state.task_definition
        
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
        return judge(**judge_args).similarity
