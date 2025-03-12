"""
Module for optimizing DSPy programs
"""
from typing import List, Dict, Any, Optional
import os
import json
import time
import importlib
import inspect

import dspy

from ..models.signature import SignatureDefinition
from ..config import Config
from .state import AppState
from .samples import SampleManager
from ..utils.metrics import judge_metric

class Optimizer:
    """Handles optimization process for the language model"""
    
    def __init__(self, app_state: AppState, sample_manager: SampleManager):
        self.app_state = app_state
        self.sample_manager = sample_manager
        self.running = False

    def _prepare_training_data(self, samples: List[Dict], signature: SignatureDefinition) -> List[dspy.Example]:
        """Prepare DSPy examples from loaded samples for a specific signature
        
        Args:
            samples (List[Dict]): List of samples to prepare
            signature (SignatureDefinition): Signature definition to use
            
        Returns:
            List[dspy.Example]: List of prepared examples
        """
        examples = []
        
        for sample in samples:
            # Validate that all required fields are present
            if not all(field in sample for field in signature.input_fields + signature.output_fields):
                continue
                
            # Create example with all fields
            example_data = {}
            for field in signature.input_fields + signature.output_fields:
                example_data[field] = sample.get(field, "")
                
            # Create example with input fields
            example = dspy.Example(**example_data).with_inputs(*signature.input_fields)
            examples.append(example)
        
        return examples

    def _create_signature_class(self, signature: SignatureDefinition) -> type:
        """Create a dynamic signature class from a signature definition
        
        Args:
            signature (SignatureDefinition): The signature definition
            
        Returns:
            type: The signature class
        """
        # Create a module to execute the signature class definition
        module_name = f"dynamic_signature_{int(time.time())}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        module.dspy = dspy
        
        # Execute the signature class definition in the module's context
        exec(signature.signature_class_def, module.__dict__)
        
        # Get the class from the module
        return getattr(module, signature.name)

    def run_optimization(self, model_name: str, signature_name: Optional[str] = None) -> None:
        """Run the optimization process with thread safety
        
        Args:
            model_name (str): Name of the model to use
            signature_name (str, optional): Name of the signature to optimize.
                                         If None, uses the current signature.
        """
        if self.running:
            return

        self.running = True
        try:
            # Get the signature to optimize
            sig_name = signature_name or self.app_state.current_signature_name
            if not sig_name:
                raise ValueError("No signature selected")
                
            signature = self.app_state.get_signature(sig_name)
            if not signature:
                raise ValueError(f"Signature {sig_name} not found")
            
            # Initialize the LM
            thread_lm = dspy.LM(model_name)
            if not thread_lm:
                return
            
            # Load samples for the signature
            samples = self.sample_manager.load_samples(sig_name)
            training_data = self._prepare_training_data(samples, signature)
            
            if not training_data:
                raise ValueError(f"No valid training data found for signature {sig_name}")

            # Create the signature class
            signature_class = self._create_signature_class(signature)

            # Base task can either be a new one or loaded from the current program
            current_program = self.app_state.current_program_id
            current_program_sig = self.app_state.programs.get(current_program, {}).get("signature_name")
            
            # Check if current program matches the signature we're optimizing
            program_path = Config.PROGRAM_DIR / current_program / "program.pkl"
            if (current_program and 
                current_program_sig == sig_name and 
                program_path.exists()):
                # Load the current program as a base
                print(f"Loading base program from {current_program}")
                try:
                    base_task = dspy.load(str(Config.PROGRAM_DIR / current_program))
                    print(f"Successfully loaded base task: {type(base_task)}")
                    
                    # Use the same optimization technique
                    optimized_task = dspy.MIPROv2(
                        metric=self._optimization_metric,
                        auto="light"
                    ).compile(base_task, trainset=training_data, max_bootstrapped_demos=0, requires_permission_to_run=False)
                except Exception as e:
                    print(f"Failed to load base task, creating new one: {e}")
                    # Fall back to creating a new task
                    task = dspy.ChainOfThought(signature_class)
                    optimized_task = dspy.MIPROv2(
                        metric=self._optimization_metric,
                        auto="light"
                    ).compile(task, trainset=training_data, max_bootstrapped_demos=0, requires_permission_to_run=False)
            else:
                # Create a new task using the specified signature
                print(f"Creating new base task for signature {sig_name}")
                task = dspy.ChainOfThought(signature_class)
                optimized_task = dspy.MIPROv2(
                    metric=self._optimization_metric,
                    auto="light"
                ).compile(task, trainset=training_data, max_bootstrapped_demos=0, requires_permission_to_run=False)
            
            # Create a new program directory
            program_id = f"program_{int(time.time())}"
            program_dir = Config.PROGRAM_DIR / program_id
            
            # Create directory if it doesn't exist
            program_dir.mkdir(exist_ok=True)
            
            # Save program to the directory
            print(f"Saving optimized program to {program_dir}")
            optimized_task.save(str(program_dir), save_program=True)
            
            # Add additional metadata
            metadata_path = program_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                # Add additional metadata
                metadata["model"] = model_name
                metadata["created_at"] = time.time()
                metadata["task_name"] = signature.description
                metadata["base_program_id"] = self.app_state.current_program_id
                metadata["signature_name"] = sig_name
                
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            
            # Update current program in app state
            self.app_state.current_program_id = program_id
            self.app_state.load_available_programs()  # Refresh program list
        except Exception as e:
            print(f"Optimization error: {e}")
        finally:
            self.running = False

    def _optimization_metric(self, example, pred, trace=None) -> float:
        """Metric for optimization process"""
        score, _ = judge_metric(example, pred)
        print("\n")
        print("score:")
        print(score)
        print("\n")
        return score