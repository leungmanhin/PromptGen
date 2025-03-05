from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from threading import Thread
import os
import json
import dspy
from .samples import SampleManager
from .optimization import Optimizer
from .evaluation import Evaluator
from .state import AppState
from .task_definition import TaskDefinition

def create_routes(app_state: AppState, sample_manager: SampleManager, 
                 optimizer: Optimizer, evaluator: Evaluator):
    """Create Flask routes blueprint"""
    bp = Blueprint('main', __name__)
    
    @bp.route('/')
    def index():
        samples = sample_manager.load_samples()
        return render_template('index.html', 
                            samples=samples, 
                            optimization_running=optimizer.running,
                            evaluation_results=app_state.evaluation_results,
                            models=app_state.AVAILABLE_MODELS,
                            current_model=app_state.current_model,
                            task_definition=app_state.task_definition)

    @bp.route('/samples')
    def view_samples():
        samples = sample_manager.load_samples()
        return render_template('samples.html', samples=samples)

    @bp.route('/sample/<int:sample_id>')
    def view_sample(sample_id):
        samples = sample_manager.load_samples()
        if 0 <= sample_id < len(samples):
            return render_template('sample.html', 
                                 sample=samples[sample_id], 
                                 sample_id=sample_id)
        return redirect(url_for('main.view_samples'))

    @bp.route('/sample/<int:sample_id>/edit', methods=['GET', 'POST'])
    def edit_sample(sample_id):
        """Edit a specific sample."""
        samples = sample_manager.load_samples()
        
        if request.method == 'POST':
            if 0 <= sample_id < len(samples):
                samples[sample_id]['input'] = request.form.get('input', '')
                samples[sample_id]['types'] = request.form.get('types', '')
                samples[sample_id]['statements'] = request.form.get('statements', '')
                samples[sample_id]['questions'] = request.form.get('questions', '')
                sample_manager.save_samples(samples)
                return redirect(url_for('main.view_sample', sample_id=sample_id))
        
        if 0 <= sample_id < len(samples):
            return render_template('edit_sample.html', sample=samples[sample_id], sample_id=sample_id)
        return redirect(url_for('main.view_samples'))

    @bp.route('/add_sample', methods=['GET', 'POST'])
    def add_sample():
        """Add a new sample."""
        if request.method == 'POST':
            # Create a new sample based on task definition
            task_def = app_state.task_definition
            new_sample = {}
            
            # Add input fields from form
            for field in task_def.input_fields:
                field_name = field["name"]
                new_sample[field_name] = request.form.get(field_name, '')
            
            # Add output fields from form
            for field in task_def.output_fields:
                field_name = field["name"]
                new_sample[field_name] = request.form.get(field_name, '')
            
            samples = sample_manager.load_samples()
            samples.append(new_sample)
            sample_manager.save_samples(samples)
            return redirect(url_for('main.view_samples'))
        
        # Create an empty sample template based on task definition
        empty_sample = sample_manager.create_empty_sample()
        
        return render_template('add_sample_dynamic.html', 
                              models=app_state.AVAILABLE_MODELS, 
                              current_model=app_state.current_model,
                              task_definition=app_state.task_definition,
                              sample=empty_sample)

    @bp.route('/generate_sample', methods=['POST'])
    def generate_sample():
        """Generate a sample using the LLM."""
        # Get the input and model
        task_def = app_state.task_definition
        model_name = request.form.get('model', app_state.current_model)
        
        # Get a model instance without configuring DSPy
        sample_lm = dspy.LM(model_name)
        if sample_lm is None:
            error_response = {"error": f"Failed to initialize model {model_name}"}
            # Add empty fields based on task definition
            for field in task_def.input_fields + task_def.output_fields:
                error_response[field["name"]] = ""
            return jsonify(error_response)
        
        try:
            # Create a dynamic signature for sample generation
            input_fields_str = ", ".join([f"{f['name']}: str" for f in task_def.input_fields])
            output_fields_str = ", ".join([f"{f['name']}: str" for f in task_def.output_fields])
            signature_str = f"task: str, {input_fields_str} -> {output_fields_str}"
            
            # Create a basic example generator with the specific LM instance
            gen_example = dspy.ChainOfThought(signature_str)
            
            # Get the task description
            task_description = task_def.description
            
            # Prepare input arguments
            gen_args = {"task": task_description}
            
            # Add input fields from form
            for field in task_def.input_fields:
                field_name = field["name"]
                gen_args[field_name] = request.form.get(field_name, '')
            
            # Generate the sample using the specific LM instance
            with dspy.context(lm=sample_lm):
                pred = gen_example(**gen_args)
            
            # Prepare response with all fields
            response = {}
            
            # Add input fields
            for field in task_def.input_fields:
                field_name = field["name"]
                response[field_name] = request.form.get(field_name, '')
            
            # Add output fields from prediction
            for field in task_def.output_fields:
                field_name = field["name"]
                response[field_name] = getattr(pred, field_name, "")
            
            # Return the generated sample
            return jsonify(response)
        except Exception as e:
            error_response = {"error": str(e)}
            # Add empty fields based on task definition
            for field in task_def.input_fields:
                field_name = field["name"]
                error_response[field_name] = request.form.get(field_name, '')
            for field in task_def.output_fields:
                error_response[field["name"]] = ""
            return jsonify(error_response)
    
    @bp.route('/optimize', methods=['POST'])
    def optimize():
        if not optimizer.running:
            model_name = request.form.get('model', app_state.current_model)
            # Update the current model in app state
            app_state.current_model = model_name
            Thread(target=optimizer.run_optimization, args=(model_name,)).start()
            return jsonify({"status": "started"})
        return jsonify({"status": "already_running"})

    @bp.route('/optimization_status')
    def optimization_status():
        return jsonify({"running": optimizer.running})

    @bp.route('/evaluate', methods=['POST'])
    def evaluate():
        """Run the evaluation on the optimized model."""
        # Get the models to use for evaluation and similarity
        model_name = request.form.get('model', app_state.current_model)
        similarity_model_name = request.form.get('similarity_model')
        
        # Update the current model in app state
        app_state.current_model = model_name
        
        # Run evaluation using the Evaluator class
        evaluation_results = evaluator.run_evaluation(
            model_name=model_name,
            similarity_model_name=similarity_model_name
        )
        
        # Store the evaluation results in app state
        app_state.evaluation_results = evaluation_results
        
        # Return the evaluation results
        return jsonify(evaluation_results)

    @bp.route('/evaluation_results')
    def view_evaluation_results():
        """View the evaluation results page."""
        return render_template('evaluation_results.html', 
                              evaluation_results=app_state.evaluation_results)
                              
    @bp.route('/api/evaluation_results')
    def get_evaluation_results():
        """Get the current evaluation results as JSON."""
        return jsonify(app_state.evaluation_results)

    @bp.route('/task_definition', methods=['GET', 'POST'])
    def task_definition():
        """View and edit task definition"""
        if request.method == 'POST':
            # Extract task definition from form
            name = request.form.get('name', 'CustomTask')
            description = request.form.get('description', '')
            
            # Process input fields
            input_field_names = request.form.getlist('input_field_name[]')
            input_field_descs = request.form.getlist('input_field_desc[]')
            input_fields = [
                {"name": name, "desc": desc}
                for name, desc in zip(input_field_names, input_field_descs)
            ]
            
            # Process output fields
            output_field_names = request.form.getlist('output_field_name[]')
            output_field_descs = request.form.getlist('output_field_desc[]')
            output_fields = [
                {"name": name, "desc": desc}
                for name, desc in zip(output_field_names, output_field_descs)
            ]
            
            # Create and save task definition
            task_def = TaskDefinition(
                name=name,
                description=description,
                input_fields=input_fields,
                output_fields=output_fields
            )
            task_def.save()
            
            # Update app state
            app_state.task_definition = task_def
            
            return redirect(url_for('main.index'))
        
        # GET request - show the form
        return render_template('task_definition.html', task=app_state.task_definition)
    
    return bp
