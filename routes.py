from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from threading import Thread
import os
import json
import dspy
from .models import ModelManager
from .samples import SampleManager
from .optimization import Optimizer
from .evaluation import Evaluator
from .state import AppState

def create_routes(app_state: AppState, model_manager: ModelManager, sample_manager: SampleManager, 
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
                            current_model=app_state.current_model)

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
            new_sample = {
                'input': request.form.get('input', ''),
                'types': request.form.get('types', ''),
                'statements': request.form.get('statements', ''),
                'questions': request.form.get('questions', '')
            }
            samples = sample_manager.load_samples()
            samples.append(new_sample)
            sample_manager.save_samples(samples)
            return redirect(url_for('main.view_samples'))
        
        return render_template('add_sample.html', 
                              models=app_state.AVAILABLE_MODELS, 
                              current_model=app_state.current_model)

    @bp.route('/generate_sample', methods=['POST'])
    def generate_sample():
        """Generate a sample using the LLM."""
        # Get the input and model
        input_text = request.form.get('input', '')
        model_name = request.form.get('model', app_state.current_model)
        
        # Get a model instance without configuring DSPy
        sample_lm = model_manager.get_lm_instance(model_name)
        if sample_lm is None:
            return jsonify({
                "error": f"Failed to initialize model {model_name}",
                "input": input_text,
                "types": "",
                "statements": "",
                "questions": ""
            })
        
        try:
            # Create a basic example generator with the specific LM instance
            gen_example = dspy.ChainOfThought('task: str, input: str -> types: str, statements: str, questions: str')
            
            # Get the task from task.json if it exists
            task = get_task_description()
            
            # Generate the sample using the specific LM instance
            with dspy.context(lm=sample_lm):
                pred = gen_example(task=task, input=input_text)
            
            # Return the generated sample
            return jsonify({
                "input": input_text,
                "types": pred.types,
                "statements": pred.statements,
                "questions": pred.questions
            })
        except Exception as e:
            return jsonify({
                "error": str(e),
                "input": input_text,
                "types": "",
                "statements": "",
                "questions": ""
            })
    
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

    return bp

def get_task_description():
    """Get task description from file or return default"""
    try:
        with open("task.json", "r") as f:
            return json.load(f)["self"]["extended_signature"]["instructions"]
    except Exception:
        return "Convert English to Logic (MeTTa PLN Light)"
