from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, abort
from threading import Thread
import os
import json
import dspy
from .samples import SampleManager
from .optimization import Optimizer
from .evaluation import Evaluator
from .state import AppState

def create_routes(app_state: AppState, sample_manager: SampleManager, 
                 optimizer: Optimizer, evaluator: Evaluator):
    """Create Flask routes blueprint"""
    bp = Blueprint('main', __name__)
    
    @bp.route('/select_program/<program_id>')
    def select_program(program_id):
        """Select a program to use"""
        if app_state.set_current_program(program_id):
            print(f"Selected program: {program_id}")
        return redirect(url_for('main.index'))
    
    @bp.route('/api/programs')
    def get_programs():
        """Get the list of available programs"""
        programs = []
        for program_id, metadata in app_state.programs.items():
            program_data = {
                "id": program_id,
                "model": metadata.get("model", "unknown"),
                "created_at": metadata.get("created_at", 0),
                "task_name": metadata.get("task_name", "unknown"),
                "is_current": program_id == app_state.current_program_id
            }
            
            # Try to load the program to get the instructions
            if program_id == app_state.current_program_id and app_state.current_program_id:
                try:
                    program_path = f"./programs/{program_id}/"
                    program = dspy.load(program_path)
                    if hasattr(program, 'predict') and hasattr(program.predict, 'signature'):
                        program_data["instructions"] = program.predict.signature.instructions
                except Exception as e:
                    print(f"Failed to load program instructions: {e}")
            
            programs.append(program_data)
        
        # Sort by creation time (newest first)
        programs.sort(key=lambda x: x["created_at"], reverse=True)
        
        return jsonify({"programs": programs})
    
    @bp.route('/')
    def index():
        samples = sample_manager.load_samples()
        return render_template('index.html', 
                            samples=samples, 
                            optimization_running=optimizer.running,
                            evaluation_results=app_state.evaluation_results,
                            models=app_state.AVAILABLE_MODELS,
                            current_model=app_state.current_model,
                            app_state=app_state)

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
                samples[sample_id]['english'] = request.form.get('english', '')
                samples[sample_id]['pln_types'] = request.form.get('pln_types', '')
                samples[sample_id]['pln_statements'] = request.form.get('pln_statements', '')
                samples[sample_id]['pln_questions'] = request.form.get('pln_questions', '')
                sample_manager.save_samples(samples)
                return redirect(url_for('main.view_sample', sample_id=sample_id))
        
        if 0 <= sample_id < len(samples):
            return render_template('edit_sample.html', sample=samples[sample_id], sample_id=sample_id)
        return redirect(url_for('main.view_samples'))

    @bp.route('/add_sample', methods=['GET', 'POST'])
    def add_sample():
        """Add a new sample."""
        if request.method == 'POST':
            # Create a new sample
            new_sample = {
                "english": request.form.get('english', ''),
                "pln_types": request.form.get('pln_types', ''),
                "pln_statements": request.form.get('pln_statements', ''),
                "pln_questions": request.form.get('pln_questions', '')
            }
            
            samples = sample_manager.load_samples()
            samples.append(new_sample)
            sample_manager.save_samples(samples)
            return redirect(url_for('main.view_samples'))
        
        # Create an empty sample template
        empty_sample = sample_manager.create_empty_sample()
        
        return render_template('add_sample.html', 
                              models=app_state.AVAILABLE_MODELS, 
                              current_model=app_state.current_model,
                              sample=empty_sample)

    @bp.route('/generate_sample', methods=['POST'])
    def generate_sample():
        """Generate a sample using the LLM."""
        # Get the input and model
        model_name = request.form.get('model', app_state.current_model)
        english_input = request.form.get('english', '')
        
        # Get a model instance without configuring DSPy
        sample_lm = dspy.LM(model_name)
        if sample_lm is None:
            return jsonify({
                "error": f"Failed to initialize model {model_name}",
                "english": english_input,
                "pln_types": "",
                "pln_statements": "",
                "pln_questions": ""
            })
        
        try:
            # Create a PLN signature for sample generation
            class PLNSampleGen(dspy.Signature):
                """Convert English text to Programming Language for Thought (PLN)"""
                english = dspy.InputField(desc="English text to convert to PLN")
                pln_types = dspy.OutputField(desc="PLN type definitions")
                pln_statements = dspy.OutputField(desc="PLN statements")
                pln_questions = dspy.OutputField(desc="PLN questions")
            
            # Create a basic example generator with the specific LM instance
            gen_example = dspy.ChainOfThought(PLNSampleGen)
            
            # Generate the sample using the specific LM instance
            with dspy.context(lm=sample_lm):
                pred = gen_example(english=english_input)
            
            # Prepare response
            response = {
                "english": english_input,
                "pln_types": getattr(pred, "pln_types", ""),
                "pln_statements": getattr(pred, "pln_statements", ""),
                "pln_questions": getattr(pred, "pln_questions", "")
            }
            
            # Return the generated sample
            return jsonify(response)
        except Exception as e:
            return jsonify({
                "error": str(e),
                "english": english_input,
                "pln_types": "",
                "pln_statements": "",
                "pln_questions": ""
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
        return render_template('evaluation_results.html', app_state=app_state)
                              
    @bp.route('/api/evaluation_results')
    def get_evaluation_results():
        """Get the current evaluation results as JSON."""
        return jsonify(app_state.evaluation_results)

    @bp.route('/edit_program_instructions', methods=['GET', 'POST'])
    def edit_program_instructions():
        """Edit the instructions of the current program"""
        program_id = app_state.current_program_id
        if not program_id:
            return redirect(url_for('main.index'))
        
        if request.method == 'POST':
            new_instructions = request.form.get('instructions', '')
            program_path = f"./programs/{program_id}/"
            
            try:
                # Load the program
                program = dspy.load(program_path)
                if hasattr(program, 'predict') and hasattr(program.predict, 'signature'):
                    # Update the instructions
                    program.predict.signature.instructions = new_instructions
                    # Save the program
                    program.save(program_path, save_program=True)
                    return redirect(url_for('main.index'))
                else:
                    flash("Program does not have the expected structure.")
            except Exception as e:
                flash(f"Failed to update program instructions: {e}")
                print(f"Error updating program instructions: {e}")
            
            return redirect(url_for('main.edit_program_instructions'))
        
        # GET request - show the form
        try:
            program_path = f"./programs/{program_id}/"
            program = dspy.load(program_path)
            instructions = ""
            if hasattr(program, 'predict') and hasattr(program.predict, 'signature'):
                instructions = program.predict.signature.instructions
            
            return render_template('edit_program_instructions.html', 
                                  program_id=program_id, 
                                  instructions=instructions)
        except Exception as e:
            flash(f"Failed to load program: {e}")
            print(f"Error loading program: {e}")
            return redirect(url_for('main.index'))
    
    return bp
