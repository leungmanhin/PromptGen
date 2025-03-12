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
                samples[sample_id]['pln_query'] = request.form.get('pln_query', '')
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
                "pln_query": request.form.get('pln_query', '')
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
                              sample=empty_sample,
                              from_evaluation=False)
                              
    @bp.route('/add_sample_from_evaluation/<int:sample_id>', methods=['GET', 'POST'])
    def add_sample_from_evaluation(sample_id):
        """Add a new sample based on evaluation results."""
        if request.method == 'POST':
            # Create a new sample
            new_sample = {
                "english": request.form.get('english', ''),
                "pln_types": request.form.get('pln_types', ''),
                "pln_statements": request.form.get('pln_statements', ''),
                "pln_query": request.form.get('pln_query', '')
            }
            
            samples = sample_manager.load_samples()
            samples.append(new_sample)
            sample_manager.save_samples(samples)
            return redirect(url_for('main.view_samples'))
        
        # Get data from evaluation results
        if not app_state.evaluation_results or not app_state.evaluation_results.get("results"):
            return redirect(url_for('main.view_evaluation_results'))
            
        # Find the evaluation result for the specified sample ID
        eval_result = None
        for result in app_state.evaluation_results.get("results", []):
            if result.get("sample_id") == sample_id:
                eval_result = result
                break
                
        if not eval_result:
            flash("Evaluation result not found.")
            return redirect(url_for('main.view_evaluation_results'))
            
        # Get the model to use for generating the sample
        model_name = app_state.current_model
        
        # Try to get program instructions
        program_instructions = ""
        if app_state.current_program_id:
            try:
                program_path = f"./programs/{app_state.current_program_id}/"
                program = dspy.load(program_path)
                if hasattr(program, 'predict') and hasattr(program.predict, 'signature'):
                    program_instructions = program.predict.signature.instructions
            except Exception as e:
                print(f"Failed to load program instructions: {e}")
        
        try:
            # Generate a new sample using the LLM
            new_sample = sample_manager.generate_new_sample_from_evaluation(
                eval_result=eval_result,
                model_name=model_name,
                program_instructions=program_instructions
            )
        except Exception as e:
            flash(f"Error generating new sample: {str(e)}")
            # Fall back to using the evaluation data directly
            new_sample = {
                "english": eval_result.get("input_english", ""),
                "pln_types": eval_result.get("predicted_pln_types", ""),
                "pln_statements": eval_result.get("predicted_pln_statements", ""),
                "pln_query": eval_result.get("predicted_pln_query", "")
            }
        
        return render_template('add_sample.html', 
                              models=app_state.AVAILABLE_MODELS, 
                              current_model=app_state.current_model,
                              sample=new_sample,
                              from_evaluation=True)

    @bp.route('/generate_sample', methods=['POST'])
    def generate_sample():
        """Generate a sample using the LLM."""
        # Get the input and model
        model_name = request.form.get('model', app_state.current_model)
        english_input = request.form.get('english', '')
        
        # Check if a program is selected
        if not app_state.current_program_id:
            return jsonify({
                "error": "No program selected. Please create and select a program first.",
                "english": english_input,
                "pln_types": "",
                "pln_statements": "",
                "pln_query": ""
            })
            
        # Check if the program exists
        if not os.path.exists(f"./programs/{app_state.current_program_id}/program.pkl"):
            return jsonify({
                "error": f"Selected program {app_state.current_program_id} not found or corrupted.",
                "english": english_input,
                "pln_types": "",
                "pln_statements": "",
                "pln_query": ""
            })
            
        # Get a model instance without configuring DSPy
        sample_lm = dspy.LM(model_name)
        if sample_lm is None:
            return jsonify({
                "error": f"Failed to initialize model {model_name}",
                "english": english_input,
                "pln_types": "",
                "pln_statements": "",
                "pln_query": ""
            })
        
        try:
            # Load the selected program for generation
            print(f"Using selected program {app_state.current_program_id} for sample generation")
            program = dspy.load(f"./programs/{app_state.current_program_id}/")
            
            # Generate the sample using the loaded program and specific LM instance
            with dspy.context(lm=sample_lm):
                pred = program(english=english_input)
            
            # Prepare response
            response = {
                "english": english_input,
                "pln_types": getattr(pred, "pln_types", ""),
                "pln_statements": getattr(pred, "pln_statements", ""),
                "pln_query": getattr(pred, "pln_query", "")
            }
            
            # Return the generated sample
            return jsonify(response)
        except Exception as e:
            return jsonify({
                "error": f"Error using selected program: {str(e)}",
                "english": english_input,
                "pln_types": "",
                "pln_statements": "",
                "pln_query": ""
            })
    
    @bp.route('/optimize', methods=['POST'])
    def optimize():
        # Check if a program is selected
        if not app_state.current_program_id:
            return jsonify({
                "status": "error",
                "message": "No program selected. Please create and select a program first."
            })
            
        # Check if the program exists
        if not os.path.exists(f"./programs/{app_state.current_program_id}/program.pkl"):
            return jsonify({
                "status": "error",
                "message": f"Selected program {app_state.current_program_id} not found or corrupted."
            })
            
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
        
    @bp.route('/create_program', methods=['POST'])
    def create_program():
        """Create a new empty program"""
        model_name = request.form.get('model', app_state.current_model)
        base_program_id = request.form.get('base_program_id')
        
        try:
            # Create a new program
            program_id = app_state.create_new_program(model_name, base_program_id)
            flash(f"Created new program: {program_id}")
            return redirect(url_for('main.index'))
        except Exception as e:
            flash(f"Failed to create program: {e}")
            print(f"Error creating program: {e}")
            return redirect(url_for('main.index'))
    
    @bp.route('/delete_program/<program_id>', methods=['POST'])
    def delete_program(program_id):
        """Delete a program"""
        try:
            if app_state.delete_program(program_id):
                flash(f"Deleted program: {program_id}")
            else:
                flash(f"Failed to delete program: {program_id}")
        except Exception as e:
            flash(f"Error deleting program: {e}")
            print(f"Error deleting program {program_id}: {e}")
        
        return redirect(url_for('main.index'))

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
