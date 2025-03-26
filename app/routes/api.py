"""
API routes for the application
"""
import dspy
from flask import Blueprint, request, jsonify, redirect, url_for, flash

def create_api_routes(app_state, sample_manager, optimizer, evaluator):
    bp = Blueprint('api', __name__)
    
    @bp.route('/programs')
    def get_programs():
        """Get the list of available programs"""
        programs = []
        for program_id, metadata in app_state.programs.items():
            program_data = {
                "id": program_id,
                "model": metadata.get("model", "unknown"),
                "created_at": metadata.get("created_at", 0),
                "task_name": metadata.get("task_name", "unknown"),
                "is_current": program_id == app_state.current_program_id,
                "signature_name": metadata.get("signature_name", "")
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
    
    @bp.route('/signatures')
    def get_signatures():
        """Get the list of available signatures"""
        signatures = []
        for name, signature in app_state.signatures.items():
            signature_data = {
                "name": name,
                "description": signature.description,
                "input_fields": signature.input_fields,
                "output_fields": signature.output_fields,
                "is_current": name == app_state.current_signature_name
            }
            signatures.append(signature_data)
        
        return jsonify({"signatures": signatures})
    
    @bp.route('/generate_sample', methods=['POST'])
    def generate_sample():
        """Generate a sample using the LLM."""
        # Get the input and model
        model_name = request.form.get('model', app_state.current_model)
        
        # Get input fields based on current signature
        signature_name = app_state.current_signature_name
        if not signature_name:
            return jsonify({
                "error": "No signature selected. Please select a signature first."
            })
            
        signature = app_state.get_signature(signature_name)
        if not signature:
            return jsonify({
                "error": f"Signature '{signature_name}' not found."
            })
        
        # Extract input fields from request
        input_data = {}
        for field in signature.input_fields:
            input_data[field] = request.form.get(field, '')
        
        # Check if a program is selected
        if not app_state.current_program_id:
            response = {
                "error": "No program selected. Please create and select a program first."
            }
            # Add input fields to response
            for field in signature.input_fields:
                response[field] = input_data.get(field, '')
            # Add empty output fields
            for field in signature.output_fields:
                response[field] = ""
            return jsonify(response)
            
        # Check if the program exists
        from ..config import Config
        program_path = Config.PROGRAM_DIR / app_state.current_program_id / "program.pkl"
        if not program_path.exists():
            response = {
                "error": f"Selected program {app_state.current_program_id} not found or corrupted."
            }
            # Add input fields to response
            for field in signature.input_fields:
                response[field] = input_data.get(field, '')
            # Add empty output fields
            for field in signature.output_fields:
                response[field] = ""
            return jsonify(response)
            
        # Get a model instance without configuring DSPy
        from ..utils import get_lm
        sample_lm = get_lm(model_name)
        if sample_lm is None:
            response = {
                "error": f"Failed to initialize model {model_name}"
            }
            # Add input fields to response
            for field in signature.input_fields:
                response[field] = input_data.get(field, '')
            # Add empty output fields
            for field in signature.output_fields:
                response[field] = ""
            return jsonify(response)
        
        try:
            # Load the selected program for generation
            program = dspy.load(str(Config.PROGRAM_DIR / app_state.current_program_id))
            
            # Generate the sample using the loaded program and specific LM instance
            with dspy.context(lm=sample_lm):
                pred = program(**input_data)
            
            # Prepare response with input fields
            response = {}
            for field in signature.input_fields:
                response[field] = input_data.get(field, '')
            
            # Add output fields from prediction
            for field in signature.output_fields:
                response[field] = getattr(pred, field, "")
            
            # Return the generated sample
            return jsonify(response)
        except Exception as e:
            response = {
                "error": f"Error using selected program: {str(e)}"
            }
            # Add input fields to response
            for field in signature.input_fields:
                response[field] = input_data.get(field, '')
            # Add empty output fields
            for field in signature.output_fields:
                response[field] = ""
            return jsonify(response)
    
    @bp.route('/optimization_status')
    def optimization_status():
        """Check optimization status"""
        return jsonify({"running": optimizer.running})
    
    @bp.route('/evaluate', methods=['POST'])
    def evaluate():
        """Evaluate the current program against samples"""
        model_name = request.form.get('model', app_state.current_model)
        similarity_model = request.form.get('similarity_model')
        
        # If similarity model is empty, use None (defaults to same as evaluation model)
        if similarity_model == '':
            similarity_model = None
            
        # Run evaluation
        results = evaluator.run_evaluation(model_name, similarity_model)
        
        # Store results in app state
        if results.get('status') == 'success':
            app_state.evaluation_results = results
            
        return jsonify(results)
    
    @bp.route('/generate_sample_from_evaluation', methods=['GET', 'POST'])
    def generate_sample_from_evaluation():
        """Generate a new sample using DSPy based on evaluation results and redirect to add_sample page"""
        # Get the sample ID and model name
        sample_id = request.args.get('sample_id') or request.form.get('sample_id')
        model_name = request.args.get('model') or request.form.get('model', app_state.current_model)
        
        # Verify we have evaluation results
        if not app_state.evaluation_results or not app_state.evaluation_results.get('results'):
            flash("No evaluation results available. Please run an evaluation first.")
            return redirect(url_for('main.index'))
        
        # Find the result for the specified sample ID
        sample_id = int(sample_id) if sample_id else None
        eval_result = None
        
        for result in app_state.evaluation_results.get('results', []):
            if result.get('sample_id') == sample_id:
                eval_result = result
                break
                
        if not eval_result:
            flash(f"Sample #{sample_id} not found in evaluation results.")
            return redirect(url_for('main.evaluation_results'))
            
        # Get the program instructions
        program_instructions = ""
        if app_state.current_program_id:
            try:
                from ..config import Config
                program_path = Config.PROGRAM_DIR / app_state.current_program_id
                program = dspy.load(str(program_path))
                if hasattr(program, 'predict') and hasattr(program.predict, 'signature'):
                    program_instructions = program.predict.signature.instructions
            except Exception as e:
                print(f"Failed to load program instructions: {e}")
        
        # Generate a new sample
        new_sample = sample_manager.generate_new_sample_from_evaluation(
            eval_result=eval_result,
            model_name=model_name,
            program_instructions=program_instructions
        )
        
        # Store the new sample in the session for the add_sample page
        from flask import session
        session['generated_sample'] = new_sample
        session['from_evaluation'] = True
        
        # Redirect to the add_sample page
        return redirect(url_for('samples.add_sample'))
        
    return bp
