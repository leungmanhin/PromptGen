"""
Routes for managing programs
"""
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
import os
import json
import dspy
from threading import Thread

def create_program_routes(app_state, optimizer):
    bp = Blueprint('programs', __name__)
    
    @bp.route('/')
    def view_programs():
        """View all programs"""
        return render_template('programs.html', 
                             programs=app_state.programs,
                             current_program_id=app_state.current_program_id)
    
    @bp.route('/select/<program_id>')
    def select_program(program_id):
        """Select a program to use"""
        if app_state.set_current_program(program_id):
            flash(f"Selected program: {program_id}")
            # Set the current signature to match this program
            program_metadata = app_state.programs.get(program_id, {})
            signature_name = program_metadata.get("signature_name")
            if signature_name:
                app_state.set_current_signature(signature_name)
        return redirect(url_for('main.index'))
    
    @bp.route('/create', methods=['POST'])
    def create_program():
        """Create a new empty program"""
        model_name = request.form.get('model', app_state.current_model)
        base_program_id = request.form.get('base_program_id')
        signature_name = request.form.get('signature_name', app_state.current_signature_name)
        
        try:
            # Check if we have a valid signature selected
            if not signature_name:
                flash("No signature selected. Please select a signature first.")
                return redirect(url_for('signatures.view_signatures'))
                
            # Create a new program
            program_id = app_state.create_new_program(
                model_name=model_name, 
                base_program_id=base_program_id,
                signature_name=signature_name
            )
            flash(f"Created new program: {program_id}")
            return redirect(url_for('main.index'))
        except Exception as e:
            flash(f"Failed to create program: {e}")
            print(f"Error creating program: {e}")
            return redirect(url_for('main.index'))
    
    @bp.route('/delete/<program_id>', methods=['POST'])
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
    
    @bp.route('/edit_instructions', methods=['GET', 'POST'])
    def edit_program_instructions():
        """Edit the instructions of the current program"""
        program_id = app_state.current_program_id
        if not program_id:
            flash("No program selected")
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
                    flash("Program instructions updated successfully")
                    return redirect(url_for('main.index'))
                else:
                    flash("Program does not have the expected structure.")
            except Exception as e:
                flash(f"Failed to update program instructions: {e}")
                print(f"Error updating program instructions: {e}")
            
            return redirect(url_for('programs.edit_program_instructions'))
        
        # GET request - show the form
        try:
            from pathlib import Path
            from ..config import Config
            
            program_path = Config.PROGRAM_DIR / program_id
            program = dspy.load(str(program_path))
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
    
    @bp.route('/optimize', methods=['POST'])
    def optimize():
        """Start the optimization process"""
        # Check if a signature is selected
        signature_name = request.form.get('signature_name', app_state.current_signature_name)
        if not signature_name:
            return jsonify({
                "status": "error",
                "message": "No signature selected. Please select a signature first."
            })
        
        # Check if the signature exists
        if not app_state.get_signature(signature_name):
            return jsonify({
                "status": "error",
                "message": f"Signature '{signature_name}' not found."
            })
        
        # Check if a valid program is selected for this signature
        current_program = app_state.current_program_id
        current_program_sig = app_state.programs.get(current_program, {}).get("signature_name")
        if not current_program or current_program_sig != signature_name:
            return jsonify({
                "status": "error",
                "message": "No valid program selected. Please create or select a program for this signature first."
            })
        
        # Start optimization in a separate thread
        if not optimizer.running:
            model_name = request.form.get('model', app_state.current_model)
            # Update the current model in app state
            app_state.current_model = model_name
            Thread(target=optimizer.run_optimization, args=(model_name, signature_name)).start()
            return jsonify({"status": "started"})
        return jsonify({"status": "already_running"})
            
    return bp