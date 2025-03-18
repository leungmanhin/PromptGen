"""
Routes for managing signatures
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash
from ..models.signature import SignatureDefinition

def create_signature_routes(app_state, sample_manager):
    bp = Blueprint('signatures', __name__)
    
    @bp.route('/')
    def view_signatures():
        """View available signatures"""
        return render_template('signatures.html', 
                             signatures=app_state.signatures,
                             current_signature=app_state.get_current_signature())
    
    @bp.route('/select/<signature_name>')
    def select_signature(signature_name):
        """Select a signature to use"""
        if app_state.set_current_signature(signature_name):
            flash(f"Selected signature: {signature_name}")
            # Get programs for this signature
            programs = app_state.get_programs_for_signature(signature_name)
            if programs:
                # Set the current program to the most recent one for this signature
                sorted_programs = sorted(programs.items(), 
                                       key=lambda x: x[1].get("created_at", 0), 
                                       reverse=True)
                app_state.set_current_program(sorted_programs[0][0])
            else:
                # Reset current program if no compatible programs exist
                app_state.current_program_id = None
                flash("No compatible programs found for this signature. You'll need to create a new program.")
        return redirect(url_for('main.index'))
        
    @bp.route('/add', methods=['GET', 'POST'])
    def add_signature():
        """Add a new signature"""
        if request.method == 'POST':
            name = request.form.get('name', '')
            description = request.form.get('description', '')
            signature_class_def = request.form.get('signature_class_def', '')
            input_fields = request.form.get('input_fields', '').split(',')
            output_fields = request.form.get('output_fields', '').split(',')
            
            # Clean input and output fields
            input_fields = [field.strip() for field in input_fields if field.strip()]
            output_fields = [field.strip() for field in output_fields if field.strip()]
            
            # Parse field processors
            field_processors = {}
            field_processors_str = request.form.get('field_processors', '')
            if field_processors_str.strip():
                for pair in field_processors_str.split(','):
                    if ':' in pair:
                        field, processor = pair.split(':', 1)
                        field_processors[field.strip()] = processor.strip()
            
            # Create signature
            new_signature = SignatureDefinition(
                name=name,
                description=description,
                signature_class_def=signature_class_def,
                input_fields=input_fields,
                output_fields=output_fields,
                field_processors=field_processors
            )
            
            # Add to app state
            if app_state.add_signature(new_signature):
                # Set as current signature
                app_state.set_current_signature(name)
                flash(f"Created new signature: {name}")
                return redirect(url_for('signatures.view_signatures'))
            else:
                flash(f"Signature with name '{name}' already exists")
        
        # GET request - show add form
        return render_template('add_signature.html')
    
    @bp.route('/edit/<signature_name>', methods=['GET', 'POST'])
    def edit_signature(signature_name):
        """Edit an existing signature"""
        signature = app_state.get_signature(signature_name)
        if not signature:
            flash(f"Signature '{signature_name}' not found")
            return redirect(url_for('signatures.view_signatures'))
            
        if request.method == 'POST':
            description = request.form.get('description', '')
            signature_class_def = request.form.get('signature_class_def', '')
            input_fields = request.form.get('input_fields', '').split(',')
            output_fields = request.form.get('output_fields', '').split(',')
            
            # Clean input and output fields
            input_fields = [field.strip() for field in input_fields if field.strip()]
            output_fields = [field.strip() for field in output_fields if field.strip()]
            
            # Parse field processors
            field_processors = {}
            field_processors_str = request.form.get('field_processors', '')
            if field_processors_str.strip():
                for pair in field_processors_str.split(','):
                    if ':' in pair:
                        field, processor = pair.split(':', 1)
                        field_processors[field.strip()] = processor.strip()
            
            # Update signature
            signature.description = description
            signature.signature_class_def = signature_class_def
            signature.input_fields = input_fields
            signature.output_fields = output_fields
            signature.field_processors = field_processors
            
            # Save signature
            app_state._save_signature(signature)
            flash(f"Updated signature: {signature_name}")
            return redirect(url_for('signatures.view_signatures'))
            
        # GET request - show edit form
        return render_template('edit_signature.html', signature=signature)
        
    return bp