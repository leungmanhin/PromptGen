"""
Routes for managing samples
"""
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash

def create_sample_routes(app_state, sample_manager):
    bp = Blueprint('samples', __name__)
    
    @bp.route('/')
    def view_samples():
        """View samples for the current signature"""
        signature_name = app_state.current_signature_name
        if not signature_name:
            flash("No signature selected. Please select a signature first.")
            return redirect(url_for('signatures.view_signatures'))
            
        signature = app_state.get_signature(signature_name)
        samples = sample_manager.load_samples(signature_name)
        return render_template('samples.html', 
                             samples=samples, 
                             signature=signature,
                             signature_name=signature_name)
                             
    @bp.route('/<signature_name>')
    def view_signature_samples(signature_name):
        """View samples for a specific signature"""
        signature = app_state.get_signature(signature_name)
        if not signature:
            flash(f"Signature '{signature_name}' not found")
            return redirect(url_for('signatures.view_signatures'))
            
        samples = sample_manager.load_samples(signature_name)
        return render_template('samples.html', 
                             samples=samples, 
                             signature=signature,
                             signature_name=signature_name)
                             
    @bp.route('/export/<signature_name>')
    def export_samples(signature_name):
        """Export samples for a specific signature as JSON"""
        signature = app_state.get_signature(signature_name)
        if not signature:
            flash(f"Signature '{signature_name}' not found")
            return redirect(url_for('signatures.view_signatures'))
            
        samples = sample_manager.load_samples(signature_name)
        if not samples:
            flash(f"No samples found for signature '{signature_name}'")
            return redirect(url_for('samples.view_signature_samples', signature_name=signature_name))
            
        # Create a response with the JSON data
        response = jsonify(samples)
        response.headers.set('Content-Disposition', f'attachment; filename={signature_name}_samples.json')
        response.headers.set('Content-Type', 'application/json')
        return response
        
    @bp.route('/import/<signature_name>', methods=['GET', 'POST'])
    def import_samples(signature_name):
        """Import samples for a specific signature"""
        signature = app_state.get_signature(signature_name)
        if not signature:
            flash(f"Signature '{signature_name}' not found")
            return redirect(url_for('signatures.view_signatures'))
            
        if request.method == 'POST':
            # Check if a file was uploaded
            if 'samples_file' not in request.files:
                flash('No file selected')
                return redirect(request.url)
                
            file = request.files['samples_file']
            
            # Check if the file name is empty
            if file.filename == '':
                flash('No file selected')
                return redirect(request.url)
                
            if file:
                try:
                    # Read the JSON file
                    import json
                    import_data = json.loads(file.read().decode('utf-8'))
                    
                    # Validate that it's a list
                    if not isinstance(import_data, list):
                        flash("Invalid file format. Expected a JSON array of samples.")
                        return redirect(request.url)
                    
                    # Validate each sample against the signature
                    valid_samples = []
                    for sample in import_data:
                        if sample_manager.validate_sample(sample, signature_name):
                            valid_samples.append(sample)
                    
                    # Check if any valid samples were found
                    if not valid_samples:
                        flash("No valid samples found in the import file.")
                        return redirect(request.url)
                    
                    # Get current samples
                    current_samples = sample_manager.load_samples(signature_name)
                    
                    # Add the imported samples
                    merge_strategy = request.form.get('merge_strategy', 'append')
                    if merge_strategy == 'replace':
                        # Replace all existing samples
                        sample_manager.save_samples(valid_samples, signature_name)
                        flash(f"Imported {len(valid_samples)} samples, replacing existing samples.")
                    else:
                        # Append to existing samples
                        current_samples.extend(valid_samples)
                        sample_manager.save_samples(current_samples, signature_name)
                        flash(f"Imported {len(valid_samples)} samples, appended to existing {len(current_samples) - len(valid_samples)} samples.")
                    
                    return redirect(url_for('samples.view_signature_samples', signature_name=signature_name))
                except json.JSONDecodeError:
                    flash("Invalid JSON file.")
                    return redirect(request.url)
                except Exception as e:
                    flash(f"Error importing samples: {e}")
                    return redirect(request.url)
            
        # GET request - show import form
        return render_template('import_samples.html', 
                             signature=signature,
                             signature_name=signature_name)
    
    @bp.route('/view/<int:sample_id>')
    def view_sample(sample_id):
        """View a sample for the current signature"""
        signature_name = app_state.current_signature_name
        if not signature_name:
            flash("No signature selected. Please select a signature first.")
            return redirect(url_for('signatures.view_signatures'))
            
        samples = sample_manager.load_samples(signature_name)
        if 0 <= sample_id < len(samples):
            return render_template('sample.html', 
                                 sample=samples[sample_id], 
                                 sample_id=sample_id,
                                 signature=app_state.get_signature(signature_name),
                                 signature_name=signature_name)
        return redirect(url_for('samples.view_samples'))
        
    @bp.route('/view/<signature_name>/<int:sample_id>')
    def view_signature_sample(signature_name, sample_id):
        """View a sample for a specific signature"""
        signature = app_state.get_signature(signature_name)
        if not signature:
            flash(f"Signature '{signature_name}' not found")
            return redirect(url_for('signatures.view_signatures'))
            
        samples = sample_manager.load_samples(signature_name)
        if 0 <= sample_id < len(samples):
            return render_template('sample.html', 
                                 sample=samples[sample_id], 
                                 sample_id=sample_id,
                                 signature=signature,
                                 signature_name=signature_name)
        return redirect(url_for('samples.view_signature_samples', signature_name=signature_name))

    @bp.route('/edit/<int:sample_id>', methods=['GET', 'POST'])
    def edit_sample(sample_id):
        """Edit a specific sample for the current signature."""
        signature_name = app_state.current_signature_name
        if not signature_name:
            flash("No signature selected. Please select a signature first.")
            return redirect(url_for('signatures.view_signatures'))
            
        signature = app_state.get_signature(signature_name)
        samples = sample_manager.load_samples(signature_name)
        
        if request.method == 'POST':
            if 0 <= sample_id < len(samples):
                # Update all fields from the form based on signature definition
                for field in signature.input_fields + signature.output_fields:
                    value = request.form.get(field, '')
                    # Check if this field is marked as a list
                    if request.form.get(f"{field}_is_list", "").lower() == 'true':
                        # Split by newlines and filter out empty lines
                        samples[sample_id][field] = [line.strip() for line in value.split('\n') if line.strip()]
                    else:
                        samples[sample_id][field] = value
                    
                sample_manager.save_samples(samples, signature_name)
                flash("Sample updated successfully")
                return redirect(url_for('samples.view_sample', sample_id=sample_id))
        
        if 0 <= sample_id < len(samples):
            return render_template('edit_sample.html', 
                                 sample=samples[sample_id], 
                                 sample_id=sample_id,
                                 signature=signature,
                                 signature_name=signature_name)
        return redirect(url_for('samples.view_samples'))
    
    @bp.route('/edit/<signature_name>/<int:sample_id>', methods=['GET', 'POST'])
    def edit_signature_sample(signature_name, sample_id):
        """Edit a specific sample for a specific signature."""
        signature = app_state.get_signature(signature_name)
        if not signature:
            flash(f"Signature '{signature_name}' not found")
            return redirect(url_for('signatures.view_signatures'))
            
        samples = sample_manager.load_samples(signature_name)
        
        if request.method == 'POST':
            if 0 <= sample_id < len(samples):
                # Update all fields from the form based on signature definition
                for field in signature.input_fields + signature.output_fields:
                    value = request.form.get(field, '')
                    # Check if this field is marked as a list
                    if request.form.get(f"{field}_is_list", "").lower() == 'true':
                        # Split by newlines and filter out empty lines
                        samples[sample_id][field] = [line.strip() for line in value.split('\n') if line.strip()]
                    else:
                        samples[sample_id][field] = value
                    
                sample_manager.save_samples(samples, signature_name)
                flash("Sample updated successfully")
                return redirect(url_for('samples.view_signature_sample', 
                                      signature_name=signature_name, 
                                      sample_id=sample_id))
        
        if 0 <= sample_id < len(samples):
            return render_template('edit_sample.html', 
                                 sample=samples[sample_id], 
                                 sample_id=sample_id,
                                 signature=signature,
                                 signature_name=signature_name)
        return redirect(url_for('samples.view_signature_samples', signature_name=signature_name))

    @bp.route('/add', methods=['GET', 'POST'])
    def add_sample():
        """Add a new sample for the current signature."""
        from flask import session
        
        signature_name = app_state.current_signature_name
        if not signature_name:
            flash("No signature selected. Please select a signature first.")
            return redirect(url_for('signatures.view_signatures'))
            
        signature = app_state.get_signature(signature_name)
        
        if request.method == 'POST':
            # Create a new sample based on signature definition
            new_sample = {}
            for field in signature.input_fields + signature.output_fields:
                value = request.form.get(field, '')
                # Check if this field is marked as a list
                if request.form.get(f"{field}_is_list", "").lower() == 'true':
                    # Split by newlines and filter out empty lines
                    new_sample[field] = [line.strip() for line in value.split('\n') if line.strip()]
                else:
                    new_sample[field] = value
            
            # Validate the sample
            if sample_manager.validate_sample(new_sample, signature_name):
                samples = sample_manager.load_samples(signature_name)
                samples.append(new_sample)
                sample_manager.save_samples(samples, signature_name)
                flash("New sample added successfully")
                return redirect(url_for('samples.view_samples'))
            else:
                flash("Invalid sample format. Please check all required fields.")
        
        # Check if we have a generated sample from an evaluation
        from_evaluation = session.get('from_evaluation', False)
        generated_sample = session.get('generated_sample', None)
        
        if generated_sample and sample_manager.validate_sample(generated_sample, signature_name):
            # Use the generated sample from the session
            sample = generated_sample
            # Clear the session data to avoid reusing it
            session.pop('generated_sample', None)
            session.pop('from_evaluation', None)
        else:
            # Create an empty sample template
            sample = sample_manager.create_empty_sample(signature_name)
        
        return render_template('add_sample.html', 
                              models=app_state.AVAILABLE_MODELS, 
                              current_model=app_state.current_model,
                              sample=sample,
                              signature=signature,
                              signature_name=signature_name,
                              from_evaluation=from_evaluation)
                              
    @bp.route('/add/<signature_name>', methods=['GET', 'POST'])
    def add_signature_sample(signature_name):
        """Add a new sample for a specific signature."""
        signature = app_state.get_signature(signature_name)
        if not signature:
            flash(f"Signature '{signature_name}' not found")
            return redirect(url_for('signatures.view_signatures'))
        
        if request.method == 'POST':
            # Create a new sample based on signature definition
            new_sample = {}
            for field in signature.input_fields + signature.output_fields:
                value = request.form.get(field, '')
                # Check if this field is marked as a list
                if request.form.get(f"{field}_is_list", "").lower() == 'true':
                    # Split by newlines and filter out empty lines
                    new_sample[field] = [line.strip() for line in value.split('\n') if line.strip()]
                else:
                    new_sample[field] = value
            
            # Validate the sample
            if sample_manager.validate_sample(new_sample, signature_name):
                samples = sample_manager.load_samples(signature_name)
                samples.append(new_sample)
                sample_manager.save_samples(samples, signature_name)
                flash("New sample added successfully")
                return redirect(url_for('samples.view_signature_samples', signature_name=signature_name))
            else:
                flash("Invalid sample format. Please check all required fields.")
        
        # Create an empty sample template
        empty_sample = sample_manager.create_empty_sample(signature_name)
        
        return render_template('add_sample.html', 
                              models=app_state.AVAILABLE_MODELS, 
                              current_model=app_state.current_model,
                              sample=empty_sample,
                              signature=signature,
                              signature_name=signature_name,
                              from_evaluation=False)
    
    @bp.route('/delete/<signature_name>/<int:index>', methods=['POST'])
    def delete_sample(signature_name, index):
        """Delete a sample from a specific signature"""
        signature = app_state.get_signature(signature_name)
        if not signature:
            flash(f"Signature '{signature_name}' not found")
            return redirect(url_for('signatures.view_signatures'))
            
        samples = sample_manager.load_samples(signature_name)
        if 0 <= index < len(samples):
            # Remove the sample
            del samples[index]
            sample_manager.save_samples(samples, signature_name)
            flash(f"Sample #{index+1} deleted successfully")
            
        return redirect(url_for('samples.view_signature_samples', signature_name=signature_name))
        
    return bp