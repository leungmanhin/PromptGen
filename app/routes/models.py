"""
Routes for managing models
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash
from ..models.model import ModelDefinition

def create_model_routes(app_state):
    bp = Blueprint('models', __name__)
    
    @bp.route('/')
    def view_models():
        """View available models"""
        return render_template('models.html', 
                             models=app_state.models,
                             current_model=app_state.get_current_model())
    
    @bp.route('/select/<model_name>')
    def select_model(model_name):
        """Select a model to use"""
        if app_state.set_current_model(model_name):
            flash(f"Selected model: {model_name}")
        return redirect(url_for('main.index'))
        
    @bp.route('/add', methods=['GET', 'POST'])
    def add_model():
        """Add a new model"""
        if request.method == 'POST':
            name = request.form.get('name', '')
            provider = request.form.get('provider', '')
            description = request.form.get('description', '')
            
            # Process parameters
            parameters = {}
            temperature = request.form.get('temperature', '')
            max_tokens = request.form.get('max_tokens', '')
            
            if temperature:
                try:
                    parameters['temperature'] = float(temperature)
                except ValueError:
                    flash("Temperature must be a number")
                    return render_template('add_model.html')
                    
            if max_tokens:
                try:
                    parameters['max_tokens'] = int(max_tokens)
                except ValueError:
                    flash("Max tokens must be an integer")
                    return render_template('add_model.html')
            
            # Create model
            new_model = ModelDefinition(
                name=name,
                provider=provider,
                description=description,
                parameters=parameters
            )
            
            # Check if model name and provider are valid
            if not name or not provider:
                flash("Model name and provider are required")
                return render_template('add_model.html')
                
            # Add to app state
            if app_state.add_model(new_model):
                # Set as current model
                app_state.set_current_model(new_model.full_name)
                flash(f"Created new model: {new_model.full_name}")
                return redirect(url_for('models.view_models'))
            else:
                flash(f"Model with name '{new_model.full_name}' already exists")
        
        # GET request - show add form
        return render_template('add_model.html')
    
    @bp.route('/edit/<path:model_name>', methods=['GET', 'POST'])
    def edit_model(model_name):
        """Edit an existing model"""
        model = app_state.get_model(model_name)
        if not model:
            flash(f"Model '{model_name}' not found")
            return redirect(url_for('models.view_models'))
            
        if request.method == 'POST':
            description = request.form.get('description', '')
            
            # Process parameters
            parameters = {}
            temperature = request.form.get('temperature', '')
            max_tokens = request.form.get('max_tokens', '')
            
            if temperature:
                try:
                    parameters['temperature'] = float(temperature)
                except ValueError:
                    flash("Temperature must be a number")
                    return render_template('edit_model.html', model=model)
                    
            if max_tokens:
                try:
                    parameters['max_tokens'] = int(max_tokens)
                except ValueError:
                    flash("Max tokens must be an integer")
                    return render_template('edit_model.html', model=model)
            
            # Update model
            model.description = description
            model.parameters = parameters
            
            # Save model
            app_state.update_model(model)
            flash(f"Updated model: {model.full_name}")
            return redirect(url_for('models.view_models'))
            
        # GET request - show edit form
        return render_template('edit_model.html', model=model)
    
    @bp.route('/delete/<path:model_name>', methods=['POST'])
    def delete_model(model_name):
        """Delete a model"""
        if app_state.delete_model(model_name):
            flash(f"Deleted model: {model_name}")
        else:
            flash(f"Failed to delete model: {model_name}")
        return redirect(url_for('models.view_models'))
        
    return bp