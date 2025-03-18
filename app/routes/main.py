"""
Main routes for the application
"""
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash

def create_main_routes(app_state, sample_manager, optimizer, evaluator):
    bp = Blueprint('main', __name__)
    
    @bp.route('/')
    def index():
        """Home page route"""
        samples = sample_manager.load_samples()
        return render_template('index.html', 
                            samples=samples, 
                            optimization_running=optimizer.running,
                            evaluation_results=app_state.evaluation_results,
                            models=app_state.AVAILABLE_MODELS,
                            current_model=app_state.current_model,
                            app_state=app_state)
                            
    @bp.route('/signatures')
    def view_signatures():
        """Redirect to the signatures blueprint"""
        return redirect(url_for('signatures.view_signatures'))
        
    @bp.route('/samples')
    def view_samples():
        """Redirect to the samples blueprint"""
        return redirect(url_for('samples.view_samples'))

    @bp.route('/programs')
    def view_programs():
        """Redirect to the programs blueprint"""
        return redirect(url_for('programs.view_programs'))
    
    @bp.route('/set_model', methods=['POST'])
    def set_model():
        """Set the current model"""
        model_name = request.form.get('model')
        if model_name in app_state.AVAILABLE_MODELS:
            app_state.current_model = model_name
            flash(f"Model changed to {model_name}")
        return redirect(url_for('main.index'))
    
    @bp.route('/optimization_status')
    def optimization_status():
        """Check optimization status"""
        return jsonify({"running": optimizer.running})
        
    @bp.route('/evaluation_results')
    def evaluation_results():
        """View detailed evaluation results"""
        if not app_state.evaluation_results:
            flash("No evaluation results available. Please run an evaluation first.")
            return redirect(url_for('main.index'))
            
        return render_template('evaluation_results.html', app_state=app_state)

    return bp