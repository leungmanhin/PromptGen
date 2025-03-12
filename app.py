import os
import dspy
from flask import Flask
from .samples import SampleManager
from .optimization import Optimizer
from .evaluation import Evaluator
from .state import AppState

def create_app():
    """Application factory function"""
    flask_app = Flask(__name__)

    flask_app.secret_key = "super secret key"
    
    # Initialize components
    app_state = AppState()
    sample_manager = SampleManager(app_state)
    optimizer = Optimizer(app_state, sample_manager)
    evaluator = Evaluator(app_state, sample_manager)
    
    # Create required directories
    @flask_app.before_request
    def setup_dirs():
        create_directories()
    
    # Import routes here to avoid circular imports
    from . import routes
    
    # Register routes
    flask_app.register_blueprint(
        routes.create_routes(app_state, sample_manager, optimizer, evaluator)
    )
    
    # Add timestamp filter for templates
    @flask_app.template_filter('timestamp_to_date')
    def timestamp_to_date(timestamp):
        """Convert timestamp to readable date"""
        from datetime import datetime
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        return str(timestamp)
    
    # Initialize the model
    dspy.configure(lm=dspy.LM(app_state.current_model))
    
    # Add CLI commands if needed
    @flask_app.cli.command("optimize")
    def optimize_command():
        """Run optimization from command line"""
        from optimization import Optimizer  # Use absolute import
        optimizer = Optimizer(sample_manager)
        optimizer.run_optimization(app_state.current_model)
    
    return flask_app

def create_directories():
    """Create required directories"""
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    os.makedirs("programs", exist_ok=True)

# These functions are no longer needed with the new architecture

# These helper functions are no longer needed with the new architecture

# Routes are now defined in routes.py and registered via blueprint
