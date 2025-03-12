from flask import Flask
from .config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    from .services.state import AppState
    from .services.samples import SampleManager
    from .services.optimization import Optimizer
    from .services.evaluation import Evaluator
    
    app_state = AppState()
    sample_manager = SampleManager(app_state)
    optimizer = Optimizer(app_state, sample_manager)
    evaluator = Evaluator(app_state, sample_manager)
    
    # Register blueprints
    from .routes.main import create_main_routes
    from .routes.signatures import create_signature_routes
    from .routes.samples import create_sample_routes
    from .routes.programs import create_program_routes
    from .routes.api import create_api_routes
    from .routes.redirect_handler import create_redirect_routes
    
    app.register_blueprint(create_main_routes(app_state, sample_manager, optimizer, evaluator))
    app.register_blueprint(create_signature_routes(app_state, sample_manager), url_prefix='/signatures')
    app.register_blueprint(create_sample_routes(app_state, sample_manager), url_prefix='/samples')
    app.register_blueprint(create_program_routes(app_state, optimizer), url_prefix='/programs')
    app.register_blueprint(create_api_routes(app_state, sample_manager, optimizer, evaluator), url_prefix='/api')
    
    # Add redirect routes for backward compatibility
    app.register_blueprint(create_redirect_routes())
    
    # Template filters
    from .utils.filters import register_filters
    register_filters(app)
    
    return app