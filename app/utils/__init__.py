# Utils package
from .cleanpln import cleanAndScore  # noqa: F401
import dspy
from app.config import Config

def get_lm(model_name=None, **kwargs):
    """Get LM with required parameters for specific models.
    
    Args:
        model_name: Optional model name, defaults to Config.DEFAULT_MODEL
        **kwargs: Additional parameters to pass to dspy.LM
        
    Returns:
        Configured dspy.LM instance
    """
    from app.services.state import AppState
    
    app_state = AppState()
    selected_model_name = model_name or app_state.current_model
    
    # Get model definition from app state if available
    model_def = app_state.get_model(selected_model_name)
    
    # Default parameters
    params = {}
    
    # If model definition exists, use its parameters
    if model_def:
        params.update(model_def.parameters)
    # Fallback to hardcoded parameters for specific models
    elif selected_model_name and "o3-mini" in selected_model_name:
        params.update({
            "temperature": 1.0,
            "max_tokens": 5000
        })

    print(f"Using model: {selected_model_name}")
    print(f"Parameters: {params}")
    
    # Allow override of defaults through kwargs
    params.update(kwargs)
    
    return dspy.LM(selected_model_name or Config.DEFAULT_MODEL, **params)
