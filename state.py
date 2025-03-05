"""
Module for managing application state
"""
import threading
from .task_definition import TaskDefinition

class AppState:
    """Singleton class for managing application state"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AppState, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initialize the application state"""
        self.current_model = "anthropic/claude-3-5-sonnet-20241022"
        self.optimization_running = False
        self.evaluation_results = {"metrics": {}, "results": []}
        
        # Available models
        self.AVAILABLE_MODELS = [
            'openrouter/anthropic/claude-3.7-sonnet',
            'deepseek/deepseek-reasoner',
            'anthropic/claude-3-7-sonnet-20250219',
            'anthropic/claude-3-5-sonnet-20240620',
            'openai/gpt-4o'
        ]
        
        # Task definition
        self.task_definition = None
        self.load_task_definition()
    
    def load_task_definition(self):
        """Load the task definition or create a default one"""
        self.task_definition = TaskDefinition.load()
        if self.task_definition is None:
            # Create a default PLN task definition
            self.task_definition = TaskDefinition(
                name="PLNTask",
                description="Convert English text to Programming Language for Thought (PLN)",
                input_fields=[
                    {"name": "english", "desc": "English text to convert to PLN"}
                ],
                output_fields=[
                    {"name": "pln_types", "desc": "PLN type definitions"},
                    {"name": "pln_statements", "desc": "PLN statements"},
                    {"name": "pln_questions", "desc": "PLN questions"}
                ]
            )
            self.task_definition.save()
