"""
Module for managing application state
"""
import threading
import os
import json
import time

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
            'anthropic/claude-3-7-sonnet-20250219',
            'openrouter/anthropic/claude-3.7-sonnet',
            'anthropic/claude-3-5-sonnet-20240620',
            'deepseek/deepseek-reasoner',
            'openai/gpt-4o'
        ]
        
        # Program management
        self.programs = {}  # Dictionary of program_id -> metadata
        self.current_program_id = None
        self._ensure_programs_directory()
        self.load_available_programs()
    
    def _ensure_programs_directory(self):
        """Create programs directory if it doesn't exist"""
        os.makedirs("./programs", exist_ok=True)
    
    def load_available_programs(self):
        """Load metadata for all available programs"""
        programs_dir = "./programs/"
        self.programs = {}
        
        # Find all program directories
        if os.path.exists(programs_dir):
            for program_id in os.listdir(programs_dir):
                program_path = os.path.join(programs_dir, program_id)
                if os.path.isdir(program_path):
                    metadata_path = os.path.join(program_path, "metadata.json")
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, "r") as f:
                                metadata = json.load(f)
                            # Add creation time from file if not in metadata
                            if "created_at" not in metadata:
                                metadata["created_at"] = os.path.getctime(metadata_path)
                            # Add program ID to metadata
                            metadata["id"] = program_id
                            # Add to programs dictionary
                            self.programs[program_id] = metadata
                        except Exception as e:
                            print(f"Error loading program {program_id}: {e}")
        
        # Set current program to the most recent if none is selected
        if not self.current_program_id and self.programs:
            # Sort by creation time (newest first)
            sorted_programs = sorted(self.programs.items(), 
                                    key=lambda x: x[1].get("created_at", 0), 
                                    reverse=True)
            self.current_program_id = sorted_programs[0][0]
    
    def set_current_program(self, program_id):
        """Set the current program"""
        if program_id in self.programs:
            self.current_program_id = program_id
            return True
        return False
        
    def create_new_program(self, model_name, base_program_id=None):
        """Create a new empty program
        
        Returns:
            str: The ID of the created program
        """
        import dspy
        import os
        import json
        import time
        
        # Define the PLN signature
        class PLNTask(dspy.Signature):
            """
            Convert the given english to PLN.
            If it is a question construct one ore more queries to answer it.
            If it is a statement construct one or more statements to add the knowledge to the KB.
            Provide type definitions for all predicates.

            Given Types are:
            (: Implication (-> (: $implicant Type) (: $consequent Type) Type))
            (: And (-> (: $a Type) (: $b Type) Type))
            (: Or (-> (: $a Type) (: $b Type) Type))
            (: Equivalence (-> (: $a Type) (: $b Type) Type))
            (: WithTV (-> (: $a Type) (: $tv TV) Type))
            (: STV (-> (: $strength Number) (: $confidence Number) TV))

            All queries or statments should be wrapped in WithTV.
            Example Statment:
            (: proofname (WithTV (Predicate object) (STV 1.0 1.0)))
            Example Query :
            (: $query (WithTV (Predicate object) $tv))
            meaning (try to find a proof $query that Predicate applies to object and get me the $tv)
            Predicate or object could also be varaibles in a query.
            """
            english = dspy.InputField(desc="English text to convert to PLN")
            pln_types = dspy.OutputField(desc="PLN type definitions")
            pln_statements = dspy.OutputField(desc="PLN statements")
            pln_query = dspy.OutputField(desc="PLN query")

        # Create a new task using the PLN signature
        task = dspy.ChainOfThought(PLNTask)
        
        # Generate a unique ID for the program
        program_id = f"program_{int(time.time())}"
        program_dir = f"./programs/{program_id}/"
        
        # Create directory if it doesn't exist
        os.makedirs(program_dir, exist_ok=True)
        
        # Save program to the directory
        task.save(program_dir, save_program=True)
        
        # Add additional metadata
        metadata_path = os.path.join(program_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Add additional metadata
            metadata["model"] = model_name
            metadata["created_at"] = time.time()
            metadata["task_name"] = "English to PLN Converter"
            metadata["base_program_id"] = base_program_id
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        # Update current program in app state
        self.current_program_id = program_id
        self.load_available_programs()  # Refresh program list
        
        return program_id
    
    def delete_program(self, program_id):
        """Delete a program
        
        Args:
            program_id (str): The ID of the program to delete
            
        Returns:
            bool: True if the program was deleted, False otherwise
        """
        import shutil
        
        if program_id not in self.programs:
            return False
        
        program_dir = f"./programs/{program_id}/"
        if not os.path.exists(program_dir):
            return False
        
        try:
            # Delete the program directory
            shutil.rmtree(program_dir)
            
            # Update the current program if needed
            if self.current_program_id == program_id:
                # Find another program to set as current
                self.current_program_id = None
                self.load_available_programs()
            else:
                # Just remove from the programs dictionary
                del self.programs[program_id]
                
            return True
        except Exception as e:
            print(f"Error deleting program {program_id}: {e}")
            return False
