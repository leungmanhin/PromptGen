from hyperon import MeTTa
import random
import string
import os
from typing import List, Tuple

class MeTTaHandler:                                                          
    def __init__(self, file: str, read_only: bool = False):
        self.metta = MeTTa()
        self.file = file
        self._read_only = read_only
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.run("!(bind! &kb (new-space))")
        self.run("!(bind! &rules (new-space))")
        self.run_metta_from_file(os.path.join(script_dir, 'chainer.metta'))
        self.run_metta_from_file(os.path.join(script_dir, 'rules.metta'))
        self.run("!(add-atom &kb (get-atoms &rules))")

    def run_metta_from_file(self, file_path):                                
        with open(file_path, 'r') as file:                                   
            chainerstringhere = file.read()                                  
            self.metta.run(chainerstringhere)                                
                                                                             
    @staticmethod                                                            
    def generate_random_identifier(length=8):                                
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    @staticmethod
    def clean_variable_names(expr: str) -> str:
        """Remove #numbers from variable names like $var#1234"""
        import re
        return re.sub(r'\$([a-zA-Z_][a-zA-Z0-9_]*?)#\d+', r'$\1', expr)

    def get_rules(self) -> list[str]:
        """Get the list of rules from the file"""
        res = self.run("!(get-atoms &rules)")[0]
        return [self.clean_variable_names(str(x)) for x in res]

    @property
    def read_only(self) -> bool:
        return self._read_only

    def add_atom_and_run_fc(self, atom: str) -> List[str]:
        self.metta.run(f'!(add-atom &kb {atom})')                  
        res = self.metta.run(f'!(ddfc &kb {atom})')
        out = [self.clean_variable_names(str(elem)) for elem in res[0]]
        if not self.read_only:
            self.append_to_file(f"{atom}")
            [self.append_to_file(elem) for elem in out]
        return out

    def bc(self, atom: str) -> Tuple[List[str], bool]:
        """Run backward chaining on an atom.
        
        Returns:
            Tuple containing:
            - List of intermediate steps/proofs
            - Boolean indicating if the conclusion was proven
        """
        results = self.metta.run('!(ddbc &kb ' + atom + ')')
        # If we got any results back, the conclusion was proven
        proven = len(results[0]) > 0
        return [str(elem) for elem in results[0]], proven

    def add_to_context(self, atom: str) -> str | None:
        """Add atom to context if no conflict exists.
        
        Returns:
            None if atom was added successfully
            The conflicting atom string if a conflict was found
        """
        exp = self.metta.parse_single(atom)
        inctx = self.metta.run("!(match &kb (: " + str(exp.get_children()[1]) + " $a) $a)")[0]

        
        if len(inctx) == 0:
            self.metta.run("!(add-atom &kb " + atom + ")")
            return None

        unify = self.metta.run("!(unify " + str(exp.get_children()[2]) + " (match &kb (: " + str(exp.get_children()[1]) + " $a) $a)  same diff)")

        if str(unify[0][0]) == "same":
            self.metta.run("!(add-atom &kb " + atom + ")")
            return None
        else:
            return inctx

        
    def run(self, atom: str):
        return self.metta.run(atom)

    def run_clean(self, atom: str) -> List[str]:
        res = self.metta.run(atom)
        return [self.clean_variable_names(str(elem)) for elem in res[0]]
                                                                             
    def store_kb_to_file(self):
        if self.read_only:
            print("Warning: Cannot store KB in read-only mode")
            return
        kb_content = self.metta.run('!(match &kb $a $a)')
        with open(self.file, 'w') as f:                                       
            for element in kb_content[0]:
                f.write(str(element) + "\n")
                                                                             
    def load_kb_from_file(self):
        if os.path.exists(self.file):
            with open(self.file, 'r') as f:                                       
                for elment in f:
                    self.metta.run(f'!(add-atom &kb {elment})')
        else:
            print(f"Warning: File {self.file} does not exist. No KB loaded.")

    def append_to_file(self, elem: str):
        if self.read_only:
            print("Warning: Cannot append to file in read-only mode")
            return
        with open(self.file, 'a') as f:
            f.write(elem)
