from typing import Dict, Any
from abc import ABC, abstractmethod
from src.utils import randNum, randMorgan
from src.my_pyvene.data_generators.causal_model import CausalModel
# from pyvene import CausalModel
import numpy as np
import ast

class CausalModelFamily(ABC): # abstract base class
    def __init__(self):
        self.causal_models: Dict[int, Dict[CausalModel, Any]] = {}
        self.construct_default()

    def add_model(self, causal_model: CausalModel, label: str):
        """Adds a CausalModel to the family. Idealy they are all possible
        hypothesis for an experiment.

        Args:
            causal_model: the CausalModel to add
            label: label the model has
            name: name of the CausalModel
        """
        if causal_model in self.causal_models:
            raise ValueError(f"CausalModel already exists.")
        
        model_info = {
            'causal_model': causal_model,
            'label': label
        }

        self.causal_models[len(self.causal_models) + 1] = model_info

    def get_model_by_id(self, id) -> CausalModel:
        """Retrieve a CausalModel form its family by its id.

        Args:
            id: The id of the CausalModel to retreive.

        Return:
            The CausalModel or None if id is not found.
        """
        return self.causal_models[id]['causal_model']
    
    def get_label_by_id(self, id) -> str:
        """Retrieve the label of the causal model form its family by its id.

        Args:
            id: The id of the label to retreive.

        Return:
            The label of the causal model.
        """
        return self.causal_models[id]['label']
    
    @abstractmethod
    def construct_default(self):
        pass

class ArithmeticCausalModels(CausalModelFamily):
    def __init__(self):
        super().__init__()
    
    def construct_default(self):

        def FILLER():
            return reps[0]
    
        variables = ["X", "Y", "Z", "P", "O"]
        number_of_entities = 20
        reps = [randNum(lower=1, upper=10) for _ in range(number_of_entities)]
        values = {variable:reps for variable in ["X", "Y", "Z"]}
        values["P"] = list(range(2, 21))
        values["O"] = list(range(3, 31))

        functions = {
                "X":FILLER, "Y":FILLER, "Z":FILLER,
                "P": lambda x, y: x + y,
                "O": lambda x, y: x + y}
        
        parents = {
            "X":[], "Y":[], "Z":[], 
            "P":["X", "Y"],
            "O":["P", "Z"]
        }

        self.add_model(CausalModel(variables, values, parents, functions), label='(X+Y)+Z')

        parents = {
            "X":[], "Y":[], "Z":[], 
            "P":["X", "Z"],
            "O":["P", "Y"]
        }
        
        self.add_model(CausalModel(variables, values, parents, functions), label="(X+Z)+Y")

        parents = {
            "X":[], "Y":[], "Z":[], 
            "P":["Y", "Z"],
            "O":["P", "X"]
        }

        pos = {
            "X": (1, 0.1),
            "Y": (2, 0.2),
            "Z": (2.8, 0),
            "P": (2, 2),
            "O": (1.5, 3),
        }
        
        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="X+(Y+Z)")

class SimpleSummingCausalModels(CausalModelFamily):
    def __init__(self):
        super().__init__()
    
    def construct_default(self):

        variables =  ["X", "Y", "Z", "P", "O"]
        number_of_entities = 20

        reps = [randNum() for _ in range(number_of_entities)]
        values = {variable:reps for variable in ["X", "Y", "Z"]}
        values["P"] = list(range(1,11)) # can possibly take values from 1 to 10
        values["O"] = list(range(3, 31))

        def FILLER():
            return reps[0]

        functions = {"X":FILLER, "Y":FILLER, "Z":FILLER,
                    "P": lambda x: x,
                    "O": lambda x, y, z: x + y + z}
        
        parents = {
            "X":[], "Y":[], "Z":[],
            "P":["X"],
            "O":["P", "Y", "Z"]
        }

        self.add_model(CausalModel(variables, values, parents, functions), label="(X)+Y+Z")

        parents = {
            "X":[], "Y":[], "Z":[],
            "P":["Y"],
            "O":["X", "P", "Z"]
        }

        pos = {
            "X": (1, 0.1),
            "Y": (2, 0.2),
            "Z": (2.8, 0),
            "P": (2, 1),
            "O": (1.5, 3),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="X+(Y)+Z")

        parents = {
            "X":[], "Y":[], "Z":[],
            "P":["Z"],
            "O":["X", "Y", "P"]
        }

        pos = {
            "X": (1, 0.1),
            "Y": (2, 0.2),
            "Z": (2.8, 0),
            "P": (2, 2),
            "O": (1.5, 3),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="X+Y+(Z)")

        parents = {
            "X":[], "Y":[], "Z":[],
            "P":["X", "Y", "Z"],
            "O":["P"]
        }

        values["P"] = list(range(3,31))
        functions = {"X":FILLER, "Y":FILLER, "Z":FILLER,
                    "P": lambda x, y, z: x+y+z,
                    "O": lambda x: x}

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="(X+Y+Z)")


class DeMorgansLawCausalModels(CausalModelFamily):
    def __init__(self):
        super().__init__()
    
    def construct_default(self):

        # OP1(OP2(A)) OP1(BIN) OP1(OP3(B))

        variables =  ["X", "Y", "X'", "Y'", "P", "W", "O", "Op1", "Op2", "Op3", "B"]
        number_of_entities = 20

        reps = [randMorgan() for _ in range(number_of_entities)]
        reps = np.array(reps)

        values = {
            "X": list(reps[:, 0]),
            "Y": list(reps[:, 1]),
            "Op1": list(reps[:, 2]),
            "Op2": list(reps[:, 3]),
            "Op3": list(reps[:, 4]),
            "B": list(reps[:, 5])
        }

        values["X'"] = [True, False]
        values["Y'"] = [True, False]
        values["P"] = [True, False]
        values["W"] = [True, False]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "X'": lambda x, op2: not ast.literal_eval(x) if op2 == 'not' else ast.literal_eval(x),
                     "Y'": lambda y, op3: not ast.literal_eval(y) if op3 == 'not' else ast.literal_eval(y),
                     "P": lambda x, op1: not x if op1 == 'not' else x,
                     "W": lambda x, op1: not x if op1 == 'not' else x,
                     "O": lambda x, y, op1, b: (
                        x or y if op1 == 'not' and b == 'and' else
                        x and y if op1 == 'not' and b == 'or' else
                        x and y if op1 == '' and b == 'and' else
                        x or y if op1 == '' and b == 'or' else
                        True
                    )}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "X'":["X", "Op2"],
            "Y'": ["Y", "Op3"],
            "P":["X'", "Op1"],
            "W": ["Y'", "Op1"],
            "O":["P", "W", "Op1", "B"]
        }

        pos = {
            "X": (1, 0),
            "Op2": (2,0),
            "Op1":(3,0),
            "Y": (4, 0),
            "Op3": (5,0),
            "B": (6,0),
            "X'": (1.5, 1),
            "Y'": (4.5, 1),
            "P": (2, 2),
            "W": (3.5, 2),
            "O": (3, 3),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="OP1(OP2(A))_OP1(BIN)_OP1(OP3(B))")

        # OP1(OP2(A) BIN OP3(B))

        variables =  ["X", "Y", "X'", "Y'", "P", "O", "Op1", "Op2", "Op3", "B"]
        number_of_entities = 20

        reps = [randMorgan() for _ in range(number_of_entities)]
        reps = np.array(reps)

        values = {
            "X": list(reps[:, 0]),
            "Y": list(reps[:, 1]),
            "Op1": list(reps[:, 2]),
            "Op2": list(reps[:, 3]),
            "Op3": list(reps[:, 4]),
            "B": list(reps[:, 5])
        }

        values["X'"] = [True, False]
        values["Y'"] = [True, False]
        values["P"] = [True, False]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "X'": lambda x, op2: not ast.literal_eval(x) if op2 == 'not' else ast.literal_eval(x),
                     "Y'": lambda y, op3: not ast.literal_eval(y) if op3 == 'not' else ast.literal_eval(y),
                     "P": lambda x, y, b: x and y if b == 'and' else x or y,
                     "O": lambda p, op1: not p if op1 == 'not' else p}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "X'":["X", "Op2"],
            "Y'": ["Y", "Op3"],
            "P":["X'", "Y'", "B"],
            "O":["P", "Op1"]
        }

        pos = {
            "X": (1, 0),
            "Op2": (2,0),
            "Op1":(3,0),
            "Y": (4, 0),
            "Op3": (5,0),
            "B": (6,0),
            "X'": (1.5, 1),
            "Y'": (4.5, 1),
            "P": (5.5, 2),
            "O": (3, 3),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="OP1(OP2(A)_BIN_OP3(B))")
