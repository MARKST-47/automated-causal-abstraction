from typing import Dict, Any
from abc import ABC, abstractmethod

from utils import randNum, randMorgan
from my_pyvene.data_generators.causal_model import CausalModel

# from utils import randNum, randMorgan
# from my_pyvene.data_generators.causal_model import CausalModel
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

        def evaluate_logic(op1, op2, x, b, op3, y):

            def apply_op(op, val):
                return not val if op == 'Not' else val

            x_val = x == 'True'
            y_val = y == 'True'

            result = apply_op(op2, x_val) and apply_op(op3, y_val) if b == 'And' else \
                    apply_op(op2, x_val) or apply_op(op3, y_val)

            return not result if op1 == 'Not' else result

        def apply_op_to_b(op, b):
            if op == 'Not':
                if b == 'And':
                    return 'Or'
                else:
                    return 'And'
            return b

        '''

        # OP1(OP2(A)) OP1(BIN) OP1(OP3(B))

        variables =  ["X", "Y", "X'", "Y'", "V", "W", "O", "Op1", "Op2", "Op3", "B", "B'"]
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
        values["V"] = [True, False]
        values["W"] = [True, False]
        values["B'"] = ['And', 'Or']
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "X'": lambda x, op2: not ast.literal_eval(x) if op2 == 'Not' else ast.literal_eval(x),
                     "Y'": lambda y, op3: not ast.literal_eval(y) if op3 == 'Not' else ast.literal_eval(y),
                     "V": lambda x, op1: not x if op1 == 'Not' else x,
                     "W": lambda x, op1: not x if op1 == 'Not' else x,
                     "B'": lambda op1, b: apply_op_to_b(op1, b),
                     "O": lambda x, y, b: x and y if b == 'And' else x or y}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "X'":["X", "Op2"],
            "Y'": ["Y", "Op3"],
            "V":["X'", "Op1"],
            "W": ["Y'", "Op1"],
            "B'": ["Op1", "B"],
            "O":["V", "W", "B'"]
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
            "V": (2, 2),
            "W": (3.5, 2),
            "B'": (6, 1.5),
            "O": (3, 3),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="OP1(OP2(A))_OP1(BIN)_OP1(OP3(B))")

        # OP1(OP2(A) BIN OP3(B))

        variables =  ["X", "Y", "X'", "Y'", "Q", "O", "Op1", "Op2", "Op3", "B"]
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
        values["Q"] = [True, False]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "X'": lambda x, op2: not ast.literal_eval(x) if op2 == 'Not' else ast.literal_eval(x),
                     "Y'": lambda y, op3: not ast.literal_eval(y) if op3 == 'Not' else ast.literal_eval(y),
                     "Q": lambda x, y, b: x and y if b == 'And' else x or y,
                     "O": lambda p, op1: not p if op1 == 'Not' else p}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "X'":["X", "Op2"],
            "Y'": ["Y", "Op3"],
            "Q":["X'", "Y'", "B"],
            "O":["Q", "Op1"]
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
            "Q": (5.5, 2),
            "O": (3, 3),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="OP1(OP2(A)_BIN_OP3(B))")

        '''

        '''
        # OP1


        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = ["Not", "I"]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda x: x,
                     "O": lambda op1, op2, x, b, op3, y: evaluate_logic(op1, op2, x, b, op3, y)}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["Op1"],
            "O":["P", "Op2", "X", "B", "Op3", "Y"]
        }

        pos = {
            "X": (1, 0),
            "Op2": (2,0),
            "Op1":(3,0),
            "Y": (4, 0),
            "Op3": (5,0),
            "B": (6,0),
            "P": (3, 2),
            "O": (3, 3),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="OP1")

        # OP2

        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = ["Not", "I"]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda x: x,
                     "O": lambda op1, op2, x, b, op3, y: evaluate_logic(op1, op2, x, b, op3, y)}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["Op2"],
            "O":["Op1", "P", "X", "B", "Op3", "Y"]
        }

        pos = {
            "X": (1, 0),
            "Op2": (2,0),
            "Op1":(3,0),
            "Y": (4, 0),
            "Op3": (5,0),
            "B": (6,0),
            "P": (2.5, 1),
            "O": (3, 2),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="OP2")

        # Op3

        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = ["Not", "I"]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda x: x,
                     "O": lambda op1, op2, x, b, op3, y: evaluate_logic(op1, op2, x, b, op3, y)}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["Op3"],
            "O":["Op1", "Op2", "X", "B", "P", "Y"]
        }

        pos = {
            "X": (1, 0),
            "Op2": (2,0),
            "Op1":(3,0),
            "Y": (4, 0),
            "Op3": (5,0),
            "B": (6,0),
            "P": (4, 1),
            "O": (3, 2),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="OP3")
        
        # X

        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = ["True", "False"]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda x: x,
                     "O": lambda op1, op2, x, b, op3, y: evaluate_logic(op1, op2, x, b, op3, y)}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["X"],
            "O":["Op1", "Op2", "P", "B", "Op3", "Y"]
        }

        pos = {
            "X": (1, 0),
            "Op2": (2,0),
            "Op1":(3,0),
            "Y": (4, 0),
            "Op3": (5,0),
            "B": (6,0),
            "P": (1.2, 1),
            "O": (3, 2),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="X")

        # Y

        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = ["True", "False"]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda x: x,
                     "O": lambda op1, op2, x, b, op3, y: evaluate_logic(op1, op2, x, b, op3, y)}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["Y"],
            "O":["Op1", "Op2", "X", "B", "Op3", "P"]
        }

        pos = {
            "X": (1, 0),
            "Op2": (2,0),
            "Op1":(3,0),
            "Y": (4, 0),
            "Op3": (5,0),
            "B": (6,0),
            "P": (3.5, 1),
            "O": (3, 2),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="Y")

        # B

        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = ["And", "Or"]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda x: x,
                     "O": lambda op1, op2, x, b, op3, y: evaluate_logic(op1, op2, x, b, op3, y)}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["B"],
            "O":["Op1", "Op2", "X", "P", "Op3", "Y"]
        }

        pos = {
            "X": (1, 0),
            "Op2": (2,0),
            "Op1":(3,0),
            "Y": (4, 0),
            "Op3": (5,0),
            "B": (6,0),
            "P": (5, 1),
            "O": (3, 2),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="B")

        '''

        # O

        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = [True, False]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda op1, op2, x, b, op3, y: evaluate_logic(op1, op2, x, b, op3, y),
                     "O": lambda p: p}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["Op1", "Op2", "X", "B", "Op3", "Y"],
            "O":["P"]
        }

        pos = {
            "X": (1, 0),
            "Op2": (2,0),
            "Op1":(3,0),
            "Y": (4, 0),
            "Op3": (5,0),
            "B": (6,0),
            "P": (3, 1),
            "O": (3, 2),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="O")

        '''
        # X'

        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = [True, False]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]
        
        def evaluate_logic_x_prim(p, op1, y, op3, b):

            def apply_op(op, val):
                return not val if op == 'Not' else val

            y_val = y == 'True'

            result = p and apply_op(op3, y_val) if b == 'And' else \
                    p or apply_op(op3, y_val)

            return not result if op1 == 'Not' else result

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda x, op2: not(x == 'True') if op2 == 'Not' else x == 'True',
                     "O": lambda p, op1, y, op3, b: evaluate_logic_x_prim(p, op1, y, op3, b)}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["X", "Op2"],
            "O":["P", "Op1", "Y", "Op3", "B"]
        }

        pos = {
            "X": (1, 0),
            "Op2": (2,0),
            "Op1":(3,0),
            "Y": (4, 0),
            "Op3": (5,0),
            "B": (6,0),
            "P": (2, 1),
            "O": (3, 2),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="X'")

        # Y'

        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = [True, False]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]
        
        def evaluate_logic_y_prim(x, op2, op1, p, b):

            def apply_op(op, val):
                return not val if op == 'Not' else val

            x_val = x == 'True'

            result = apply_op(op2, x_val) and p if b == 'And' else \
                    apply_op(op2, x_val) or p

            return not result if op1 == 'Not' else result

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda y, op3: not(y == 'True') if op3 == 'Not' else y == 'True',
                     "O": lambda x, op2, op1, p, b: evaluate_logic_y_prim(x, op2, op1, p, b)}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["Y", "Op3"],
            "O":["X", "Op2", "Op1", "P", "B"]
        }

        pos = {
            "X": (1, 0),
            "Op2": (2,0),
            "Op1":(3,0),
            "Y": (4, 0),
            "Op3": (5,0),
            "B": (6,0),
            "P": (4, 1),
            "O": (3, 2),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="Y'")

        # Q

        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = [True, False]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]
        
        def evaluate_logic_q(x, op2, y, op3, b):

            def apply_op(op, val):
                return not val if op == 'Not' else val

            x_val = x == 'True'
            y_val = y == 'True'

            result = apply_op(op2, x_val) and apply_op(op3, y_val) if b == 'And' else \
                    apply_op(op2, x_val) or apply_op(op3, y_val)

            return result

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda x, op2, y, op3, b: evaluate_logic_q(x, op2, y, op3, b),
                     "O": lambda op1, p: not p if op1 == 'Not' else p}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["X", "Op2", "Y", "Op3", "B"],
            "O":["Op1", "P"]
        }

        pos = {
            "X": (3, 0),
            "Op2": (2,0),
            "Op1":(1,0),
            "Y": (6, 0),
            "Op3": (5,0),
            "B": (4,0),
            "P": (4, 1),
            "O": (3, 2),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="Q")

        # V

        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = [True, False]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]
        
        def evaluate_logic_v(op1, op2, x):

            def apply_op(op, val):
                return not val if op == 'Not' else val

            x_val = x == 'True'

            result = apply_op(op2, x_val)

            return not result if op1 == 'Not' else result
        
        def evaluate_logic_output(op1, p, b, op3, y):

            def apply_op(op, val):
                return not val if op == 'Not' else val

            y_val = not y == 'True' if op1 == 'Not' else y == 'True'

            if op1 == 'Not':
                if b == 'And':
                    b = 'Or'
                else:
                    b = 'And'

            return apply_op(op3, y_val) and p if b == 'And' else apply_op(op3, y_val) or p

        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda op1, op2, x: evaluate_logic_v(op1, op2, x),
                     "O": lambda op1, p, b, op3, y: evaluate_logic_output(op1, p, b, op3, y)}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["Op1", "Op2", "X"],
            "O":["Op1", "P", "B", "Op3", "Y"]
        }

        pos = {
            "X": (3, 0),
            "Op2": (2,0),
            "Op1":(1,0),
            "Y": (6, 0),
            "Op3": (5,0),
            "B": (4,0),
            "P": (3, 1),
            "O": (3, 2),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="V")

        # W

        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = [True, False]
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]
        
        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda op1, op2, x: evaluate_logic_v(op1, op2, x),
                     "O": lambda op1, p, b, op3, y: evaluate_logic_output(op1, p, b, op3, y)}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["Op1", "Op3", "Y"],
            "O":["Op1", "P", "B", "Op2", "X"]
        }

        pos = {
            "X": (3, 0),
            "Op2": (2,0),
            "Op1":(1,0),
            "Y": (6, 0),
            "Op3": (5,0),
            "B": (4,0),
            "P": (5, 1),
            "O": (3, 2),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="W")

        # B'

        variables =  ["X", "Y", "P", "O", "Op1", "Op2", "Op3", "B"]
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

        values["P"] = ['And', 'Or']
        values["O"] = [True, False]

        def FILLER_XY():
            return reps[:,0][0]
        
        def FILLER_OP():
            return reps[:,3][0]
        
        def FILLER_BIN_OP():
            return reps[:,5][0]
        
        def apply_op(op, val):
            return not val if op == 'Not' else val
        
        functions = {"X":FILLER_XY, "Y":FILLER_XY, "Op1": FILLER_OP, "Op2": FILLER_OP, "Op3": FILLER_OP, "B": FILLER_BIN_OP,
                     "P": lambda op1, b: apply_op_to_b(op1, b),
                     "O": lambda op1, op2, x, b, op3, y: apply_op(op1, apply_op(op2, ast.literal_eval(x))) and apply_op(op1, apply_op(op3, ast.literal_eval(y))) if b == 'And' \
                                                         else apply_op(op1, apply_op(op2, ast.literal_eval(x))) or apply_op(op1, apply_op(op3, ast.literal_eval(y)))}
        
        parents = {
            "X":[], "Y":[], "Op1":[], "Op2":[], "Op3":[], "B": [],
            "P":["Op1", "B"],
            "O":["Op1", "Op2", "X", "P", "Op3", "Y"]
        }

        pos = {
            "X": (3, 0),
            "Op2": (2,0),
            "Op1":(1,0),
            "Y": (6, 0),
            "Op3": (5,0),
            "B": (4,0),
            "P": (3.5, 1),
            "O": (3, 2),
        }

        self.add_model(CausalModel(variables, values, parents, functions, pos=pos), label="B'")
        '''