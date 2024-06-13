from typing import Dict, Any
from abc import ABC, abstractmethod
from utils import randNum
from my_pyvene.data_generators.causal_model import CausalModel
# from pyvene import CausalModel

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

class RedundantSummingCausalModels(CausalModelFamily):
    def __init__(self):
        super().__init__()
    
    def construct_default(self):

        variables =  ["X1", "X2", "X3", "Y", "Z", "P", "Q", "O"]
        number_of_entities = 20

        reps = [randNum() for _ in range(number_of_entities)]
        values = {variable:reps for variable in ["X1", "Y", "Z"]}
        values["X2"] = values["X1"]
        values["X3"] = values["X2"]
        values["P"] = list(range(1,11)) # can possibly take values from 1 to 10
        values["Q"] = list(range(2, 21))
        values["O"] = list(range(3, 31))

        def FILLER():
            return reps[0]

        functions = {"X1":FILLER, "X2":FILLER, "X3":FILLER, "Y":FILLER, "Z":FILLER,
                    "P": lambda x, y, z: x,
                    "Q": lambda x, y: x + y,
                    "O": lambda x, y: x + y}
        
        parents = {
            "X1":[], "X2":[], "X3":[], "Y":[], "Z":[],
            "P":["X1", "X2", "X3"],
            "Q": ["Y", "Z"],
            "O":["P", "Q"]
        }

        self.add_model(CausalModel(variables, values, parents, functions), label="(X+X-X)+Y+Z")