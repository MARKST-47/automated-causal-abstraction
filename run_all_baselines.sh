#!/bin/bash

echo "=== 1. Creating Isolated Python Virtual Environment ==="
python3 -m venv venv

echo "=== 2. Activating Environment ==="
source venv/bin/activate

echo "=== 3. Upgrading pip and Installing Dependencies ==="
pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: "sentencepiece>=0.1.99"
python -m pip install --no-deps "pyvene==0.0.7"
pip install -q nnsight transformers torch datasets accelerate evaluate networkx scikit-learn matplotlib seaborn tqdm scikit-learn

# Ensure the results output directory exists
mkdir -p results

echo "=== 4. Executing Task 1: Arithmetic Intervenable Probes ==="
python3 /home/causalml26_team002/Causal_Project/src/run_das.py --model_path mara589/arithmetic-gpt2 --causal_model_type arithmetic --results_path /home/causalml26_team002/Causal_Project/results/ --n_training 25000 --n_testing 256 --batch_size 1280 --epochs 4

echo "=== 5. Executing Task 2: Simple/Copy Probes ==="
python3 /home/causalml26_team002/Causal_Project/src/run_das.py --model_path mara589/arithmetic-gpt2 --causal_model_type simple --results_path /home/causalml26_team002/Causal_Project/results/ --n_training 25000 --n_testing 256 --batch_size 1280 --epochs 4

echo "=== 6. Executing Task 3: Evaluating Arithmetic Probes ==="
python3 /home/causalml26_team002/Causal_Project/src/evaluate_das.py --model_path mara589/arithmetic-gpt2 --results_path /home/causalml26_team002/Causal_Project/results/ --n_testing 25600 --batch_size 256 --causal_model_type arithmetic

echo "=== 7. Executing Task 4: Boolean Logic Probes ==="
# CHANGED: Using the discovered HuggingFace path 'mara589/binary-gpt2' 
python3 /home/causalml26_team002/Causal_Project/src/run_binary_task.py --model_path mara589/binary-gpt2 --results_path /home/causalml26_team002/Causal_Project/results/ --n_training 25000 --batch_size 1280 --epochs 4

echo "=== All Pipeline Tasks Completed Successfully ==="
