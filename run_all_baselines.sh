#!/bin/bash

echo "=== 1. Creating Isolated Python Virtual Environment ==="
python3 -m venv venv

echo "=== 2. Activating Environment ==="
source venv/bin/activate

echo "=== 3. Upgrading pip and Installing Dependencies ==="
pip install --upgrade pip -q
pip install -q pyvene nnsight transformers torch datasets accelerate evaluate networkx scikit-learn matplotlib seaborn tqdm

# Ensure the results output directory exists
mkdir -p results

echo "=== 4. Executing Task 1: Arithmetic Intervenable Probes ==="
python3 src/run_das.py --model_path mara589/arithmetic-gpt2 --causal_model_type arithmetic --results_path results/ --n_training 25000 --n_testing 256 --batch_size 1280 --epochs 4

echo "=== 5. Executing Task 2: Simple/Copy Probes ==="
python3 src/run_das.py --model_path mara589/arithmetic-gpt2 --causal_model_type simple --results_path results/ --n_training 25000 --n_testing 256 --batch_size 1280 --epochs 4

echo "=== 6. Executing Task 3: Evaluating Arithmetic Probes ==="
python3 src/evaluate_das.py --model_path mara589/arithmetic-gpt2 --results_path results/ --n_testing 25600 --batch_size 256 --causal_model_type arithmetic

echo "=== 7. Fine-Tuning Base Boolean GPT-2 Model Locally ==="
# Explicitly runs the binary logic trainer and targets a deterministic local output directory
python3 src/train_binary_gpt2.py --results_path ./boolean_local_results --epochs 5 --batch_size 64 --n_training 4096

echo "=== 8. Executing Task 4: Boolean Logic Probes ==="
# Points directly to the output folder generated in Step 7
python3 src/run_binary_task.py --model_path ./boolean_local_results/trained_gpt2forseq --results_path results/ --n_training 25000 --batch_size 1280 --epochs 4

echo "=== All Pipeline Tasks Completed Successfully ==="