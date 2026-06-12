#!/bin/bash
# Install dependencies safely
python3 -m pip install --user pyvene transformers torch datasets accelerate evaluate networkx scikit-learn matplotlib seaborn tqdm

# 1. Run the Arithmetic Intervenable Probes (Streamlined Volume)
python3 src/run_das.py --model_path mara589/arithmetic-gpt2 --causal_model_type arithmetic --n_training 25000 --n_testing 256 --batch_size 1280 --epochs 4

# 2. Train the Simple/Copy Probes (Arithmetic Baseline Check)
python3 src/run_das.py --model_path mara589/arithmetic-gpt2 --causal_model_type simple --n_training 25000 --n_testing 256 --batch_size 1280 --epochs 4

# 3. Evaluate the Arithmetic Probes to Calculate IIA Scores
python3 evaluate_das.py --model_path mara589/arithmetic-gpt2 --results_path results/ --n_testing 25600 --batch_size 256 --causal_model_type arithmetic

# 4. Train & Evaluate the Boolean Logic Probes
python3 src/run_binary_task.py --model_path mara589/boolean-gpt2 --results_path results/ --n_training 25000 --n_testing 256 --batch_size 1280 --epochs 4