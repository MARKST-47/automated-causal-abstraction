# Disentangling Causal Models with DAS
Master Thesis - Discovering Causal Models which Align with LLMs Using Distributed Alignment Search (DAS)

## Sanity Check Experiment

Example of command to run the sanity check:


```
python3 src/run_das.py --model_path /home/mpislar/LLM_causal_model_learning/models/trained_gpt2forseq --n_training 2 --n_testing 2 --batch_size 2 --epochs 1
```


## Empirical Experiment (Where does each variable live?)

Example of command to run the experiment:

```
python3 src/run_das.py --model_path /home/mpislar/LLM_causal_model_learning/models/trained_gpt2forseq --causal_model_type simple --n_training 256000 --n_testing 256 --batch_size 1280 --epochs 4
```
