# LLM Alignment with Causal Models
Master Thesis - Discovering Causal Models which Align with LLMs Using Distributed Alignment Search (DAS)

## Sanity Check Experiment

Example of command to run the sanity check:


```
python3 src/run_das.py --model_path /home/mpislar/LLM_causal_model_learning/models/trained_gpt2forseq --n_training 2 --n_testing 2 --batch_size 2 --epochs 1
```


## Empirical Experiment (Where does each variable live?)

Example of command to run the experiment:

```
python3 src/empirical_living_variables.py --model_path /home/mpislar/LLM_causal_model_learning/models/trained_gpt2forseq --n_examples 2560 --batch_size 128
```