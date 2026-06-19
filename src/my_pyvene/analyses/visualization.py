import seaborn
import torch
import matplotlib.pyplot as plt
import numpy as np
from pyvene import IntervenableModel
import os
import pandas as pd
import seaborn as sns

def rotation_token_heatmap(rotate_layer, 
                           tokens, 
                           token_size, 
                           variables, 
                           intervention_size,
                           fig_path=''):

    W = rotate_layer.weight.data
    in_dim, out_dim = W.shape

    assert in_dim % token_size == 0
    assert in_dim / token_size >= len(tokens) 

    assert out_dim % intervention_size == 0
    assert out_dim / intervention_size >= len(variables) 
    
    heatmap = []
    for j in range(len(variables)):
        row = []
        for i in range(len(tokens)):
            row.append(torch.norm(W[i*token_size:(i+1)*token_size, j*intervention_size:(j+1)*intervention_size]))
        mean = sum(row)
        heatmap.append([x/mean for x in row])
    
    heatmap_fig = seaborn.heatmap(heatmap, 
                       xticklabels=tokens, 
                       yticklabels=variables)
    if fig_path != '':
        plt.savefig(fig_path)
        plt.close()

    return heatmap_fig

def rotation_token_layers_heatmap(results_path, 
                                  cm_id,
                                  label,
                                  low_rank_dimension, 
                                  model, 
                                  tokens,
                                  token_size,
                                  variables,
                                  intervention_size,
                                  fig_path='attention_weights.png'):
    heatmap_data = []

    for layer in range (12):

        intervenable_model_path = os.path.join(results_path, f'intervenable_models/cm_{cm_id}/intervenable_{low_rank_dimension}_{layer}')
        intervenable = IntervenableModel.load(intervenable_model_path, model=model)
        intervenable.set_device("cuda")
        intervenable.disable_model_gradients()

        first_key = list(intervenable.interventions)[0]
        rotation_layer = intervenable.interventions[first_key][0].rotate_layer.cpu()

        W = rotation_layer.weight.data
        in_dim, out_dim = W.shape

        assert in_dim % token_size == 0
        assert in_dim / token_size >= len(tokens) 

        assert out_dim % intervention_size == 0
        assert out_dim / intervention_size >= len(variables) 
        
        heatmap_layer = []
        for j in range(len(variables)):
            row = []
            for i in range(len(tokens)):
                row.append(torch.norm(W[i*token_size:(i+1)*token_size, j*intervention_size:(j+1)*intervention_size]))
            mean = sum(row)
            heatmap_layer.append([x/mean for x in row])

        heatmap_data.append(heatmap_layer)
    heatmap_df = pd.DataFrame(np.vstack(heatmap_data), 
                              index=pd.Index([f"Layer {i+1}" for i in range(12)]),  # Add layer labels
                              columns=tokens)
    
    plt.figure(figsize=(10, 8)) 
    heatmap_fig = seaborn.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Normalized Norm'}, vmin=0, vmax=1) 
    # plt.title(f"Causal Model {label} - Rotation Weights by Layer and Token")
    plt.xlabel("Tokens")
    plt.ylabel(f"P={label}")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return heatmap_fig