import os.path

import optuna
import numpy as np
import pandas as pd
import logging

from arena import Arena
from cheetah_laboratory import ClassicNNCheetahLab
from evolution import Evolution




def objective(trial):
    nn_depth = trial.suggest_int('nn_depth', 1, 5)
    nn_width = trial.suggest_categorical('nn_width', [2,4,8,16])
    
    nn_hidden_layers = [nn_width for _ in range(nn_depth)]
    arena = Arena()
    cheetah_lab = ClassicNNCheetahLab([17, *nn_hidden_layers, 6])
    evolution = Evolution(cheetah_lab, arena)
    
    num_parents_mating = trial.suggest_int('num_parents_mating', 10, 40)
    parent_selection_type = trial.suggest_categorical('parent_selection_type', ['tournament', 'rws'])
    crossover_type = trial.suggest_categorical('crossover_type', ['single_point', 'uniform'])
    crossover_probability = trial.suggest_float('crossover_probability', 0.5, 1.0)
    mutation_type = trial.suggest_categorical('mutation_type', ['random', 'normal_distribution'])
    mutation_probability = trial.suggest_float('mutation_probability', 0.001, 0.5)
    mutation = evolution.mutation_normal_distribution if mutation_type == 'normal_distribution' else 'random'
    keep_elitism = trial.suggest_int('keep_elitism', 0, 10)
    
    parameters = {
        'num_generations': 30,
        'sol_per_pop': 100,
        'num_parents_mating': num_parents_mating,
        'init_range_low': -1,
        'init_range_high': 1,
        'parent_selection_type': parent_selection_type,
        'crossover_type': crossover_type,
        'crossover_probability': crossover_probability,
        'mutation_type': mutation,
        'mutation_probability': mutation_probability,
        'keep_elitism': keep_elitism,
    }

    evolution.run_evolution(parameters)
    evolution.plot_fitness()
    return evolution.get_best_solution()[1]

DATA_PATH = 'data/cheetah_hyperparams.csv'
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=150)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(DATA_PATH, index=False)
print(df)


