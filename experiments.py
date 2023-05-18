from evolution import Evolution
from cheetah_laboratory import ClassicNNCheetahLab
from arena import Arena
import gymnasium as gym

import plotly.express as px
import numpy as np
import pandas as pd
import logging

nn_depth = 1
nn_width = 8

nn_hidden_layers = [nn_width for _ in range(nn_depth)]
arena = Arena()
cheetah_lab = ClassicNNCheetahLab([17, *nn_hidden_layers, 6])

num_parents_mating = 17
parent_selection_type = 'tournament'
crossover_type = 'single_point'
crossover_probability = 0.8
mutation_type = 'random'
mutation_probability = 0.06
mutation = 'random'
keep_elitism = 10

parameters = {
    'num_generations': 100,
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

evolution = Evolution(cheetah_lab, arena)
evolution.run_evolution(parameters)
evolution.plot_fitness()

behavior = evolution.get_best_behavior()
print(arena.fight(behavior, max_steps=200, render='best8', repetitions=1))



