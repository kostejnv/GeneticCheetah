import numpy as np

from arena import Arena
from cheetah_laboratory import CheetahLab
import pygad

import os

class Evolution:
    def __init__(self, cheetah_lab: CheetahLab, arena: Arena, sigma=0.5):
        self.cheetah_lab = cheetah_lab
        self.arena = arena
        self.ga_instance = None
        self.cpu_count = None# , int(os.cpu_count())-1 # One thread free for other computing - NONE = no multithreading
        self.sigma = sigma
        
    def _fitness_func(self, ga_instance, solution, solution_idx):
        # [PyGAD documentation] - fitness must be maximization function | https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#preparing-the-fitness-func-parameter
        cheetah = self.cheetah_lab.get_cheetah_behavior(solution)
        score = self.arena.fight(cheetah, max_steps=100)
        return score
        
    def run_evolution(self, parameters: dict):
        self.ga_instance = pygad.GA(
            fitness_func=self._fitness_func,
            num_genes= self.cheetah_lab.get_genom_length(),
            save_solutions= False, # True,  #UNDO Daniel
            save_best_solutions=True,
            parallel_processing=["process", 8],  # 8 #UNDO Daniel
            **parameters)
        self.ga_instance.run()
        
    @staticmethod
    def mutation_normal_distribution(offspring,ga_instance):
        mask = np.random.choice([True, False], offspring.shape, p=[ga_instance.mutation_probability, 1-ga_instance.mutation_probability])
        random_values = np.random.normal(0, 0.5, mask.sum())
        offspring[mask] = offspring[mask] + random_values
        return offspring
    
    def set_cheetah_lab(self, cheetah_lab: CheetahLab):
        self.cheetah_lab = cheetah_lab

    def plot_fitness(self):
        self.ga_instance.plot_fitness()
    
    def save(self, filename):
        self.ga_instance.save(filename)
        
    def load(self, filename):
        self.ga_instance.load(filename)
        
    def get_best_solution(self):
        return self.ga_instance.best_solution()
    
    def get_best_behavior(self):
        return self.cheetah_lab.get_cheetah_behavior(self.get_best_solution()[0])
    
    
    