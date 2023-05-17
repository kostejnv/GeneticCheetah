from arena import Arena
from cheetah_laboratory import CheetahLab
import pygad

import os

class Evolution:
    def __init__(self, cheetah_lab: CheetahLab, arena: Arena):
        self.cheetah_lab = cheetah_lab
        self.arena = arena
        self.ga_instance = None
        self.cpu_count = None# , int(os.cpu_count())-1 # One thread free for other computing - NONE = no multithreading
        
    def _fitness_func(self, ga_instance, solution, solution_idx):
        # [PyGAD documentation] - fitness must be maximization function | https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#preparing-the-fitness-func-parameter
        cheetah = self.cheetah_lab.get_cheetah_behavior(solution)
        score = self.arena.fight(cheetah, 1000) # Number of steps - TODO parametrizable?
        return score
        
    def run_evolution(self, parameters: dict):
        self.ga_instance = pygad.GA(
            fitness_func=self._fitness_func,
            num_genes= self.cheetah_lab.get_genom_length(),
            save_solutions= False, # True,  #UNDO Daniel
            save_best_solutions=True,
            parallel_processing=self.cpu_count,  # 8 #UNDO Daniel
            **parameters)
        self.ga_instance.summary()
        self.ga_instance.run()
    
    def set_cheetah_lab(self, cheetah_lab: CheetahLab):
        self.cheetah_lab = cheetah_lab

    def plot_fitness(self):
        self.ga_instance.plot_fitness()
    
    def save(self, filename):
        self.ga_instance.save(filename)
        
    def load(self, filename):
        self.ga_instance.load(filename)
        
    def get_best_solutions(self):
        # return self.ga_instance.best_solutions
        return self.ga_instance.best_solution()
    
    
    