from abc import ABC, abstractmethod

import numpy as np


class CheetahLab(ABC):
    @abstractmethod
    def get_genom_length(self) -> int:
        pass
    
    @abstractmethod
    def get_cheetah_behavior(self, genom) -> callable:
        pass
    
class ClassicNNCheetahLab(CheetahLab):
    def __init__(self, nn_architecture: list, activation_function=None):
        self.nn_architecture = nn_architecture
        self.activation_function = self._logsig if activation_function is None else activation_function
    
    def get_genom_length(self) -> int:
        total_count = 0
        arch = self.nn_architecture
        for i in range(len(arch) - 1):
            layer_weights = arch[i] * arch[i + 1]
            layer_biases = arch[i + 1]
            total_count += layer_weights + layer_biases
        return total_count
    
    def get_cheetah_behavior(self, genom) -> callable:
        """Returns a function that outputs the neural netowork output based on initial genom"""
        net = self._genom_to_net(genom)
        behavior = lambda observation: self._net_behaviour(net, observation)
        return behavior
    
    def _genom_to_net(self, genom):
        net = []
        start_idx = 0
        arch = self.nn_architecture
        for i in range(len(arch) - 1):
            layer_length = (arch[i] + 1) * arch[i + 1]
            layer_values = genom[start_idx:start_idx + layer_length]
            net.append(np.array(layer_values).reshape((arch[i] + 1, arch[i + 1])))
            start_idx += layer_length
        return net
    
    @staticmethod
    def _logsig(xi):
        # Problem with too big values
        xi = np.float64(xi)
        cc = np.clip(xi, -88.72, 88.72) #   for float64, use: (-709.78, 709.78)
        # rewrite the formula>> 1 / (1+ exp(-x)) = exp(x) / (1+exp(x))
        return np.exp(cc) / (1 +  np.exp(cc)) # old one: 1 / (1 + np.exp(-xi))

    def _net_behaviour(self, net, observation):
        # Calculates the output of neural network
        hidden = observation[np.newaxis, :]
        for i, layer in enumerate(net):
            hidden_extended = np.hstack((hidden, np.ones((hidden.shape[0], 1))))
            if i == len(net) - 1:
                hidden = np.tanh(hidden_extended @ layer)
            else:
                hidden = self.activation_function(hidden_extended @ layer)
        return hidden
    