import numpy as np
import torch
from torch import nn
import torch.nn.functional as fnc
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, input, hidden, output):
        super(NeuralNetwork, self).__init__()
        self.f_connected1 = nn.Linear(input, hidden[0])
        self.f_connected2 = nn.Linear(hidden[0], hidden[1])
        self.out_connected = nn.Linear(hidden[1], output)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = fnc.relu(self.f_connected1(tensor))
        output = fnc.relu(self.f_connected2(output))
        output = self.out_connected(output)
        return output

    def predict(self, tensor):
        prediction = self.forward(tensor)
        return torch.argmax(prediction)
        
