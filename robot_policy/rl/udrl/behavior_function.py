import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
Tensor = torch.Tensor


class Behavior(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_size, command_scale=[1, 1], device='cpu', learning_rate=1e-3,
                 structure=None):
        super().__init__()

        self.command_scale = torch.tensor(command_scale, dtype=torch.float, device=device)
        self.state_function = nn.Sequential(nn.Linear(s_dim, 64),
                                            nn.Tanh())

        self.command_function = nn.Sequential(nn.Linear(2, 64),
                                              nn.Sigmoid())

        self.shared_function = nn.Sequential(nn.Linear(64, 128),
                                             nn.ReLU(),
                                             # nn.Dropout(0.2),
                                             nn.Linear(128, 128),
                                             nn.ReLU(),
                                             # nn.Dropout(0.2),
                                             nn.Linear(128, 128),
                                             nn.ReLU(),
                                             nn.Linear(128, a_dim))

        self.to(device)
        self.optim = Adam(self.parameters(), learning_rate)

    def forward(self, state: Tensor, command: Tensor) -> Tensor:
        '''
        Forward pass takes state and command as input and gives action logits
        '''
        state_output = self.state_function(state)
        command_output = self.command_function(command * self.command_scale)
        lattent = torch.mul(state_output, command_output)
        return self.shared_function(lattent)

    def action(self, state: Tensor, command: Tensor) -> Tensor:
        '''
        takes state and command as input and samples actions from categorical distribution
        '''
        logits = self.forward(state, command)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.sample().item()

    def greedy_action(self, state: Tensor, command: Tensor) -> np.ndarray:
        '''
        takes state and command as input and return the greedy action
        '''
        logits = self.forward(state, command)
        probs = F.softmax(logits, dim=-1)
        return np.argmax(probs.detach().cpu().numpy())

    def save(self, filename: str):
        torch.save(self.state_dict(), filename)

    def load(self, filename: str):
        self.load_state_dict(torch.load(filename))


