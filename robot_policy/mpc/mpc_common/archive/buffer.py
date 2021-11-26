import random
import numpy as np


# TODO make replay buffer a dataset, the same as in rl replay buffer, maybe can reuse it
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, next_action, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, next_action, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, next_action, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, next_action, done

    def sample_all(self, ordered=False):
        if ordered:
            state, action, reward, next_state, next_action, done = map(np.stack, zip(*self.buffer))
            return state, action, reward, next_state, next_action, done
        else:
            return self.sample(self.position)

    def __len__(self):
        return len(self.buffer)
