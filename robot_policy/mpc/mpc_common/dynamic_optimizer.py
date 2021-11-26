import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable
from robot_utils.math.math_tools import wrap_value_to_pi
from robot_policy.rl.common import ReplayBuffer


class ModelOptimizer(object):
    def __init__(self, model, replay_buffer, lr=1e-2, eps=1e-1, lam=0.95):

        # reference the model and buffer
        self.model          = model
        self.replay_buffer  = replay_buffer
        # set the model optimizer
        self.model_optimizer  = optim.Adam(self.model.parameters(), lr=lr)
        # logger
        self._eps = eps
        self._lam = lam
        self.log = {'loss' : [], 'rew_loss': []}

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

    def update_model(self, batch_size, mini_iter=1):

        for k in range(mini_iter):
            states, actions, rewards, next_states, next_action, done = self.replay_buffer.sample(batch_size)

            states = torch.FloatTensor(states).to(self.device)
            states.requires_grad = True
            next_states = torch.FloatTensor(next_states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            next_action = torch.FloatTensor(next_action).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            done    = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

            pred_mean, pred_std, pred_rew = self.model(states, actions)

            state_dist = Normal(pred_mean, pred_std)

            next_vals = self.model.reward_fun(torch.cat([next_states, next_action], axis=1))

            rew_loss = torch.mean(torch.pow((rewards+self._lam*(1-done)*next_vals).detach() - pred_rew,2))

            model_loss = -torch.mean(state_dist.log_prob(next_states))

            loss = 0.5 * rew_loss + model_loss

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

        self.log['loss'].append(loss.item())
        self.log['rew_loss'].append(rew_loss.item())


class EqlModelOptimizer(object):
    def __init__(self, model, replay_buffer, lr=1e-2, eps=1e-1, lam=0.95, max_steps=6000):
        self.model = model
        self.replay_buffer = replay_buffer
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self._eps = eps
        self._lam = lam
        self.log = {'loss': [], 'rew_loss': []}

        self.device = self.model.device

        self.t1 = min(0.25 * max_steps, 300)
        self.t2 = min(0.9 * max_steps, 600)

    def update_model(self, step, batch_size, mini_iter=1):
        for k in range(mini_iter):
            states, actions, rewards, next_states, next_action, done = self.replay_buffer.sample(batch_size)

            states      = torch.tensor(states,      device=self.device, requires_grad=True).float()
            next_states = torch.tensor(next_states, device=self.device).float()
            actions     = torch.tensor(actions,     device=self.device).float()
            next_action = torch.tensor(next_action, device=self.device).float()
            rewards     = torch.tensor(rewards,     device=self.device).float().unsqueeze(1)
            done        = torch.tensor(done,        device=self.device).float().unsqueeze(1)

            pred_mean, pred_std, pred_rew, pen_div = self.model(states, actions)
            if self.t1 < step < self.t2:
                l1_norm_loss = self.model.mu.c.lamb * self.model.mu.l1_norm()
            else:
                l1_norm_loss = Variable(torch.tensor([0.], device=self.device).float())

            state_dist = Normal(pred_mean, pred_std)

            next_val, _ = self.model.reward_fun(torch.cat([next_states, next_action], dim=1))
            rew_loss = ((rewards + self._lam * (1 - done) * next_val).detach() - pred_rew).pow(2).mean()

            model_loss = -torch.mean(state_dist.log_prob(next_states))

            # ic(rew_loss, model_loss, l1_norm_loss)
            loss = rew_loss + model_loss + 10 * l1_norm_loss

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

        if step > self.t2 and self.model.mu.c.train.pruning:
            self.model.mu.pruning(self.model.mu.c.param_threshold)

        self.log['loss'].append(loss.item())
        self.log['rew_loss'].append(rew_loss.item())


class MDNModelOptimizer(object):

    def __init__(self, model, replay_buffer, lr=1e-2):
        self.model = model
        self.replay_buffer = replay_buffer

        self.model_optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.log = {'loss': [], 'rew_loss': []}

    def update_model(self, batch_size, mini_iter=1):

        for k in range(mini_iter):
            states, actions, rewards, next_states, done = self.replay_buffer.sample(batch_size)

            states = torch.FloatTensor(states)
            next_states = torch.FloatTensor(next_states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            done = torch.FloatTensor(np.float32(done)).unsqueeze(1)

            log_probs, pred_rewards = self.model(states, actions, next_states)

            next_value = self.model.predict_reward(next_states)

            rew_loss = torch.mean(torch.pow((rewards+(1-done)*0.99*next_value).detach()-pred_rewards,2))
            model_loss = -torch.mean(log_probs)

            loss = 0.5 * rew_loss + model_loss

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

        self.log['loss'].append(loss.item())
        self.log['rew_loss'].append(rew_loss.item())


class DynamicsModelOptimizer(object):
    """ Optimizer for the dynamics model in MPPIPolicy """
    def __init__(
            self,
            dynamics:           nn.Module,
            replay_buffer:      ReplayBuffer,
            lr:                 float = 1e-2,
            max_steps:          int = 6000,
            bootstrap_steps:    int = 100,
            l1_norm:            bool = False,
            prune:              bool = False
    ):
        self.dynamics = dynamics
        self.replay_buffer = replay_buffer
        self.dynamics_optimizer = optim.Adam(self.dynamics.parameters(), lr=lr)
        self.l1_norm = l1_norm
        self.prune = prune
        self.bootstrap_steps = bootstrap_steps
        self.device = self.dynamics.device

        self.log = {'loss': [], 'rew_loss': []}

        self.t1 = min(0.25 * max_steps, 300)
        self.t2 = min(0.9 * max_steps, 600)

    def update_model(self, step, batch_size, mini_iter=1):
        if self.replay_buffer.idx < batch_size:
            mini_iter = self.bootstrap_steps
            batch_size = self.replay_buffer.idx
            t1, t2 = 0.25 * self.bootstrap_steps, 0.9 * self.bootstrap_steps
            use_total_step = False
        else:
            t1, t2 = self.t1, self.t2
            use_total_step = True

        s = self.replay_buffer.sample(batch_size)

        for k in range(mini_iter):
            step_ = step if use_total_step else k
            self.dynamics_optimizer.zero_grad()

            pred_next_state = self.dynamics.step_state(s.obs, s.act)

            if t1 < step_ < t2 and self.l1_norm:
                l1_norm_loss = self.dynamics.mu.c.lamb * self.dynamics.mu.l1_norm()
            else:
                l1_norm_loss = Variable(torch.tensor([0.], device=self.device).float())

            # state_dist = Normal(pred_next_state, pred_std)
            # model_loss = -torch.mean(state_dist.log_prob(next_states))
            err = s.next_obs - pred_next_state
            err[:, 0] = wrap_value_to_pi(err[:, 0])
            model_loss = err.pow(2).mean()

            # next_val = self.model.forward_value(torch.cat([next_states, next_action], dim=1))
            # rew_loss = ((rewards + self._lam * (1 - done) * next_val).detach() - pred_value).pow(2).mean()

            # ic(rew_loss, model_loss, l1_norm_loss)
            # loss = rew_loss + model_loss + 10 * l1_norm_loss
            loss = model_loss + 10 * l1_norm_loss
            loss.backward()
            self.dynamics_optimizer.step()

            if step_ > t2 and self.prune:
                self.dynamics.mu.pruning(self.dynamics.mu.c.param_threshold)

        self.log['loss'].append(loss.item())
        # self.log['rew_loss'].append(rew_loss.item())


# TODO unify the above and below, to configure whether to optimize reward
class DynamicsRewardModelOptimizer(object):
    def __init__(
            self,
            dynamics:           nn.Module,
            # critic:             nn.Module,
            replay_buffer:      ReplayBuffer,
            # dyn_lr:             float = 1e-2,
            lr:                 float = 1e-2,
            # critic_lr:          float = 1e-2,
            max_steps:          int = 6000,
            bootstrap_steps:    int = 100,
            l1_norm:            bool = False,
            prune:              bool = False,
            eps:                float = 0.,
            lam:                float = 0.
    ):
        self.dynamics = dynamics
        #self.critic = twin_env.rew
        self.replay_buffer = replay_buffer
        self.dynamics_optimizer = optim.Adam(self.dynamics.parameters(), lr=lr)
        self.l1_norm = l1_norm
        self.prune = prune
        self.bootstrap_steps = bootstrap_steps
        #self.device = self.dynamics.device

        self._eps = eps
        self._lam = lam

        self.log = {'loss': [], 'rew_loss': []}

        self.t1 = min(0.25 * max_steps, 300)
        self.t2 = min(0.9 * max_steps, 600)

    def add_dynamics(self, dynamics):
        self.dynamics = dynamics

    def add_critic(self, critic):
        self.critic = critic

    def update_model(self, step, batch_size, mini_iter=1):
        if self.replay_buffer.idx < batch_size:
            mini_iter = self.bootstrap_steps
            batch_size = self.replay_buffer.idx
            t1, t2 = 0.25 * self.bootstrap_steps, 0.9 * self.bootstrap_steps
            use_total_step = False
        else:
            t1, t2 = self.t1, self.t2
            use_total_step = True

        s = self.replay_buffer.sample(batch_size)

        for k in range(mini_iter):
            step_ = step if use_total_step else k
            self.dynamics_optimizer.zero_grad()
            # self.critic_optimizer.zero_grad()

            pred_next_obs, pred_rew, _, _ = self.dynamics.step(s.obs, s.act)
            # pred_value = self.critic(s.obs, s.act)

            # ic(self.l1_norm)
            # if t1 < step_ < t2 and self.l1_norm:
            #     l1_norm_loss = self.dynamics.mu.c.lamb * self.dynamics.mu.l1_norm()
            # else:
            #     l1_norm_loss = Variable(torch.tensor([0.], device=self.device).float())
            l1_norm_loss = 0.

            # state_dist = Normal(pred_next_state, pred_std)
            # model_loss = -torch.mean(state_dist.log_prob(next_states))
            err = s.next_obs - pred_next_obs
            err[:, 0] = wrap_value_to_pi(err[:, 0])  # TODO, this might be wrong for other type of joints
            dynamics_loss = err.pow(2).mean() + 10 * l1_norm_loss

            # next_val = self.critic(s.next_obs, s.next_act)
            # value_loss = ((s.rew + self._lam * (1 - s.done) * next_val).detach() - pred_value).pow(2).mean()
            dynamics_loss += (pred_rew - s.rew).pow(2).mean()
            # ic(rew_loss, model_loss, l1_norm_loss)
            # loss = rew_loss + model_loss + 10 * l1_norm_loss

            dynamics_loss.backward()
            # value_loss.backward()
            self.dynamics_optimizer.step()
            # self.critic_optimizer.step()

            # if step_ > t2 and self.prune:
            #     self.dynamics.mu.pruning(self.dynamics.mu.c.param_threshold)
            torch.cuda.empty_cache()

        self.log['loss'].append(dynamics_loss.item())
        # self.log['rew_loss'].append(value_loss.item())

    # def update_model(self, step, batch_size, mini_iter=1):
    #     if self.replay_buffer.idx < batch_size:
    #         mini_iter = self.bootstrap_steps
    #         batch_size = self.replay_buffer.idx
    #         t1, t2 = 0.25 * self.bootstrap_steps, 0.9 * self.bootstrap_steps
    #         use_total_step = False
    #     else:
    #         t1, t2 = self.t1, self.t2
    #         use_total_step = True
    #
    #     s = self.replay_buffer.sample_ext(batch_size)
    #
    #     for k in range(mini_iter):
    #         step_ = step if use_total_step else k
    #         self.dynamics_optimizer.zero_grad()
    #         self.critic_optimizer.zero_grad()
    #
    #         pred_next_obs, pred_value, _, _ = self.twin_env.step(s.obs, s.act)
    #
    #         dynamics_loss = self.loss_func(s.next_obs, pred_next_obs)
    #
    #         _, next_val, _, _ = self.twin_env.step(s.next_obs, s.next_act)
    #         value_target = (s.rew + self._lam * next_val).detach()
    #         value_loss = self.loss_func(pred_value, value_target)
    #
    #         dynamics_loss.backward()
    #         value_loss.backward()
    #         self.dynamics_optimizer.step()
    #         self.critic_optimizer.step()
    #
    #
    #         torch.cuda.empty_cache()
    #
    #     self.log['loss'].append(dynamics_loss.item())
    #     self.log['rew_loss'].append(value_loss.item())
class CriticOptimizer(object):
    def __init__(
            self,
            critic:           nn.Module,
            replay_buffer:      ReplayBuffer,
            lr:                 float = 1e-2,
            max_steps:          int = 6000,
            lam:                float = 0.95,
            bootstrap_steps:    int = 100,

    ):
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.bootstrap_steps = bootstrap_steps
        self.lam = lam
        #self.device = self.dynamics.device

    def update_model(self, batch_size, mini_iter=1):
        if self.replay_buffer.idx < batch_size:
            mini_iter = self.bootstrap_steps
            batch_size = self.replay_buffer.idx

        s = self.replay_buffer.sample(batch_size)

        for k in range(mini_iter):
            self.critic_optimizer.zero_grad()

            pred_value = self.critic(s.obs)
            next_val = self.critic(s.next_obs)
            value_target = (s.rew + self.lam * next_val)

            value_loss = self.loss_func(value_target, pred_value)

            value_loss.backward()
            self.critic_optimizer.step()


