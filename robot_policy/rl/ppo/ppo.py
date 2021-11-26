import gym
import numpy as np
import torch
import logging
from typing import Union, Dict, Optional
from marshmallow_dataclass import dataclass
from torch.nn import functional as F

from robot_policy.rl.common.on_policy_agent import OnPolicyAgent, OnPolicyConfig
from robot_policy.utils import get_schedule_fn, Schedule
from robot_utils.py.utils import load_dataclass_from_dict


@dataclass
class PPOConfig(OnPolicyConfig):
    clip_range:     float = 0.2
    clip_range_vf:  Union[float, None] = None
    target_kl:      Union[float, None] = None


class PPO(OnPolicyAgent):
    def __init__(self, config: Optional[Dict], model_path: str):
        super(PPO, self).__init__(config, model_path)
        self.clip_range     = self.c.clip_range
        self.clip_range_vf  = self.c.clip_range_vf
        self.target_kl      = self.c.target_kl

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _load_config(self, config):
        self.c = load_dataclass_from_dict(PPOConfig, config)

    def _update_clip_range(self):
        clip_range = self.clip_range(1 - self.progress)
        # logging.info(clip_range)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(1 - self.progress)
            return clip_range, clip_range_vf
        else:
            return clip_range, None

    def train(self):
        clip_range, clip_range_vf = self._update_clip_range()

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # train for multiple epochs per rollout collection. Early stop when kl-divergence exceed threshold
        for epoch in range(self.c.train_mini_epochs):
            approx_kl_divs = []
            epoch_loss = []
            for data in self.buffer.get(self.c.batch_size):
                obs, act, val, logp, adv, ret = data.obs, data.act, data.val, data.logp, data.adv, data.ret

                if isinstance(self.env.action_space, gym.spaces.Discrete):
                    act = act.long().flatten()

                if self.c.use_sde:
                    self.policy.reset_noise(self.c.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(obs, act)
                values = values.flatten()

                # COMM: Normalize advantage
                advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

                # ratio between old (logp) and new policy (log_prob), should be one at the first iteration
                ratio = torch.exp(log_prob - logp)

                # ic(adv, adv.mean(), adv.std())
                # ic(advantages, ratio)

                # COMM: clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                # ic(policy_loss)

                # COMM: clipping loss
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                # ic(clip_fraction)

                if clip_range_vf is None:
                    # No clipping
                    # ic("no clip vf")
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = val + torch.clamp(values - val, -clip_range_vf, clip_range_vf)
                # ic(values_pred, ret)
                # COMM: Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(ret, values_pred)
                value_losses.append(value_loss.item())
                # ic(value_loss)

                # COMM: Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                # ic(entropy_loss)

                # COMM: Total Loss
                loss = policy_loss + self.c.ent_coef * entropy_loss + self.c.val_coef * value_loss
                epoch_loss.append(loss.item())
                # ic(loss)
                # exit()

                self.optimize_step(loss, clipping_norm=True, retain_graph=False)

                approx_kl_divs.append(torch.mean(logp - log_prob).detach().cpu().numpy())

            all_kl_divs.append(np.mean(approx_kl_divs))
            early_stop = self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl

            log_str = f"Policy Gradient loss: {pg_losses[-1]:>15.8f} | " \
                      f"value loss: {value_losses[-1]:>15.8f} | " \
                      f"entropy loss: {entropy_losses[-1]:>15.8f} | " \
                      f"total loss: {np.mean(epoch_loss):>15.8f} | " \
                      f"e_stop: {early_stop}"
                      # f"std: {torch.exp(self.policy.log_std).mean().item():>10.5f} | " +

            self.log_bar.set_description_str(log_str)

            if early_stop:
                # print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        # self._n_updates += self.n_epochs
        # explained_var = explained_variance(self.buffer.values.flatten(), self.buffer.returns.flatten())



# class PPO_(OnPolicyAgent):
#     def __init__(self, c: AgentConfig, t: PPOTrain, model_path: str = None):
#         super(PPO_, self).__init__(c, t, model_path)
#
#         self.clip_range = self.t.clip_range
#         self.clip_range_vf = self.t.clip_range_vf
#         self.target_kl = self.t.target_kl
#
#         self._setup_model()
#
#     def _setup_model(self):
#         super(PPO, self)._setup_model()
#         self.clip_range = get_schedule_fn(self.clip_range)
#         if self.clip_range_vf is not None:
#             if isinstance(self.clip_range_vf, (float, int)):
#                 assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
#
#             self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
#
#     def _setup_lr_scheduler(self):
#         pass
#
#     def _update_clip_range(self):
#         clip_range = self.clip_range(1 - self.progress)
#         # Optional: clip range for the value function
#         if self.clip_range_vf is not None:
#             clip_range_vf = self.clip_range_vf(1 - self.progress)
#             return clip_range, clip_range_vf
#         else:
#             return clip_range, None
#
#     def train(self):
#         clip_range, clip_range_vf = self._update_clip_range()
#
#         entropy_losses, all_kl_divs = [], []
#         pg_losses, value_losses = [], []
#         clip_fractions = []
#
#         # train for multiple epochs per rollout collection. Early stop when kl-divergence exceed threshold
#         for epoch in range(self.t.epochs):
#             approx_kl_divs = []
#             epoch_loss = []
#             for data in self.buffer.get(self.t.batch_size):
#                 obs, act, val, logp, adv, ret = data.obs, data.act, data.val, data.logp, data.adv, data.ret
#
#                 if isinstance(self.env.action_space, gym.spaces.Discrete):
#                     act = act.long().flatten()
#
#                 if self.t.use_sde:
#                     self.policy.reset_noise(self.t.batch_size)
#
#                 values, log_prob, entropy = self.policy.evaluate_actions(obs, act)
#                 values = values.flatten()
#
#                 # COMM: Normalize advantage
#                 advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
#
#                 # ratio between old and new policy, should be one at the first iteration
#                 ratio = torch.exp(log_prob - logp)
#
#                 # ic(adv, adv.mean(), adv.std())
#                 # ic(advantages, ratio)
#
#                 # COMM: clipped surrogate loss
#                 policy_loss_1 = advantages * ratio
#                 policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
#                 policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
#                 # ic(policy_loss)
#
#                 # COMM: clipping loss
#                 pg_losses.append(policy_loss.item())
#                 clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
#                 clip_fractions.append(clip_fraction)
#                 # ic(clip_fraction)
#
#                 if clip_range_vf is None:
#                     # No clipping
#                     # ic("no clip vf")
#                     values_pred = values
#                 else:
#                     # Clip the different between old and new value
#                     # NOTE: this depends on the reward scaling
#                     values_pred = val + torch.clamp(values - val, -clip_range_vf, clip_range_vf)
#                 # ic(values_pred, ret)
#                 # COMM: Value loss using the TD(gae_lambda) target
#                 value_loss = F.mse_loss(ret, values_pred)
#                 value_losses.append(value_loss.item())
#                 # ic(value_loss)
#
#                 # COMM: Entropy loss favor exploration
#                 if entropy is None:
#                     # Approximate entropy when no analytical form
#                     entropy_loss = -torch.mean(-log_prob)
#                 else:
#                     entropy_loss = -torch.mean(entropy)
#                 entropy_losses.append(entropy_loss.item())
#                 # ic(entropy_loss)
#
#                 # COMM: Total Loss
#                 loss = policy_loss + self.t.ent_coef * entropy_loss + self.t.val_coef * value_loss
#                 epoch_loss.append(loss)
#                 # ic(loss)
#                 # exit()
#
#                 self.optimize_step(loss, clipping_norm=True, retain_graph=False)
#
#                 approx_kl_divs.append(torch.mean(data.logp - log_prob).detach().cpu().numpy())
#
#             all_kl_divs.append(np.mean(approx_kl_divs))
#             logging.info("Progress: {:>3.2f}, pg loss: {:>25.8f}, value loss: {:>25.8f}, loss: {:>25.8f}".format(
#                 self.progress, pg_losses[-1], value_losses[-1], sum(epoch_loss)/len(epoch_loss)))
#
#             if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
#                 print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
#                 break
#
#         # self._n_updates += self.n_epochs
#         # explained_var = explained_variance(self.buffer.values.flatten(), self.buffer.returns.flatten())
