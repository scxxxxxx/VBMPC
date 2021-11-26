import gym
import numpy as np
import torch
import logging
from typing import Union, Dict, Optional
from marshmallow_dataclass import dataclass
from torch.nn import functional as F

from robot_policy.rl.common.off_policy_agent import OffPolicyConfig, OffPolicyAgent
from robot_policy.utils import get_schedule_fn, Schedule
from robot_utils.py.utils import load_dataclass_from_dict


from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F


class SAC(OffPolicyAgent):
    def __init__(self, config: Optional[Dict], model_path: str):
        super().__init__(config, model_path)

    def _load_config(self, config: Union[str, Dict, None]) -> None:
        pass

    def _init_agent(self, config: Dict):
        super()._init_agent(config)

    def _get_action(self) -> Tuple[np.ndarray, np.ndarray]:
        return super()._get_action()

    def save_buffer(self) -> None:
        super().save_buffer()

    def load_buffer(self) -> None:
        super().load_buffer()

    def rollout(self, render: bool = False):
        return super().rollout(render)

    def train(self):
        pass

    def learn(self):
        super().learn()
