import os
import csv
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from tabulate import tabulate
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Callable, Type, List
from marshmallow_dataclass import dataclass
from tqdm import trange, tqdm

import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, SubsetRandomSampler
from robot_utils.py.utils import create_path, load_dataclass, load_dict_from_yaml, save_to_yaml
from robot_utils.torch.torch_utils import get_device, split_indices


@dataclass
class ModelConfig:
    use_gpu:                bool = True
    in_feats:               Optional[int] = None
    out_feats:              Optional[int] = None
    re_init:                Optional[bool] = True
    init_scheme:            str = 'uniform_in_dim'
    require_input_grad:     bool = False


class BaseModel(torch.nn.Module, ABC):
    def __init__(self, config: Union[str, Dict, None] = None, model_path: str = None):
        super(BaseModel, self).__init__()

        self.model_path = create_path(f"./saved_model/{self.name}" if model_path is None else model_path)
        self.config_file = os.path.join(self.model_path, f"{self.name}_model_config.yaml")
        self.params_file = os.path.join(self.model_path, f"{self.name}_model_params.pt")
        # self.config_file = os.path.join(self.model_path, f"{self.name}_model_config.pt")
        if config is None:
            if os.path.isfile(self.config_file):
                config = load_dict_from_yaml(self.config_file)
            else:
                raise RuntimeError("The model config doesn't exist, did you forget to train?")
        elif isinstance(config, str):
            shutil.copy(config, self.config_file)
            config = load_dict_from_yaml(config)
        else:
            save_to_yaml(config, self.config_file)
        self._load_config(config)

        self.device = get_device(self.c.use_gpu)

        self._build_model()
        # self.loss_func = None

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def _load_config(self, config: Dict) -> None:
        self.c = load_dataclass(ModelConfig, config)

    @abstractmethod
    def _build_model(self) -> None:
        raise NotImplementedError

    # @abstractmethod
    # def loss_func(self):
    #     raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs):
        raise NotImplementedError

    def save_model(self, best_param=None, optimizer=None, **kwargs):
        best_param = self.state_dict() if best_param is None else best_param
        if optimizer is not None:
            torch.save({"model": best_param, "optimizer": optimizer.state_dict()}, self.params_file)
        else:
            torch.save(best_param, self.params_file)
        # torch.save(self.c, self.config_file)

    def load_model(self, optimizer=None):
        checkpoint = torch.load(self.params_file)
        if optimizer is not None:
            self.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            self.load_state_dict(checkpoint)

    def print(self, param=True, info=True, shape_only=False, *args, **kwargs):
        if param:
            print("=" * 45, " Printing Parameters of {} ".format(self.__class__.__name__), "=" * 45)
            for name, param in self.state_dict().items():
                if shape_only:
                    ic(name, param.shape)
                else:
                    ic(name, param)
                print("-" * 120)
        if info:
            logging.info(f"Configuration of {self.__class__.__name__}")
            print(tabulate([(k, v) for k, v in self.c.__dict__.items()], headers=["name", "value"], tablefmt="psql"))
            # print("-" * 120)

    def get_param_scheme(self, state_dict: Dict = None):
        if state_dict is None:
            state_dict = self.state_dict()
        scheme = {
            "name": [],
            "shape": [],
            "chunk": []
        }
        for name, param in state_dict.items():
            scheme['name'].append(name)
            scheme['shape'].append(param.shape)
            scheme['chunk'].append(torch.numel(param))
        return scheme

    def get_param_scheme_in_groups(self, state_dict: Dict = None, group_name_list: List = None):
        if state_dict is None:
            state_dict = self.state_dict()
        scheme = dict(groups=group_name_list,
                      schemes=[dict(name=[], shape=[], chunk=[]) for _ in group_name_list],
                      group_chunk=[],
                      names=[], shapes=[], chunks=[])

        for name, param in state_dict.items():
            group_idx = [i for i in range(len(group_name_list)) if group_name_list[i] in name][0]
            scheme['schemes'][group_idx]['name'].append(name)
            scheme['schemes'][group_idx]['shape'].append(param.shape)
            scheme['schemes'][group_idx]['chunk'].append(torch.numel(param))

        for i in range(len(group_name_list)):
            scheme['names'].extend(scheme['schemes'][i]['name'])
            scheme['shapes'].extend(scheme['schemes'][i]['shape'])
            scheme['chunks'].extend(scheme['schemes'][i]['chunk'])
            scheme['group_chunk'].append(sum(scheme['schemes'][i]['chunk']))
        return scheme

    def get_param_vec_with_scheme(self, scheme):
        param_vec = []
        for name in scheme['name']:
            param_vec.append(torch.flatten(self.state_dict()[name]))
        return torch.cat(param_vec)

    def get_param_vec_with_scheme_in_groups(self, scheme, as_list: bool=False):
        param_vec = []
        for name in scheme['names']:
            param_vec.append(torch.flatten(self.state_dict()[name]))
        param_vec = torch.cat(param_vec)
        if as_list:
            return torch.split(param_vec, scheme['group_chunk'])
        else:
            return param_vec

    def load_state_dict_with_scheme(self, scheme, param_vec):
        state_dict = {}
        for name, shape, chunk in zip(scheme['name'], scheme['shape'], torch.split(param_vec, scheme['chunk'])):
            state_dict[name] = chunk.reshape(shape)
        self.load_state_dict(state_dict)

    def load_state_dict_with_scheme_in_groups(self, scheme, param_vec):
        state_dict = {}
        for name, shape, chunk in zip(scheme['names'], scheme['shapes'], torch.split(param_vec, scheme['chunks'])):
            state_dict[name] = chunk.reshape(shape)
        self.load_state_dict(state_dict)

    def get_total_num_params(self, trainable: bool = True):
        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


class ModelFactory:
    """ The model factory class"""

    registry = {}  # type: Dict[str, Type[BaseModel]]
    """ Internal registry for available models """

    @classmethod
    def register(cls, name: str) -> Callable:
        """ Class method to register sub-class to the internal registry.
        Args:
            name (str): The name of the sub-class.
        Returns:
            The sub-class itself.
        """

        def inner_wrapper(wrapped_class: Type[BaseModel]) -> Type[BaseModel]:
            if name in cls.registry:
                logging.warning(f'Model {name} already exists. Will replace it.')
            cls.registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def create_model(cls, name: str, config_file: Union[str, Dict, None] = None, model_path: str = None) -> BaseModel:
        """ Factory method to create the model with the given name and pass parameters as``kwargs``.
        Args:
            name (str): The name of the model to create.
            config_file:
            model_path:
        Returns:
            An instance of the model that is created.
        """
        # logging.info(f"available model factory {cls.registry}")
        logging.info(f"create model with name {name}")
        if name not in cls.registry:
            logging.warning(f'Model {name} does not exist in the registry')
            return None

        model_class = cls.registry[name]
        model = model_class(config_file, model_path)
        return model


@dataclass
class TrainConfig:
    batch_size:             int = 50
    max_epochs:             int = 1000
    learning_rate:          float = 1e-3
    min_learning_rate:      float = 1e-6
    train_data_ratio:       float = 1.0
    gamma:                  float = 0.2
    save_model_steps:       int = 100
    plot_train_process:     bool = False
    sample_batch_data_flag: bool = True
    clip_grad_norm:         float = 10.0
    patience:               int = 10
    cooldown:               int = 10

    # def __str__(self):
    #     return "\n".join([f'{k}: {v}' for (k, v) in self.__dict__.items()])


class BaseTrainer:
    def __init__(self,
                 model:             BaseModel,
                 config:            Optional[Dict] = None,
                 loss_func:         Callable[..., torch.Tensor] = None,
                 train_dataset:     torch.utils.data.Dataset = None,
                 test_dataset:      torch.utils.data.Dataset = None):
        self.model = model
        self.device = model.device

        self.save_dir = self.model.model_path
        self.config_file = os.path.join(self.model.model_path, "{}_train_config.pt".format(self.model.name))
        self._load_config(config)

        if train_dataset:
            self._setup_training(loss_func, train_dataset)
        if test_dataset:
            self._setup_testing(test_dataset)

    def save_config(self):
        torch.save(self.t, self.config_file)

    def _load_config(self, config: Union[str, Dict, None]) -> None:
        self.t = load_dataclass(TrainConfig, config) if config else torch.load(self.config_file)

    def print(self, param=True, info=True, shape_only=False, *args, **kwargs):
        logging.info(f"Training Configuration of {self.__class__.__name__}")
        print(tabulate([(k, v) for k, v in self.t.__dict__.items()], headers=["name", "value"], tablefmt="psql"))

    def get_train_sampler(self, train_dataset):
        """
        train_dataset is the original training dataset
        """
        train_idx, validate_idx = split_indices(train_dataset, train_data_ratio=self.t.train_data_ratio)
        return BatchSampler(SubsetRandomSampler(train_idx), batch_size=self.t.batch_size, drop_last=False), validate_idx

    def _setup_training(self, loss_func, train_dataset):
        train_sampler, validate_idx = self.get_train_sampler(train_dataset)
        self.train_dataloader = DataLoader(train_dataset, sampler=train_sampler)
        if validate_idx and hasattr(self.model, 'validate'):
            self.validate = True
            valid_sampler = BatchSampler(SubsetRandomSampler(validate_idx), batch_size=len(validate_idx), drop_last=False)
            self.validate_dataloader = DataLoader(train_dataset, sampler=valid_sampler)
        else:
            self.validate = False
            self.validate_dataloader = None

        self._setup_loss_fn(loss_func)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.t.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=self.t.patience, cooldown=self.t.cooldown, min_lr=self.t.min_learning_rate,
            factor=self.t.gamma, verbose=False)

        self.monitor = dict(train=[], validate=[])
        self.plot_train_process = self.t.plot_train_process
        self.save_model_steps = self.t.save_model_steps
        self._sample_batch_from_dataloader = self.t.sample_batch_data_flag

        # log
        self.log = tqdm(total=0, position=1, bar_format='{desc}', leave=True)
        self.log_str = ""

    def _setup_loss_fn(self, loss_func):
        if loss_func:
            self.loss_func = loss_func
        elif hasattr(self.model, 'loss_func'):
            self.loss_func = self.model.loss_func
        else:
            self.loss_func = None
            logging.warning("Loss function is not defined when creating the model, \n"
                            "you should implement your own loss in the training process")

    def _setup_testing(self, test_dataset):
        sampler = BatchSampler(SequentialSampler(test_dataset), batch_size=len(test_dataset), drop_last=False)
        self.test_dataloader = DataLoader(test_dataset, sampler=sampler)

    def _append_loss(self, train_loss=None, validate_loss=None):
        if train_loss:
            self.monitor['train'].append(train_loss)
        if validate_loss:
            self.monitor['validate'].append(validate_loss)

    def _get_loss(self, xt, yt, step):
        y, x = self.model(xt.to(self.device))
        if isinstance(y, tuple):
            y, pen_div = y[0], y[1]

        if isinstance(yt, torch.Tensor) and yt.shape[0] == 1:
            yt = yt[0]

        loss = self.loss_func(yt.to(self.device), y, x)
        return loss

    def _train_step(self, data, step):
        xt, yt = data
        self.optimizer.zero_grad(set_to_none=True)
        if xt.shape[0] == 1:
            xt = xt[0]

        loss = self._get_loss(xt, yt, step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.t.clip_grad_norm)
        self.optimizer.step()
        return loss

    def train_model(self):
        self.model.train()
        best_val_loss = 1e5
        best_param = None

        data = next(iter(self.train_dataloader))
        for step in trange(1, self.t.max_epochs+1, desc="iteration", leave=True):
            epoch_loss = 0
            count = 0
            self.log_str = ""
            if self._sample_batch_from_dataloader:
                for i, data in enumerate(self.train_dataloader):
                    loss = self._train_step(data, step)
                    epoch_loss += loss
                    count += 1
            else:
                epoch_loss = self._train_step(data, step)
                count = 1

            train_loss = epoch_loss / count
            self._append_loss(train_loss=train_loss)
            self.scheduler.step(train_loss)
            if step % self.save_model_steps == 0:
                self.model.save_model()
            self.log_str = f"epoch: {step:>7} loss={train_loss:>16.10f}" + self.log_str

            if self.validate:
                # with torch.no_grad:
                validate_loss = self.model.validate(self.validate_dataloader)
                self._append_loss(validate_loss=validate_loss)
                if validate_loss < best_val_loss:
                    best_val_loss = validate_loss
                    best_param = self.model.state_dict()
                self.log_str += " validate_loss: {:>16.10f}".format(validate_loss)

            self.log_str += " lr: {:>10.7f}".format(self.optimizer.param_groups[0]['lr'])
            self.log.set_description_str(self.log_str)

        self.model.save_model(best_param) if best_param else self.model.save_model()
        self._plot_train_process()
        self.save_config()

    def _plot_train_process(self):
        if self.plot_train_process:
            plt.figure()
            loss = np.array(self.monitor['train'])
            x = np.arange(len(loss))
            plt.plot(x, loss, 'b', label="train_loss")
            if self.validate:
                validate_loss = np.array(self.monitor['validate'])
                plt.plot(x, validate_loss, 'r', label="validate_loss")
            plt.savefig(os.path.join(self.model.model_path, 'training_process.png'), dpi=300)

    def test_model(self, *args, **kwargs):
        self.model.eval()
        xt, yt = next(iter(self.test_dataloader))
        xt, yt = xt[0], yt[0]
        yt = yt.detach().cpu().numpy()

        ic(self.device, xt.device)
        y, x = self.model(xt.to(self.device))
        if isinstance(y, tuple):
            y = y[0]

        y = y.detach().cpu().numpy()

        saved_file = os.path.join(self.model.model_path, 'prediction_torch.csv')
        saved_figure = os.path.join(self.model.model_path, 'prediction_torch.png')
        with open(saved_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(y)

        # t = np.arange(y.shape[0])
        t = xt.detach().cpu().numpy()
        n_plots = y.shape[1]
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.000)
        for j in range(n_plots):
            ax = plt.subplot(n_plots, 1, j + 1)
            ic(type(yt))
            ax.plot(t, yt[:, j], 'k-', label='GT')
            ax.plot(t, y[:, j], 'r-.', label='Prediction')
            ax.legend()

            err = yt[:, j] - y[:, j]
            mse = np.mean(err ** 2)
            max_err = np.max(np.abs(err))
            logging.info(f"the testing error: mse {mse}, max error: {max_err}")

        plt.tight_layout()
        plt.savefig(saved_figure, dpi=300)
        plt.show()

