import os
import shutil
import numpy as np
from pathlib import Path
import yaml
import marshmallow
from marshmallow_dataclass import class_schema
# from dataclasses import asdict
from typing import TypeVar, Dict, Union
T = TypeVar('T')


class AttributeAccess(object):
    def __init__(self, d): self.__dict__ = d


def get_root_path():
    module_path = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(module_path, '..'))


def get_home():
    return Path.home()


def get_example_path():
    return os.path.join(get_root_path(), "examples")


def get_datasets_path():
    """
    get the control dataset path
    Returns: the control dataset path

    """
    module_path = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(module_path, '..', 'eql_datasets'))


def create_path(path, remove_existing=False):
    """
    check if dir exist, otherwise create new dir
    Args:
        path: the absolute path to be created. If already exists, don't do anything

    Returns:

    """
    p = Path(path)
    if remove_existing and p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return str(p.resolve(path))


def sample_uniform_joint(config, kc):
    """
    sample a joint configuration from a uniform distribution in the joint controlable range
    Args:
        config: joint limit configuration of the robot
        kc: the kinematic chain

    Returns: sampled joint configuration

    """
    qpos = np.zeros(len(config.position_limit))
    for i in range(len(config.position_limit)):
        limits = config.position_limit[kc.joint_list[i]]
        qpos[i] = np.random.uniform(limits[0], limits[1], 1)
    return qpos


def sample_gaussian_joint(config, kc):
    """
    sample a joint configuration from a gaussian distribution, whose mean is the center of the joint range and
    the scale is the half range.
    Args:
        config: joint limit configuration of the robot
        kc: the kinematic chain

    Returns: sampled joint configuration

    """
    qpos = np.zeros(len(config.position_limit))
    for i in range(len(config.position_limit)):
        limits = config.position_limit[kc.joint_list[i]]
        qpos[i] = 0.5 * (np.random.normal(size=1) * (limits[1] - limits[0]) + np.sum(limits))
    return qpos


# def load_dict_from_yaml(filename):
#     return yaml.load(open(filename), Loader=yaml.CLoader)
def load_dict_from_yaml(filename):
    return yaml.load(open(filename), Loader=yaml.SafeLoader)


def save_to_yaml(data, filename, flush=False):
    with open(filename, 'w') as outfile:
        yaml.safe_dump(data, outfile, encoding='utf-8', allow_unicode=True, default_flow_style=False)
        if flush:
            outfile.flush()


def load_dataclass_from_yaml(dataclass_type: T, filename: str, include_unknown: bool = False) -> T:
    loaded = load_dict_from_yaml(filename)
    return load_dataclass_from_dict(dataclass_type, loaded, include_unknown)


def load_dataclass_from_dict(dataclass_type: T, dic: Dict, include_unknown: bool = False) -> T:
    schema = class_schema(dataclass_type)
    include_schema = marshmallow.INCLUDE if include_unknown else marshmallow.EXCLUDE
    return schema().load(dic, unknown=include_schema)


def dump_data_to_dict(dataclass_type: T, data):
    schema = class_schema(dataclass_type)
    return schema().dump(data)


def dump_data_to_yaml(dataclass_type: T, data, filename: str):
    save_to_yaml(dump_data_to_dict(dataclass_type, data), filename)


# Note: the following code is expected to do the same, but this doesn't resolve the problem of user defined Enum, e.g.
# def dump_data_to_dict(data):
#     return asdict(data)
# def dump_data_to_yaml(data, filename: str):
#     save_to_yaml(asdict(data), filename)


def load_dataclass(dataclass_type: T, config: Union[str, Dict], include_unknown: bool = False) -> T:
    if isinstance(config, str):
        return load_dataclass_from_yaml(dataclass_type, config, include_unknown)
    elif isinstance(config, Dict):
        return load_dataclass_from_dict(dataclass_type, config, include_unknown)
    else:
        raise RuntimeError(f"type of config {type(config)} is not supported, please use yaml configuration"
                           f"file or configuration dictionaries.")



