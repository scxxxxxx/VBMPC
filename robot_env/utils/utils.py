import os

from pybullet_utils import bullet_client as bulllet_client, urdfEditor as urdfEditor

from robot_utils.py.utils import create_path


def get_root_path():
    module_path = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(module_path, '../../'))


def get_robot_path(relative_path: str = "", simulator: str = "pybullet"):
    """convert the relative path to the robot xml file to absolute path"""
    return os.path.join(get_root_path(), "robot_model", simulator, relative_path)


def get_basic_model_path():
    return os.path.join(get_root_path(), "robots/basic_models")


def get_world_path():
    return os.path.join(get_root_path(), "robots/worlds")


def convert_mjcf_to_urdf(input_mjcf, output_path):
    """
    Convert MuJoCo mjcf to URDF format with pybullet.
    Args:
        input_mjcf: str input path of mjcf file.
        output_path: str output directory path of urdf.

    Returns:

    """

    client = bulllet_client.BulletClient()
    objs = client.loadMJCF(input_mjcf, flags=client.URDF_USE_IMPLICIT_CYLINDER)

    create_path(output_path)

    for obj in objs:
        humanoid = objs[obj]
        ue = urdfEditor.UrdfEditor()
        ue.initializeFromBulletBody(humanoid, client._client)
        robot_name = str(client.getBodyInfo(obj)[1], 'utf-8')
        part_name = str(client.getBodyInfo(obj)[0], 'utf-8')
        save_visuals = True
        outpath = os.path.join(
            output_path, "{}_{}.urdf".format(robot_name, part_name))
        ue.saveUrdf(outpath, save_visuals)