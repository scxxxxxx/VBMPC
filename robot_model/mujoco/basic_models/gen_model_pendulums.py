import os
from lxml import etree
import numpy as np
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control.utils import io as resources
from robot_env.utils import get_root_path


def make_pendulum(n_bodies, base_file=None, path=None):
    """Generates an xml string defining a swimmer with `n_bodies` bodies."""
    if path is None:
        path = os.path.join(get_root_path(), "robots/basic_models/generated_pendulum{}.xml".format(n_bodies))
    if base_file is None:
        base_file = os.path.join(get_root_path(), "robots/basic_models/pendulum_base.xml")
    if n_bodies <= 0:
        raise ValueError('Invalid number of bodies: {}'.format(n_bodies))
    mjcf = etree.fromstring(resources.GetResource(base_file))
    head_body = mjcf.find('./worldbody/body')
    actuator = etree.SubElement(mjcf, 'actuator')

    parent = head_body
    pos = 0
    lengths = np.random.normal(0.5, 0.1, size=n_bodies)
    lengths = np.clip(lengths, 0.1, 1.0)
    masses = np.random.uniform(0.5, 2.5, size=n_bodies)
    for body_index in range(n_bodies):
        pose = "0 0 {}".format(pos)
        pos = lengths[body_index]
        child = make_body(body_index=body_index, pose=pose, len=pos, mass=masses[body_index])
        # site
        child.append(etree.Element('site', {'name': "pole{}_0".format(body_index+1),
                                            'pos': "0 0 0",
                                            "size": "0.01 0.01"}))
        child.append(etree.Element('site', {'name': "pole{}_1".format(body_index+1),
                                            'pos': "0 0 {}".format(pos),
                                            "size": "0.01 0.01"}))
        if body_index == n_bodies - 1:
            child.append(etree.Element('site', {'name': "tip",
                                                'pos': "0 0 {}".format(pos),
                                                "size": "0.01 0.01"}))

        # joint
        child.append(etree.Element('joint', {'name': "pendulum{}".format(body_index+1),
                                             'axis': "0 1 0",
                                             "pos": "0 0 0",
                                             'type': "hinge"}))

        parent.append(child)
        parent = child
        actuator.append(etree.Element('motor',
                                      name='pendulum{}_motor'.format(body_index+1),
                                      joint='pendulum{}'.format(body_index+1),
                                      forcerange="-50 50",
                                      ctrlrange="-20 20"))

    model = etree.tostring(mjcf, pretty_print=True)
    with open(path, 'wb') as f:
        f.write(model)
    return model


def make_body(body_index, pose='0 0 0', len=1., mass=1.):
    """Generates an xml string defining a single physical body."""
    body = etree.Element('body', name="pole{}".format(body_index+1))
    body.set('pos', pose)
    etree.SubElement(body, 'geom', {'name': "geom_b{}".format(body_index+1),
                                    'fromto': '0 0 0 0 0 {}'.format(len),
                                    'rgba': "0.588 0.909 0.972 1",
                                    'size': "0.045 {}".format(0.5 * len),
                                    'type': "capsule",
                                    "mass": "{}".format(mass)
                                    })

    return body


if __name__ == "__main__":
    x = make_pendulum(3)
