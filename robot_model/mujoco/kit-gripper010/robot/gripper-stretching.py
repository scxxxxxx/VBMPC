#!/usr/bin/env python3
import mujoco_py
import os
import math
import sys
import numpy as np
from matplotlib import pyplot as plt
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./gripper-stretching MODEL-XML")
        sys.exit(0)

    plot = True

    model_path = sys.argv[1]

    print(f"Loading Model: {model_path}")
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)

    motion = {
        "KIT_Gripper_PG_Rotation" : [-np.pi/4, np.pi/4],
        "KIT_Gripper_PG_Close" : [-np.pi/4, 0],
        "KIT_Gripper_Arm_1" : [0, 0.5],
        "KIT_Gripper_Arm_2" : [-0.3, 0.5],
        "KIT_Gripper_Arm_3" : [-1.2, -0.5],
        "KIT_Gripper_Arm_4" : [-0.6, 0.3],
        "KIT_Gripper_Arm_Bit" : [-np.pi, np.pi],
    }

    control_suffix = "_pos"

    actuator_ids = {a: model.actuator_name2id(a + control_suffix) for a in motion}

    targets = {n : [] for n in motion}
    positions = {n : [] for n in motion}
   
    t = 0
    try:
        while True:
            t += 1

            for i, n in enumerate(motion):
                target = motion[n][0] + ((motion[n][1] - motion[n][0]) / 2) + ((motion[n][1] - motion[n][0]) / 2) * np.sin(t / 1000.0)
                sim.data.ctrl[actuator_ids[n]] = target

                targets[n].append(target)
                positions[n].append(sim.data.get_sensor(f"{n}{control_suffix}"))

                tcp = sim.data.get_body_xpos("KIT_Gripper_Arm_Bit_TCP")
                print(tcp)
                
            sim.step()
            viewer.render()
    except (SystemExit, KeyboardInterrupt):
        if plot:
            for i, n in enumerate(motion):
                plt.subplot(len(motion)//3 + 1, 3, i+1)
                plt.plot(targets[n], label=f"{n}{control_suffix} (Target)")
                plt.plot(positions[n], label=f"{n}{control_suffix} (Real)")
                plt.legend()
            plt.show()
        print("Goodbye...")
