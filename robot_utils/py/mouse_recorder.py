import os
import cv2
import click
import numpy as np
import matplotlib.pyplot as plt
from time import time
from random import random
from icecream import ic
from tabulate import tabulate

from robot_utils.py.utils import get_datasets_path, create_path, get_home


class MouseRecorder:
    def __init__(self, path):
        self.path = path
        self.is_mouse_lb_pressed = False
        self.circle_color = (0, 0, 0)
        self.circle_radius = 5
        self.last_point = (0, 0)

        self.img = np.ones((512, 512, 3), np.uint8) * 255               # drawing workspace
        self.color_preview_img = np.zeros((100, 100, 3), np.uint8)      # color preview
        self.radius_preview = np.ones((100, 100, 3), np.uint8) * 255    # thickness preview

        self.trajectories = []
        self.trajectory = []
        self.time = 0

        cv2.namedWindow('Image')            # drawing window
        cv2.namedWindow('color_Preview')    # color preview window
        cv2.namedWindow('Radius_Preview')   # thickness preview window

        cv2.createTrackbar('Channel_R', 'Image', 0, 255, self.update_circle_color)
        cv2.createTrackbar('Channel_G', 'Image', 0, 255, self.update_circle_color)
        cv2.createTrackbar('Channel_B', 'Image', 0, 255, self.update_circle_color)
        cv2.createTrackbar('Circle_Radius', 'Image', 1, 20, self.update_circle_radius)

        cv2.setMouseCallback('Image', self.draw_circle)

        print("click and drag mouse to draw!")
        self.key_mapping = {"q": "quit",
                            "a": "append trajectory",
                            "s": "save all trajectories to file",
                            "c": "clear and reset data collection"}
        print(tabulate([(k, v) for k, v in self.key_mapping.items()], headers=["key", "functionality"], tablefmt="github"))

    def draw_circle(self, event, x, y, flags, param):
        """
        the coordinates is relative to the window
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_mouse_lb_pressed = True
            self.time = time()
            cv2.circle(self.img, (x, y), int(self.circle_radius / 2), self.circle_color, -1)
            self.last_point = (x, y)
            self.trajectory.append([0.0, x, y])
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_mouse_lb_pressed = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_mouse_lb_pressed:
                cv2.line(self.img, pt1=self.last_point, pt2=(x, y), color=self.circle_color, thickness=self.circle_radius)
                self.last_point = (x, y)
                self.trajectory.append([time() - self.time, x, y])

    def update_circle_color(self, x):
        r = cv2.getTrackbarPos('Channel_R', 'Image')
        g = cv2.getTrackbarPos('Channel_G', 'Image')
        b = cv2.getTrackbarPos('Channel_B', 'Image')

        self.circle_color = (b, g, r)
        self.color_preview_img[:] = self.circle_color

    def random_circle_color(self):
        self.circle_color = plt.cm.jet(random())[:3]
        self.color_preview_img[:] = self.circle_color

    def update_circle_radius(self, x):
        self.circle_radius = cv2.getTrackbarPos('Circle_Radius', 'Image')
        self.radius_preview[:] = (255, 255, 255)
        cv2.circle(self.radius_preview, center=(50, 50), radius=int(self.circle_radius / 2), color=(0, 0, 0), thickness=-1)

    def start(self):
        while True:
            cv2.imshow('color_Preview', self.color_preview_img)
            cv2.imshow('Radius_Preview', self.radius_preview)
            cv2.imshow('Image', self.img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.img = np.ones((512, 512, 3), np.uint8) * 255
                self.trajectory = []
                self.trajectories = []
            elif key == ord('a'):
                traj = np.vstack(self.trajectory)
                vel = traj[1:, :] - traj[:-1, :]
                vel[:, 1:] = vel[:, 1:] / (vel[:, [0]] + 1e-10)
                vel = np.append(vel, [[0., 0., 0.]], axis=0)
                traj = np.concatenate((traj, vel[:, 1:]), axis=1)
                self.trajectories.append(traj)
                self.trajectory = []
                print("append trajectory, shape {}".format(traj.shape))
            elif key == ord('s'):
                ic(len(self.trajectories))
                traj = np.vstack(self.trajectories)
                file_name = os.path.join(
                    self.path,
                    "trial_{:04d}.csv".format(int(input("save trajectories to file: trial_")))
                )
                np.savetxt(file_name, traj, delimiter=",")
                self.trajectory = []
                self.trajectories = []
                print("saving trajectories to file {}, shape: {}".format(file_name, traj.shape))

        cv2.destroyAllWindows()
        # cv2.imwrite("last_view.png",  self.img)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--name', "-n", type=str, default='default_recording')
def main(name):
    file_path = os.path.join(get_home(), 'projects/data/mouse_recorder', name)
    ic(create_path(file_path))
    mouse_recorder = MouseRecorder(path=file_path)
    mouse_recorder.start()


if __name__ == "__main__":
    main()
