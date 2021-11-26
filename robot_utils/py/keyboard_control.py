import threading, time
import glfw
import numpy as np
import logging
import mujoco_py
from pynput.keyboard import Key, Listener


class KeyBoardControl:
    def __init__(self):
        self.key = None

    def on_press(self, key):
        self.key = key

    def on_release(self, key):
        self.key = None
        if key == Key.esc:
            return False

    def keyboard_thread(self):
        lock = threading.Lock()
        with lock:
            with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
                listener.join()

    def start_keyboard_listener(self):
        threading.Thread(target=self.keyboard_thread).start()


class KeyboardControl:
    def __init__(self, env, perturb_scale=1, perturb_site_list=None):
        self.env = env
        self._show_instructions()
        self._reset()

        self.perturb_scale = 1
        self.perturb_site_list = perturb_site_list
        if perturb_site_list is None:
            self.perturb_index = -1
        else:
            self.perturb_index = len(perturb_site_list)

        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        env.viewer.add_keypress_callback("any", self.on_press)
        env.viewer.add_keyup_callback("any", self.on_release)
        env.viewer.add_keyrepeat_callback("any", self.on_press)

    def _show_instructions(self):
        instr = "INSTRUCTIONS ON KEYBOARD CONTROL\nKeys\tCommand\n"
        instr = instr + "w-s\tmove CoM forward, backward in x-y plane\n"
        instr = instr + "a-d\tmove CoM to the left, right in x-y plane\n"
        instr = instr + "e-r\tturn left, right around z axis\n"
        instr = instr + "j-l\tperturb in x axis\n"
        instr = instr + "i-k\tperturb in y axis\n"
        instr = instr + "u-o\tperturb in z axis\n"
        instr = instr + "alt_l\tswitch perturbation site\n"
        instr = instr + "ESC\tquit\n"
        logging.info(instr)

    def _reset(self):
        self.perturb_index = -1
        self.force = np.zeros(3)

    def get_perturbation_status(self):
        return self.force, self.torque, self.perturb_index

    def on_press(self, window, key, scancode, action, mods):
        f = 1.0
        force = np.zeros(3)
        # controls for moving position
        if key == glfw.KEY_J:
            force = np.array([-f, 0.0, 0.0], dtype=np.float64)
            logging.info("apply force {} in -x direction".format(force))
        elif key == glfw.KEY_L:
            force = np.array([f, 0.0, 0.0], dtype=np.float64)
            logging.info("apply force {} in x direction".format(force))
        elif key == glfw.KEY_I:
            force = np.array([0.0, f, 0.0], dtype=np.float64)
            logging.info("apply force {} in y direction".format(force))
        elif key == glfw.KEY_K:
            force = np.array([0.0, -f, 0.0], dtype=np.float64)
            logging.info("apply force {} in -y direction".format(force))
        elif key == glfw.KEY_U:
            force = np.array([0.0, 0.0, f], dtype=np.float64)
            logging.info("apply force {} in z direction".format(force))
        elif key == glfw.KEY_O:
            force = np.array([0.0, 0.0, -f], dtype=np.float64)
            logging.info("apply force {} in -z direction".format(force))
        elif key == glfw.KEY_LEFT_ALT:
            if self.perturb_site_list is not None:
                self.perturb_index += 1
                if self.perturb_index >= len(self.perturb_site_list):
                    self.perturb_index = 0
                logging.info("perturbation changed to {}".format(self.perturb_site_list[self.perturb_index]))

        if self.perturb_site_list is not None:
            self.force = force
            torque = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            point = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            body = self.env.sim.model.body_name2id(self.perturb_site_list[0])
            qfrc = np.zeros(self.env.sim.model.nv, dtype=np.float64)
            mujoco_py.functions.mj_applyFT(self.env.sim.model, self.env.sim.data, force, torque, point, body, qfrc)
            self.env.sim.data.qfrc_applied[:] = qfrc
            logging.debug("qfrc: {}".format(np.sum(qfrc)))

    def on_release(self, window, key, scancode, action, mods):
        if key == glfw.KEY_Q:
            self._reset()


