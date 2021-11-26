import logging
import numpy as np
import matplotlib.pyplot as plt
from robot_utils.math.Quaternion import Quaternion
from robot_utils.math.ProcessTraj import TrajNormalizer


class TVMP:
    def __init__(self, kernel_num=100, sigma=0.05, elementary_type='linear'):
        self.kernel_num = kernel_num
        self.centers = np.linspace(1.5, -0.5, kernel_num)
        self.D = sigma * sigma
        self.traj_dim = 7
        self.ElementaryType = elementary_type
        self.lamb = 0.01
        self.muW = np.zeros(shape=(kernel_num, self.traj_dim))

    def __psi__(self, x):
        return np.exp(-0.5 * np.multiply(np.square(x - self.centers), 1/self.D))

    def __dpsi__(self, x):
        fx = np.exp(-0.5 * np.multiply(np.square(x - self.centers), 1/self.D))
        dft = - np.multiply(fx, -(x - self.centers) / self.D)
        return dft

    def __Psi__(self, X):
        X = np.array(X)
        Xmat = np.transpose(np.tile(X, (self.kernel_num,1)))
        Cmat = np.tile(self.centers, (np.shape(X)[0], 1))
        return np.exp(-0.5 * np.multiply(np.square(Xmat - Cmat), 1 / self.D))

    def linearDecayCanonicalSystem(self, t0, t1, numOfSamples):
        return np.linspace(t0, t1, numOfSamples)

    def train(self, trajectories):
        logging.debug("training on traj. with shape: {}".format(np.shape(trajectories)))
        if np.shape(trajectories)[-1] != 8:
            logging.error("TVMP: can only be trained on 7 dimensional trajectories with one dimensional timestamp")
            return

        n_data = np.shape(trajectories)[0]
        self.n_samples = np.shape(trajectories)[1]

        X = self.linearDecayCanonicalSystem(1,0, self.n_samples )
        Psi = self.__Psi__(X)

        # position
        if self.ElementaryType == "linear":
            y0 = np.sum(trajectories[:, 0, 1:4], 0) / n_data
            g = np.sum(trajectories[:, -1, 1:4], 0) / n_data
            self.h_params = np.transpose(np.stack([g, y0-g]))

        if self.ElementaryType == "minjerk":
            y0 = np.sum(trajectories[:, 0:3, 1:4], 0) / n_data
            g = np.sum(trajectories[:, -2: , 1:4], 0) / n_data
            dy0 = (y0[1,2:] - y0[0,2:]) / (y0[1,1] - y0[0,1])
            dy1 = (y0[2,2:] - y0[1,2:]) / (y0[2,1] - y0[1,1])
            ddy0 = (dy1 - dy0) / (y0[1,1] - y0[0,1])
            dg0 = (g[1,2:] - g[0,2:]) / (g[1,1] - g[0,1])
            dg1 = (g[2,2:] - g[1,2:]) / (g[2,1] - g[1,1])
            ddg = (dg1 - dg0) / (g[1,1] - g[0,1])

            b = np.stack([y0[0,:],dy0,ddy0,g[-1,:], dg1, ddg])
            A = np.array([[1,1,1,1,1,1],[0,1,2,3,4,5],[0,0,2,6,12,20],[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,2,0,0,0]])
            self.h_params = np.transpose(np.linalg.solve(A,b))

        self.y0 = y0
        self.goal = g
        Hx = self.H(X)
        H = np.tile(Hx, (n_data, 1, 1))
        Ft = trajectories[:, :, 1:4] - H

        # calculate the quaternion trajectory
        q0 = Quaternion.average(trajectories[:, 0, 4:])
        q1 = Quaternion.average(trajectories[:,-1, 4:])
        Hq = Quaternion.get_slerp_traj(q0, q1, num_sample=self.n_samples)

        Fqs = []
        for i in range(n_data):
            Fqs.append(Quaternion.qtraj_diff(Hq, trajectories[i,:,4:]))

        Fq = np.stack(Fqs, axis=0)
        F = np.concatenate([Ft, Fq], axis=-1)

        pseudo_inv = np.linalg.inv(np.matmul(np.transpose(Psi), Psi) + self.lamb * np.eye(self.kernel_num))
        W = np.matmul(np.matmul(pseudo_inv, np.transpose(Psi)), F)

        self.muW = np.sum(W, 0) / n_data

        return W

    def save_weights_to_file(self, filename):
        np.savetxt(filename, self.muW, delimiter=',')

    def load_weights_from_file(self, filename):
        self.muW = np.loadtxt(filename, delimiter=',')

    def get_weights(self):
        return self.muW

    def get_flatten_weights(self):
        return self.muW.flatten('F')

    def H(self, X):
        if self.ElementaryType is 'linear':
            Xmat = np.stack([np.ones(np.shape(X)[0]), X])
            return np.transpose(np.matmul(self.h_params, Xmat))
        if self.ElementaryType is 'minjerk':
            Xmat = np.stack([np.ones(shape=(1,np.shape(X)[0])), X, np.power(X,2), np.power(X,3), np.power(X,4), np.power(X,5)])
            return np.transpose(np.matmul(self.h_params, Xmat))

    def h(self, x):
        if self.ElementaryType is 'linear':
            return np.matmul(self.h_params, np.matrix([[1],[x]]))
        if self.ElementaryType is 'minjerk':
            return np.matmul(self.h_params, np.matrix([[1],[x],[np.power(x,2)],[np.power(x,3)], [np.power(x,4)], [np.power(x,5)]]))

    def get_min_jerk_params(self, y0, g, dy0, dg, ddy0, ddg):
        b = np.stack([y0, dy0, ddy0, g, dg, ddg])
        A = np.array(
            [[1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5], [0, 0, 2, 6, 12, 20], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
             [0, 0, 2, 0, 0, 0]])

        return np.transpose(np.linalg.solve(A, b))

    def roll(self, y0, g):
        # calculate translation
        X = self.linearDecayCanonicalSystem(1,0, self.n_samples)
        gt = np.reshape(g[:3],(-1,))
        y0t = np.reshape(y0[:3],(-1,))
        print('gt is {}'.format(gt))
        print('y0t is {}'.format(y0t))
        print('X is {}'.format(np.shape(X)))

        if self.ElementaryType == "minjerk":
            dv = np.zeros(shape=np.shape(y0t))
            self.h_params = self.get_min_jerk_params(y0t, gt, dv,dv,dv,dv)
        else:
            self.h_params = np.transpose(np.stack([gt, y0t - gt]))

        Ht = self.H(X)
        Psi = self.__Psi__(X)
        print("Psi size: {}".format(np.shape(Psi)))
        print("Ht size: {}".format(np.shape(Ht)))
        print("muW size: {}".format(np.shape(self.muW)))

        Xi = Ht + np.matmul(Psi,self.muW[:,:3])

        # calculate orientation
        print("y0[0,3:] size: {}".format(np.shape(y0)))
        Hq = Quaternion.get_slerp_traj(y0[3:], g[3:], num_sample=self.n_samples)
        Fq = np.matmul(Psi, self.muW[:,3:])
        Xiq = Quaternion.get_multi_qtraj(Hq, Fq)

        t = 1 - np.expand_dims(X,1)
        traj = np.concatenate([t, Xi, Xiq], axis=1)
        return traj

    def get_shape_modulation(self, t, deri=0):
        x = 1 - t
        if deri == 0:
            res = np.matmul(self.__psi__(x), self.muW)
        elif deri == 1:
            res = np.matmul(self.__dpsi__(x), self.muW)

        return res

    def get_target(self, t):
        F = self.get_shape_modulation(t)
        trasl = np.transpose(self.h(1-t)) + F[0:3]

        hq = Quaternion.slerp(t, self.q0, self.q1)
        rot = Quaternion.qmulti(hq, F[3:])
        return trasl, np.expand_dims(rot, axis=0)

    def get_vel(self,t):
        F = self.get_shape_modulation(t)
        dFt = self.get_shape_modulation(t, deri=1)
        translv = self.h_params[:,-1] + dFt[0:3]

        hq = Quaternion.slerp(t, self.q0, self.q1)
        fq = F[3:]

        dhq = Quaternion.slerp(t, self.q0, self.q1, deri=1)
        dfq = dFt[3:]
        rotv = Quaternion.qmulti(dhq, fq) + Quaternion.qmulti(hq, dfq)
        return np.expand_dims(translv, axis=0), np.expand_dims(rotv, axis=0)

    def set_start_goal(self, y0, g, dy0=None, dg=None, ddy0=None, ddg=None):
        self.y0 = y0[:3]
        self.g = g[:3]
        self.q0 = y0[3:]
        self.q1 = g[3:]

        self.goal = g
        self.start = y0


        if self.ElementaryType == "minjerk":
            zerovec = np.zeros(shape=np.shape(self.y0))
            if dy0 is not None and np.shape(dy0)[0] == np.shape(self.y0)[0]:
                dy0 = dy0
            else:
                dy0 = zerovec

            if ddy0 is not None and np.shape(ddy0)[0] == np.shape(self.y0)[0]:
                ddy0 = ddy0
            else:
                ddy0 = zerovec

            if dg is not None and np.shape(dg)[0] == np.shape(self.y0)[0]:
                dg = dg
            else:
                dg = zerovec

            if ddg is not None and np.shape(ddg)[0] == np.shape(self.y0)[0]:
                ddg = ddg
            else:
                ddg = zerovec

            self.h_params = self.get_min_jerk_params(self.y0 , self.g, dy0=dy0, dg=dg, ddy0=ddy0, ddg=ddg)
        else:
            self.h_params = np.transpose(np.stack([self.g, self.y0 - self.g]))



if __name__ == '__main__':
    tvmp = TVMP()
    trajs = np.loadtxt('pickplace7d.csv', delimiter=',')
    traj_normalizer = TrajNormalizer(trajs)
    trajs = traj_normalizer.normalize_timestamp()

    trajs = np.expand_dims(trajs, axis=0)
    tvmp.train(trajs)
   # ttraj = tvmp.roll(trajs[0,0, 1:] , trajs[0,-1, 1:])

    t = np.linspace(0, 1, 100)
    ttraj = []
    tvmp.set_start_goal(trajs[0,0,1:], trajs[0,-1,1:])

    vtraj = []
    for i in range(100):
        transl, rot = tvmp.get_target(t[i])
        translv, rotv = tvmp.get_vel(t[i])
        print(translv, rotv)

        ttraj.append(np.concatenate([np.array([[t[i]]]), transl, rot], axis=1))
        vtraj.append(np.concatenate([np.array([[t[i]]]), translv, rotv], axis=1))

    ttraj = np.stack(ttraj)
    print('ttraj has shape: {}'.format(np.shape(ttraj)))
    for i in range(7):
        plt.figure(i)
        plt.plot(trajs[0, :, 0], trajs[0, :, i+1], 'k-.')
        plt.plot(ttraj[:, 0], ttraj[:, i+1], 'r-')
        

    plt.show()