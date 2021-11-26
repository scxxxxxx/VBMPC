import numpy as np
import matplotlib.pyplot as plt
from icecream import ic


class VMP:
    def __init__(self, dim, kernel_num=30, sigma=0.01, elementary_type='linear', use_outrange_kernel=True):
        self.kernel_num = kernel_num
        if use_outrange_kernel:
            self.centers = np.linspace(1.2, -0.2, kernel_num)
        else:
            self.centers = np.linspace(1, 0, kernel_num)

        self.D = sigma
        self.ElementaryType = elementary_type
        self.lamb = 0.01
        self.dim = dim
        self.n_samples = 100
        self.muW = np.zeros(shape=(kernel_num, self.dim))

    def __psi__(self, x):
        return np.exp(-0.5 * np.multiply(np.square(x - self.centers), 1/self.D))

    def __Psi__(self, X):
        X = np.array(X)
        Xmat = np.transpose(np.tile(X, (self.kernel_num,1)))
        Cmat = np.tile(self.centers, (np.shape(X)[0], 1))
        return np.exp(-0.5 * np.multiply(np.square(Xmat - Cmat), 1 / self.D))

    def train(self, trajectories):
        if len(np.shape(trajectories)) == 2:
            trajectories = np.expand_dims(trajectories, 0)

        n_data = np.shape(trajectories)[0]
        self.n_samples = np.shape(trajectories)[1]
        n_dim = np.shape(trajectories)[2] - 1

        self.dim = n_dim
        X = self.linearDecayCanonicalSystem(1,0, self.n_samples )
        Psi = self.__Psi__(X)

        if self.ElementaryType is 'linear':
            y0 = np.sum(trajectories[:, 0, 1:], 0) / n_data
            g = np.sum(trajectories[:, -1, 1:], 0) / n_data
            self.h_params = np.transpose(np.stack([g, y0-g]))

        if self.ElementaryType is 'minjerk':
            y0 = np.sum(trajectories[:, 0:3, 1:], 0) / n_data
            g = np.sum(trajectories[:, -2: , 1:], 0) / n_data
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
        H = np.tile(Hx, (n_data,1,1))
        F = trajectories[:,:,1:] - H

        pseudo_inv = np.linalg.inv(np.matmul(np.transpose(Psi), Psi) + self.lamb * np.eye(self.kernel_num))
        W = np.matmul(np.matmul(pseudo_inv, np.transpose(Psi)), F)

        self.muW = np.sum(W, 0) / n_data

        return

    def save_weights_to_file(self, filename):
        np.savetxt(filename, self.muW, delimiter=',')

    def load_weights_from_file(self, filename):
        self.muW = np.loadtxt(filename, delimiter=',')

    def get_weights(self):
        return self.muW

    def get_flatten_weights(self):
        return self.muW.flatten('F')

    def set_weights(self, ws):
        if np.shape(ws)[-1] == self.dim * self.kernel_num:
            self.muW = np.reshape(ws, (self.kernel_num, self.dim), 'F')
        elif np.shape(ws)[0] == self.kernel_num and np.shape(ws)[-1] == self.dim:
            self.muW = ws
        else:
            raise Exception("The weights have wrong shape. It should have {} rows (for kernel number) and {} columns (for dimensions)".format(self.kernel_num, self.dim))

    def get_position(self, t):
        x = 1 - t
        return np.matmul(self.__psi__(x), self.muW)

    def set_start(self, y0):
        self.y0 = np.reshape(y0,(-1,))
        self.h_params = np.transpose(np.stack([self.goal, self.y0 - self.goal]))

    def set_goal(self, g):
        self.goal = np.reshape(g,(-1,))
        self.h_params = np.transpose(np.stack([self.goal, self.y0 - self.goal]))

    def set_start_goal(self, y0, g):
        self.y0 = np.reshape(y0,(-1,))
        self.goal = np.reshape(g,(-1,))
        self.h_params = np.transpose(np.stack([self.goal, self.y0 - self.goal]))

    def set_start_goal(self, y0, g, dy0=None, dg=None, ddy0=None, ddg=None):
        self.y0 = y0
        self.g = g
        self.q0 = y0
        self.q1 = g

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


    def roll(self, y0, g, n_samples=None):
        n_samples = self.n_samples if n_samples is None else n_samples
        X = self.linearDecayCanonicalSystem(1,0, n_samples)
        g = np.reshape(g,(-1,))
        y0 = np.reshape(y0,(-1,))

        if self.ElementaryType == "minjerk":
            dv = np.zeros(shape=np.shape(y0))
            self.h_params = self.get_min_jerk_params(y0, g, dv,dv,dv,dv)
        else:
            self.h_params = np.transpose(np.stack([g, y0 - g]))

        H = self.H(X)

        Psi = self.__Psi__(X)
        Xi = H + np.matmul(Psi,self.muW)

        t = 1 - np.expand_dims(X,1)
        traj = np.concatenate([t, Xi], axis=1)
        return traj

    def get_target(self, t):
        action = np.transpose(self.h(1-t)) + self.get_position(t)
        return action

    def get_action_with_explored_param(self, t, ws):
        self.set_weights(ws)
        return self.get_action(t)

    def test(self):
        traj = self.roll(self.y0, self.goal)
        for i in range(np.shape(traj)[-1]):
            plt.plot(traj[:,i])

        plt.show()

    def cansys(self, t):
        return 1 - t

    def linearDecayCanonicalSystem(self, t0, t1, numOfSamples):
        return np.linspace(t0, t1, numOfSamples)

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

    def H(self, X):
        if self.ElementaryType is 'linear':
            Xmat = np.stack([np.ones(np.shape(X)[0]), X])
            return np.transpose(np.matmul(self.h_params, Xmat))
        if self.ElementaryType is 'minjerk':
            Xmat = np.stack([np.ones(shape=(1,np.shape(X)[0])), X, np.power(X,2), np.power(X,3), np.power(X,4), np.power(X,5)])
            return np.transpose(np.matmul(self.h_params, Xmat))


if __name__ == '__main__':
    vmp = VMP(1)
    t = np.linspace(0, 1, 1000)
    traj0 = np.stack([t, np.sin(t * 2 * np.pi) + 0.2 * (np.random.random(t.shape) - 0.5)])
    traj0 = np.transpose(traj0)
    # traj1 = np.stack([t, np.cos(t * 2 * np.pi)])
    traj1 = np.stack([t, np.sin(t * 2 * np.pi) + 0.2 * (np.random.random(t.shape) - 0.5)])
    traj1 = np.transpose(traj1)

    trajs = np.stack([traj0, traj1])
    ic(trajs.shape)
    vmp.train(trajs)
    Xi = vmp.roll(y0=0, g=0) #y0=(traj0[0,1:]+traj1[0,1:])/2, g=(traj0[-1,1:]+traj1[-1,1:])/2)

    plt.plot(t, traj0[:,1:], 'r')
    plt.plot(t, traj1[:,1:], 'g')
    plt.plot(t, Xi[:,1:], 'b')
    plt.show()

    xi = vmp.get_position(0.5,1)
    print(xi)