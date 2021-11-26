import numpy as np
from numpy import linalg as LA


class Quaternion:
    @staticmethod
    def dot(q0, q1):
        return np.dot(q0, q1)

    @staticmethod
    def qmulti(q0, q1):
        qw = q0[0] * q1[0] - q0[1] * q1[1] - q0[2] * q1[2] - q0[3] * q1[3]
        qx = q0[0] * q1[1] + q0[1] * q1[0] + q0[2] * q1[3] - q0[3] * q1[2]
        qy = q0[0] * q1[2] - q0[1] * q1[3] + q0[2] * q1[0] + q0[3] * q1[1]
        qz = q0[0] * q1[3] + q0[1] * q1[2] - q0[2] * q1[1] + q0[3] * q1[0]


        return np.array([qw,qx,qy,qz])

    @staticmethod
    def get_multi_qtraj(qtraj0, qtraj1):
        l = np.min([np.shape(qtraj0)[0], np.shape(qtraj1)[0]])
        res = [Quaternion.qmulti(qtraj0[i, :], qtraj1[i, :]) for i in range(l)]
        return np.stack(res)

    @staticmethod
    def normalize(q):
        norm = np.sqrt(np.sum(np.square(q)))
        if norm == 0:
            return q
        else:
            return np.array([q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm])

    @staticmethod
    def isquat(q):
        if np.sum(np.square(q)) == 1:
            return True
        else:
            return False

    @staticmethod
    def qinv(q):
        norm = np.sum(np.square(q))
        return np.array([q[0]/norm, -q[1]/norm, -q[2]/norm, -q[3]/norm  ])

    @staticmethod
    def angular_diff(q0, q1):
        theta = 2 * np.arccos(np.dot(q0, q1))
        return theta

    @staticmethod
    def qtraj_diff(qtraj0, qtraj1):
        l = np.min([np.shape(qtraj0)[0], np.shape(qtraj1)[0]])
        res = [Quaternion.qmulti(Quaternion.qinv(qtraj0[i,:]), qtraj1[i,:]) for i in range(l)]
        return np.stack(res)

    @staticmethod
    def slerp(t, q0, q1, deri=0):
        cosHalfTheta = np.dot(q0,q1)
        if cosHalfTheta < 0:
            q1x = -q1
        else:
            q1x = q1

        if np.fabs(cosHalfTheta) >= 1.0:
            return q0

        halfTheta = np.arccos(cosHalfTheta)
        sinHalfTheta = np.sqrt(1 - cosHalfTheta**2)
        if np.fabs(sinHalfTheta) < 0.001:
            if deri == 0:
                res = [(1 - t) * q0[i] + t * q1x[i] for i in range(4)]
            elif deri == 1:
                res = [-q0[i] + q1x[i] for i in range(4)]

        else:
            if deri == 0:
                ratioA = np.sin((1-t) * halfTheta) / sinHalfTheta
                ratioB = np.sin(t * halfTheta) / sinHalfTheta
            elif deri == 1:
                ratioA = -halfTheta * np.cos((1-t) * halfTheta) / sinHalfTheta
                ratioB = halfTheta * np.cos(t * halfTheta) / sinHalfTheta

            res = [ratioA * q0[i] + ratioB * q1x[i] for i in range(4)]

        return np.array(res)

    @staticmethod
    def get_slerp_traj(q0, q1, num_sample=100, deri=0):
        t = np.linspace(0, 1, num_sample)
        qtraj = np.zeros(shape=(num_sample, 4))
        for i in range(num_sample):
            q = Quaternion.slerp(t[i], q0, q1, deri=deri)
            qtraj[i,:] = q

        return qtraj


    @staticmethod
    def to_matrix(q):
        qw = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]
        res = np.zeros(shape=(3,3))
        res[0,0] = 1 - 2 * qy**2 - 2 * qz**2
        res[0,1] = 2 * qx * qy - 2 * qz * qw
        res[0,2] = 2 * qx * qz + 2 * qy * qw
        res[1,0] = 2 * qx * qy + 2 * qz * qw
        res[1,1] = 1 - 2 * qx**2 - 2 * qz**2
        res[1,2] = 2 * qy * qz - 2 * qx * qw
        res[2,0] = 2 * qx * qz - 2 * qy * qw
        res[2,1] = 2 * qy * qz + 2 * qx * qw
        res[2,2] = 1 - 2 * qx**2 - 2 * qy**2
        return res

    @staticmethod
    def average(qs):
        M = 0
        w = 1 / np.shape(qs)[0]
        for i in range(np.shape(qs)[0]):
            q = np.expand_dims(qs[i,:], axis=1)
            M = M + w * np.matmul(q, np.transpose(q))

        sig, vec = LA.eig(M)
        return vec[:,np.argmax(sig)]


if __name__ == '__main__':
    q0 = np.array([1,0,0,0])
    q1 = np.array([0,0,-1,0])
    qs = np.array([[0.707,0.707,0,0], [0.5,0.5,0.5,0.5]])
