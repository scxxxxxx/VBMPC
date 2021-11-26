import math
import numpy as np
# from eql_utils.math.mju_quaternion import quat2Mat


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


# def get_rpy_error_from_quat(target_quat, pose_quat):
#     """
#     given the orientation (in quaternion) of the current pose and the target, compute the error,
#     as if "error = target - pose".
#     The rotation error is infered from the difference rotation matrix.
#     :param target_quat:
#     :param pose_quat:
#     :return:
#     """
#     target_mat = quat2Mat(target_quat)
#     pose_mat = quat2Mat(pose_quat)
#     diff_mat = target_mat.dot(pose_mat.T)
#     rpy = mat2rpy(diff_mat)
#     return rpy


def get_pose_error(target, pose):
    """
    given the current and the target pose,
    :param target:
    :param pose:
    :return:
    """
    xpos_error = target[:3] - pose[:3]
    rpy = get_rpy_error_from_quat(target[3:], pose[3:])
    return np.concatenate((xpos_error, rpy))


def mat2rpy(m):
    beta = np.arctan2(-m[2, 0], np.sqrt(m[0, 0] * m[0, 0] + m[1, 0] * m[1, 0]))

    if abs(beta - np.pi * 0.5) < 1e-10:
        alpha = 0
        gamma = np.arctan2(m[0, 1], m[1, 1])
    elif abs(beta + np.pi * 0.5) < 1e-10:
        alpha = 0.0
        gamma = - np.arctan2(m[0, 1], m[1, 1])
    else:
        cb = 1.0 / np.cos(beta)
        alpha = np.arctan2(m[1, 0] * cb, m[0, 0] * cb)
        gamma = np.arctan2(m[2, 1] * cb, m[2, 2] * cb)

    return np.array([gamma, beta, alpha])


# def mat2rpy(R):
#     a = ((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2
#     a = max(a,-1)
#     a = min(a,1)
#
#     theta = np.arccos(a)
#     if theta == 0:
#         return np.zeros(3)
#     else:
#         multi = 1 / (2 * math.sin(theta))
#
#         rpy = np.zeros(3)
#
#         rpy[0] = multi * (R[2, 1] - R[1, 2]) * theta
#         rpy[1] = multi * (R[0, 2] - R[2, 0]) * theta
#         rpy[2] = multi * (R[1, 0] - R[0, 1]) * theta
#         return rpy.copy()


def rpy2mat(rpy):
    m = np.zeros((3, 3))
    sgamma = np.sin(rpy[0])
    cgamma = np.cos(rpy[0])
    sbeta = np.sin(rpy[1])
    cbeta = np.cos(rpy[1])
    salpha = np.sin(rpy[2])
    calpha = np.cos(rpy[2])

    m[0, 0] = calpha * cbeta
    m[0, 1] = calpha * sbeta * sgamma - salpha * cgamma
    m[0, 2] = calpha * sbeta * cgamma + salpha * sgamma

    m[1, 0] = salpha * cbeta
    m[1, 1] = salpha * sbeta * sgamma + calpha * cgamma
    m[1, 2] = salpha * sbeta * cgamma - calpha * sgamma

    m[2, 0] = - sbeta
    m[2, 1] = cbeta * sgamma
    m[2, 2] = cbeta * cgamma
    return m


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rot_mat_from_2_vec(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-5 or n2 < 1e-5:
        return np.identity(3)

    v1 = v1 / n1
    v2 = v2 / n2
    axis = np.cross(v1, v2)
    # axis = axis / np.linalg.norm(axis)

    cosA = v1.dot(v2)
    if (cosA + 1) < 1e-5:
        axis = get_orthogonal_vec(v1)
        return R.from_rotvec(np.pi * axis).as_matrix()

    k = 1.0 / (1.0 + cosA)

    mat = np.array([
        (axis[0] * axis[0] * k) + cosA,
        (axis[1] * axis[0] * k) - axis[2],
        (axis[2] * axis[0] * k) + axis[1],
        (axis[0] * axis[1] * k) + axis[2],
        (axis[1] * axis[1] * k) + cosA,
        (axis[2] * axis[1] * k) - axis[0],
        (axis[0] * axis[2] * k) - axis[1],
        (axis[1] * axis[2] * k) + axis[0],
        (axis[2] * axis[2] * k) + cosA
    ]).reshape((3, 3))
    return mat


def get_orthogonal_vec(vec):
    b0 = (vec[0] <  vec[1]) and (vec[0] <  vec[2])
    b1 = (vec[1] <= vec[0]) and (vec[1] <  vec[2])
    b2 = (vec[2] <= vec[0]) and (vec[2] <= vec[1])

    return np.cross(vec, np.array([int(b0), int(b1), int(b2)]))


def vector_projection(x, y):
    p = np.zeros_like(x)
    for i in range(x.shape[0]):
        p[i, :] = y[i, :] * np.dot(x[i, :], y[i, :]) / np.dot(y[i, :], y[i, :])
    return p


def one_hot_encode(batch_labels: np.ndarray, n_labels: int):
    """
    Args:
        batch_labels: numpy vector of size batch_size, or (batch_size, 1)
        n_labels: the total number of classes

    Returns:

    """
    assert np.min(batch_labels) >= 0 and np.max(batch_labels) < n_labels
    batch_labels = batch_labels.flatten()
    y = np.zeros([batch_labels.size, n_labels])
    y[range(batch_labels.size), batch_labels] = 1
    return y


def wrap_value_to_range(value, start, end):
    return start + (value - start) % (end - start)


def wrap_value_to_pi(value):
    return wrap_value_to_range(value, -math.pi, math.pi)