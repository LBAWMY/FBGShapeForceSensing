# Author(s): Bin Li
# Created on: 2023-08-25

import numpy as np

# rotation matrix Y axis
def RMY(theta):
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    return np.array([[costheta, 0, sintheta], [0, 1, 0], [-sintheta, 0, costheta]])

# rotation matrix Z axis
def RMZ(phi):
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    return np.array([[cosphi, -sinphi, 0], [sinphi, cosphi, 0], [0, 0, 1]])

def Curve2Shape(curve, twist):
    base_rotation = 0
    num = 37
    position = np.zeros((curve.shape[0], 3, num))

    for i in range(curve.shape[0]):
        Rg = RMY(np.pi / 2) @ RMZ(base_rotation * np.pi / 180)
        Pg = np.array([[0], [0], [0]])
        reslution_temp = 3.3

        for j in range(1, num):
            phi = twist[i, j-1] * 1
            # phi = 0
            curvature = curve[i, j-1] * 1
            RZ = RMZ(phi)
            thelta = curvature * reslution_temp
            RY = RMY(thelta)
            r = 1 / curvature
            p = np.array([[r - r * np.cos(thelta)], [0], [r * np.sin(thelta)]])
            Pg_nxt = Rg @ RZ @ p + Pg
            position[i, :, j] = Pg_nxt.reshape(-1,)
            Rg = Rg @ RZ @ RY @ RZ.T
            Pg = Pg_nxt

    return position.reshape(curve.shape[0], 3 * num)