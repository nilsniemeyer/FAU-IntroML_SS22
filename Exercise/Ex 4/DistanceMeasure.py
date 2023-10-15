import numpy as np


def calculate_R_Distance(Rx, Ry):
    Dr = 1/len(Rx) * np.linalg.norm(Rx - Ry,1)
    return Dr


def calculate_Theta_Distance(Thetax, Thetay):
    Ixx = np.sum((Thetax - 1/len(Thetax) * np.sum(Thetax))**2)
    Iyy = np.sum((Thetay - 1/len(Thetay) * np.sum(Thetay))**2)
    Ixy = np.sum((Thetax - 1/len(Thetax) * np.sum(Thetax)) * (Thetay - 1/len(Thetax) * np.sum(Thetay)))
    Dtheta = (1 - ((Ixy*Ixy)/(Ixx*Iyy))) * 100
    return Dtheta

