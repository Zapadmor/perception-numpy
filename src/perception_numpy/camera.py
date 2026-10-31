"""Modèle de caméra sténopé."""

import numpy as np


def project_points(points, K):
    """Projette des points 3D (repère caméra, mètres) en pixels (u, v).

    points : (N, 3) ; K : matrice intrinsèque (3, 3). Retourne (N, 2).
    """
    points = np.atleast_2d(np.asarray(points))
    if points.shape[1] != 3:
        points = points.reshape(-1, 3)
    temp = np.dot(points, K.T)
    Z = temp[:, 2]
    uv = temp[:, :2] / Z[:, None]
    return uv


def field_of_view(W, fx):
    """Champ de vision horizontal, en degrés, d'une image de largeur W (px) et de focale fx (px)."""
    FOV_deg = 2 * np.arctan(W / (2 * fx))
    return np.degrees(FOV_deg)


def back_project(u, v, K, Z):
    """Point 3D situé à la profondeur Z sur le rayon du pixel (u, v). Retourne un tableau (3,)."""
    P_Z = Z * np.linalg.solve(K, np.transpose(np.array([u, v, 1])))
    return P_Z
