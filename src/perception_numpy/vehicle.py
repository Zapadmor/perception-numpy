"""Modèle cinématique d'une voiture (bicyclette) et régulation latérale simulée."""

import numpy as np


def lateral_dynamics(x, delta, v, L):
    """Dérivée de l'état [y, theta] sur route horizontale (delta déjà saturé)."""
    y_dot = v * np.sin(x[1])
    theta_dot = (v / L) * np.tan(delta)
    return np.array([y_dot, theta_dot])


def planar_dynamics(x, delta, v, L):
    """Dérivée de l'état [x, y, theta] dans le plan."""
    x_dot = v * np.cos(x[2])
    y_dot = v * np.sin(x[2])
    theta_dot = (v / L) * np.tan(delta)
    return np.array([x_dot, y_dot, theta_dot])


def simulate_lateral_regulation(
    y_init=-2.0,
    theta_init=0.0,
    y_d=0.0,
    v=1.0,
    L=3.0,
    poles=(-2.0, -2.0),
    duration=20.0,
    Te=0.02,
    delta_max=np.pi / 4,
):
    """Simulation d'Euler d'un régulateur par retour d'état saturé suivant la droite y = y_d.

    Gains par placement de pôles : K = [s1 s2 L / v, -(s1 + s2) L / v].
    Retourne (temps, y, theta, delta, erreur).
    """
    s1, s2 = poles
    K = np.array([s1 * s2 * L / v, -(s1 + s2) * L / v])

    t = np.arange(0, duration, Te)
    n = len(t)
    x_system = np.array([y_init, theta_init], dtype=float)
    y_hist, theta_hist = np.zeros(n), np.zeros(n)
    delta_hist, error_hist = np.zeros(n), np.zeros(n)

    for k in range(n):
        y, theta = x_system
        error = y - y_d
        delta = np.clip(-K[0] * error - K[1] * theta, -delta_max, delta_max)

        y_hist[k], theta_hist[k] = y, theta
        delta_hist[k], error_hist[k] = delta, error

        x_system = x_system + lateral_dynamics(x_system, delta, v, L) * Te

    return t, y_hist, theta_hist, delta_hist, error_hist
