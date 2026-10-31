import numpy as np
import pytest

from perception_numpy.camera import back_project, field_of_view, project_points

K = np.array([[800.0, 0.0, 400.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]])


def test_project_points_axe_optique_et_decalage():
    uv = project_points([[0, 0, 2], [1, 0, 2]], K)
    np.testing.assert_allclose(uv, [[400, 300], [800, 300]])


def test_back_project_est_l_inverse_de_project_points():
    P = np.array([1.5, -0.5, 7.0])
    u, v = project_points(P, K)[0]
    np.testing.assert_allclose(back_project(u, v, K, P[2]), P)


def test_field_of_view_focale_egale_a_la_demi_largeur_donne_90_degres():
    assert field_of_view(800, 400) == pytest.approx(90.0)
