import numpy as np
import pytest

from perception_numpy.vehicle import lateral_dynamics, planar_dynamics, simulate_lateral_regulation


def test_dynamiques_en_ligne_droite():
    np.testing.assert_allclose(lateral_dynamics([0.0, 0.0], 0.0, 1.0, 3.0), [0.0, 0.0])
    np.testing.assert_allclose(planar_dynamics([0.0, 0.0, 0.0], 0.0, 2.0, 3.0), [2.0, 0.0, 0.0])


def test_vitesse_de_lacet_v_sur_L_tan_delta():
    d = lateral_dynamics([0.0, 0.0], np.pi / 6, 1.0, 3.0)
    assert d[1] == pytest.approx(np.tan(np.pi / 6) / 3.0)


def test_regulation_converge_vers_la_consigne():
    _, y, _, _, error = simulate_lateral_regulation(y_init=-2.0)
    assert abs(error[-1]) < 0.01
    assert abs(y[-1]) < 0.01


def test_commande_toujours_saturee():
    _, _, _, delta, _ = simulate_lateral_regulation(y_init=-100.0, duration=5.0)
    assert np.max(np.abs(delta)) <= np.pi / 4 + 1e-12
    assert np.max(np.abs(delta)) == pytest.approx(np.pi / 4)
