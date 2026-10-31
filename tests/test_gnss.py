import numpy as np
import pytest

from perception_numpy.gnss import standalone_positioning

RX = np.array([1000.0, -2000.0, 30.0])
CLOCK = 150.0
SATS = 2.0e7 * np.array(
    [
        [1.0, 0.2, 0.5],
        [-0.8, 0.6, 0.7],
        [0.1, -0.9, 0.6],
        [0.5, 0.5, 0.9],
        [-0.4, -0.7, 0.8],
        [0.9, -0.3, 0.4],
    ]
)


def _pseudoranges():
    return np.linalg.norm(SATS - RX, axis=1) + CLOCK


def test_retrouve_position_et_horloge_sans_bruit():
    X, res, hdop = standalone_positioning(_pseudoranges(), SATS)
    np.testing.assert_allclose(X[:3], RX, atol=1e-4)
    assert X[3] == pytest.approx(CLOCK, abs=1e-4)
    assert res < 1e-6
    assert 0 < hdop < 10


def test_erreur_de_mesure_sur_un_satellite_se_voit_dans_les_residus():
    pr = _pseudoranges()
    pr[2] += 50.0
    _, res, _ = standalone_positioning(pr, SATS)
    assert res > 1.0


def test_moins_de_quatre_satellites_est_refuse():
    with pytest.raises(AssertionError):
        standalone_positioning(_pseudoranges()[:3], SATS[:3])
