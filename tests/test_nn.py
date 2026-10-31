import numpy as np
import pytest

from perception_numpy.nn import (
    conv2d,
    cross_entropy,
    max_pool2d,
    output_size,
    receptive_field,
    softmax,
)


def test_softmax_uniforme_et_somme_a_un():
    np.testing.assert_allclose(softmax(np.zeros(2)), [0.5, 0.5])
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_softmax_invariante_par_translation_et_sans_debordement():
    z = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(softmax(z), softmax(z + 1000.0))
    assert np.all(np.isfinite(softmax(np.array([1000.0, 0.0]))))


def test_cross_entropy_deux_classes_equiprobables():
    assert cross_entropy([[0.5, 0.5]], [0]) == pytest.approx(np.log(2))


@pytest.mark.parametrize(
    ("H", "k", "kwargs", "attendu"),
    [(5, 3, {}, 3), (5, 3, {"pad": 1}, 5), (5, 3, {"stride": 2}, 2), (7, 3, {"dilation": 2}, 3)],
)
def test_output_size(H, k, kwargs, attendu):
    assert output_size(H, k, **kwargs) == attendu


def test_conv2d_difference_diagonale():
    image = np.arange(9).reshape(3, 3)
    out = conv2d(image, np.array([[1, 0], [0, -1]]))
    np.testing.assert_allclose(out, np.full((2, 2), -4.0))


def test_conv2d_remplissage_conserve_la_taille_et_somme_au_centre():
    image = np.arange(9).reshape(3, 3)
    out = conv2d(image, np.ones((3, 3)), pad=1)
    assert out.shape == (3, 3)
    assert out[1, 1] == image.sum()


def test_max_pool2d_et_rognage_des_bords():
    image = np.arange(16).reshape(4, 4)
    np.testing.assert_array_equal(max_pool2d(image), [[5, 7], [13, 15]])
    assert max_pool2d(np.zeros((5, 5))).shape == (2, 2)


def test_receptive_field():
    assert receptive_field([(3, 1, 1), (3, 1, 1)]) == 5
    assert receptive_field([(3, 2, 1), (3, 1, 1)]) == 7
