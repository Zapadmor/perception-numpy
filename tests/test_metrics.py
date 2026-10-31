import numpy as np
import pytest

from perception_numpy.metrics import (
    average_precision,
    class_frequencies,
    class_weights,
    confusion_matrix,
    iou_from_confusion,
    match_detections,
    precision_recall,
    precision_recall_curve,
)


def test_precision_recall_et_denominateurs_nuls():
    assert precision_recall(8, 2, 2) == pytest.approx((0.8, 0.8))
    assert precision_recall(0, 0, 0) == (0.0, 0.0)


def test_class_frequencies_ignore_les_pixels_non_annotes():
    f = class_frequencies([[0, 0, 1, 255]], n_classes=2)
    np.testing.assert_allclose(f, [2 / 3, 1 / 3])


def test_class_weights_decroissants_avec_la_frequence():
    w = class_weights([0.01, 0.1, 0.9])
    assert w[0] > w[1] > w[2]
    assert w[1] == pytest.approx(1 / np.log(1.12))


def test_confusion_matrix_exclut_la_classe_ignoree():
    M = confusion_matrix([0, 1, 1, 255], [0, 1, 0, 0], n_classes=2)
    np.testing.assert_array_equal(M, [[1, 0], [1, 1]])


def test_iou_from_confusion():
    iou, miou, acc = iou_from_confusion([[1, 0], [1, 1]])
    np.testing.assert_allclose(iou, [0.5, 0.5])
    assert miou == pytest.approx(0.5)
    assert acc == pytest.approx(2 / 3)


def test_iou_from_confusion_classe_absente_exclue_de_la_moyenne():
    M = np.zeros((3, 3))
    M[0, 0], M[1, 1] = 4, 4
    iou, miou, _ = iou_from_confusion(M)
    assert np.isnan(iou[2])
    assert miou == pytest.approx(1.0)


def test_match_detections_une_reference_ne_sert_qu_une_fois():
    scores, is_vp = match_detections([[0, 0, 10, 10], [0, 0, 10, 10]], [0.8, 0.9], [[0, 0, 10, 10]])
    np.testing.assert_allclose(scores, [0.9, 0.8])
    np.testing.assert_array_equal(is_vp, [True, False])


def test_courbe_precision_rappel_et_precision_moyenne():
    P, R = precision_recall_curve([0.9, 0.8, 0.7], [True, False, True], n_obj=2)
    np.testing.assert_allclose(P, [1, 0.5, 2 / 3])
    np.testing.assert_allclose(R, [0.5, 0.5, 1.0])
    assert average_precision(P, R) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)
