import numpy as np
import pytest

from perception_numpy.boxes import box_iou, decode_boxes, encode_boxes, generalized_iou, nms


def test_box_iou_identique_disjoint_et_recouvrement_partiel():
    a = [[0, 0, 2, 2]]
    np.testing.assert_allclose(box_iou(a, [[0, 0, 2, 2]]), [[1.0]])
    np.testing.assert_allclose(box_iou(a, [[5, 5, 6, 6]]), [[0.0]])
    np.testing.assert_allclose(box_iou(a, [[1, 1, 3, 3]]), [[1 / 7]])


def test_generalized_iou():
    assert generalized_iou([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert generalized_iou([0, 0, 1, 1], [2, 0, 3, 1]) == pytest.approx(-1 / 3)


def test_encode_boxes_ancre_identique_donne_zero_et_aller_retour():
    anchors = np.array([[10.0, 10.0, 30.0, 50.0]])
    np.testing.assert_allclose(encode_boxes(anchors, anchors), np.zeros((1, 4)), atol=1e-12)
    boxes = np.array([[12.0, 8.0, 40.0, 44.0]])
    np.testing.assert_allclose(decode_boxes(encode_boxes(boxes, anchors), anchors), boxes)


def test_nms_supprime_le_doublon_et_garde_l_eloigne():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]]
    keep = nms(boxes, [0.9, 0.8, 0.7], tau=0.5)
    np.testing.assert_array_equal(keep, [0, 2])
