import numpy as np
import pytest

from perception_numpy.boxes import boxes_from_segmentation
from perception_numpy.nn import cross_entropy_and_gradient
from perception_numpy.segmentation import (
    ego_vehicle_mask,
    logits_to_mask,
    normalize_image,
    queries_to_instances,
    queries_to_semantic,
)


def test_gradient_de_l_entropie_croisee_par_differences_finies():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(6, 3)), np.array([0, 1, 2, 1, 0, 2])
    W, b = rng.normal(size=(3, 3)), rng.normal(size=3)
    weights = np.array([1.0, 2.0, 0.5])
    L, dW, db = cross_entropy_and_gradient(X, y, W, b, weights)
    eps = 1e-6
    for idx in [(0, 0), (1, 2), (2, 1)]:
        Wp = W.copy()
        Wp[idx] += eps
        Lp = cross_entropy_and_gradient(X, y, Wp, b, weights)[0]
        assert dW[idx] == pytest.approx((Lp - L) / eps, abs=1e-4)
    bp = b.copy()
    bp[1] += eps
    assert db[1] == pytest.approx(
        (cross_entropy_and_gradient(X, y, W, bp, weights)[0] - L) / eps, abs=1e-4
    )


def test_entropie_croisee_poids_nuls_equivaut_a_log_C():
    X, y = np.ones((4, 2)), np.array([0, 1, 0, 1])
    L, _, _ = cross_entropy_and_gradient(X, y, np.zeros((2, 3)), np.zeros(3))
    assert L == pytest.approx(np.log(3))


def test_normalize_image():
    img = np.full((2, 2, 3), 255, dtype=np.uint8)
    out = normalize_image(img, normalize=False)
    assert out.shape == (3, 2, 2) and out.dtype == np.float32
    np.testing.assert_allclose(out, 1.0)
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    np.testing.assert_allclose(normalize_image(rgb, bgr=True, normalize=False)[:, 0, 0], [0, 0, 1])


def test_logits_to_mask():
    logits = np.zeros((2, 2, 2))
    logits[1] = 5.0
    probs, mask = logits_to_mask(logits, 4, 4)
    assert probs.shape == (2, 4, 4) and mask.shape == (4, 4)
    np.testing.assert_allclose(probs.sum(axis=0), 1.0)
    assert np.all(mask == 1)


def _deux_requetes():
    class_logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    mask_logits = np.full((2, 4, 4), -10.0)
    mask_logits[0, :, :2] = 10.0
    mask_logits[1, :, 2:] = 10.0
    return class_logits, mask_logits


def test_queries_to_semantic_partage_gauche_droite():
    _, mask = queries_to_semantic(*_deux_requetes(), 4, 4)
    assert np.all(mask[:, :2] == 0) and np.all(mask[:, 2:] == 1)


def test_queries_to_instances_deux_boites():
    boxes, scores, masks = queries_to_instances(*_deux_requetes(), 4, 4, class_ids=[0, 1])
    np.testing.assert_array_equal(boxes, [[0, 0, 2, 4], [2, 0, 4, 4]])
    assert scores.shape == (2,) and masks.shape == (2, 4, 4)
    boxes, _, _ = queries_to_instances(*_deux_requetes(), 4, 4, class_ids=[1])
    np.testing.assert_array_equal(boxes, [[2, 0, 4, 4]])


def test_boxes_from_segmentation_taille_minimale_et_masque_ego():
    mask = np.zeros((60, 60), dtype=int)
    mask[10:30, 10:30] = 1
    mask[40:42, 40:42] = 1
    np.testing.assert_array_equal(
        boxes_from_segmentation(mask, [1], min_area=300), [[10, 10, 30, 30]]
    )
    ego = np.zeros_like(mask, dtype=bool)
    ego[20, 20] = True
    assert boxes_from_segmentation(mask, [1], ego=ego, min_area=1).shape == (1, 4)


def test_ego_vehicle_mask_garde_la_composante_du_bord_bas():
    masks = np.zeros((4, 20, 20), dtype=int)
    masks[:, 14:20, 5:15] = 13
    masks[:, 2:5, 2:5] = 13
    ego = ego_vehicle_mask(masks, class_id=13, margin=0)
    assert ego[16, 10] and not ego[3, 3]
    assert ego.sum() == 6 * 10
