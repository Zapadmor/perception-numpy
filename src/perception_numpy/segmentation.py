"""Segmentation sémantique : prétraitement, logits vers masques, masque du véhicule porteur."""

import numpy as np
from scipy import ndimage, special

from perception_numpy.nn import softmax

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_image(rgb, bgr=False, normalize=True):
    """Image uint8 (H, W, 3) -> float32 (3, H, W), mise à l'échelle puis normalisée ImageNet.

    `bgr=True` inverse l'ordre des canaux (erreur classique avec OpenCV).
    """
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    if bgr:
        image = image[:, :, ::-1]
    if normalize:
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(image, (2, 0, 1)).astype(np.float32)


def logits_to_mask(logits, H, W):
    """Logits (C, h, w) -> (probabilités (C, H, W), masque (H, W) uint8).

    Interpolation bilinéaire à la taille de l'image, softmax par pixel, puis argmax.
    """
    logits = np.asarray(logits, dtype=float)
    _, h, w = logits.shape
    logits_up = ndimage.zoom(logits, (1, H / h, W / w), order=1)
    probabilities = softmax(logits_up, axis=0)
    mask = np.argmax(probabilities, axis=0).astype(np.uint8)
    return probabilities, mask


def queries_to_semantic(class_logits, mask_logits, H, W):
    """Combine N_q requêtes (type Mask2Former) en une segmentation sémantique.

    class_logits : (N_q, C + 1), dernière colonne = « pas d'objet » ; mask_logits : (N_q, h, w).
    Retourne (scores (C, H, W), masque (H, W) uint8).
    """
    p = softmax(np.asarray(class_logits, dtype=float), axis=-1)[:, :-1]
    m = special.expit(np.asarray(mask_logits, dtype=float))

    S = np.einsum("qc,qhw->chw", p, m)
    _, h, w = S.shape
    S_up = ndimage.zoom(S, (1, H / h, W / w), order=1)
    mask = np.argmax(S_up, axis=0).astype(np.uint8)
    return S_up, mask


def queries_to_instances(class_logits, mask_logits, H, W, class_ids, min_score=0.5):
    """Une instance par requête retenue (classe dans `class_ids`, score >= `min_score`).

    Chaque pixel appartient à la seule requête maximisant score x masque, si ce masque dépasse
    0,5. Retourne (boîtes (M, 4), scores (M,), masques (M, H, W) bool).
    """
    class_logits = np.asarray(class_logits, dtype=float)
    mask_logits = np.asarray(mask_logits, dtype=float)

    p = softmax(class_logits, axis=-1)[:, :-1]
    s_q = p.max(axis=1)
    l_q = p.argmax(axis=1)

    idx = np.nonzero(np.isin(l_q, class_ids) & (s_q >= min_score))[0]
    if idx.size == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros((0, H, W), dtype=bool)

    s_kept = s_q[idx]
    m = special.expit(mask_logits[idx])
    _, h, w = m.shape
    m_up = ndimage.zoom(m, (1, H / h, W / w), order=1)

    weighted = s_kept[:, None, None] * m_up
    winner = np.argmax(weighted, axis=0)
    winner_value = np.take_along_axis(m_up, winner[None], axis=0)[0]
    valid = winner_value > 0.5

    boxes, scores, masks = [], [], []
    for k in range(idx.size):
        instance_mask = (winner == k) & valid
        if not instance_mask.any():
            continue
        rows, cols = np.nonzero(instance_mask)
        boxes.append([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1])
        scores.append(s_kept[k])
        masks.append(instance_mask)

    return (
        np.array(boxes, dtype=float).reshape(-1, 4),
        np.array(scores),
        np.array(masks, dtype=bool),
    )


def ego_vehicle_mask(masks, class_id=13, min_freq=0.5, margin=5):
    """Masque booléen (H, W) du véhicule porteur (capot) à partir de masques (N, H, W).

    Pixels de classe `class_id` dans au moins `min_freq` des masques, composantes touchant le bord
    bas, trous comblés, puis dilatation de `margin` pixels.
    """
    masks = np.asarray(masks)

    frequency = (masks == class_id).mean(axis=0)
    candidate = frequency >= min_freq

    labels, _ = ndimage.label(candidate)
    bottom_labels = np.unique(labels[-1, :])
    bottom_labels = bottom_labels[bottom_labels != 0]
    ego = np.isin(labels, bottom_labels)

    ego = ndimage.binary_fill_holes(ego)
    if margin > 0:
        ego = ndimage.binary_dilation(ego, iterations=margin)
    return ego
