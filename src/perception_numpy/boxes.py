"""Boîtes englobantes au format (u1, v1, u2, v2) : coin haut-gauche, coin bas-droit."""

import numpy as np


def box_iou(a, b):
    """Matrice (N, M) des IoU entre les boîtes de `a` (N, 4) et celles de `b` (M, 4)."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))

    assert a.ndim == 2 and a.shape[1] == 4, "a doit être de forme (N, 4)"
    assert b.ndim == 2 and b.shape[1] == 4, "b doit être de forme (M, 4)"

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # (N,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # (M,)

    u1 = np.maximum(a[:, None, 0], b[None, :, 0])
    v1 = np.maximum(a[:, None, 1], b[None, :, 1])
    u2 = np.minimum(a[:, None, 2], b[None, :, 2])
    v2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(u2 - u1, 0, None) * np.clip(v2 - v1, 0, None)  # (N, M)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union


def generalized_iou(a, b):
    """GIoU entre deux boîtes (4,) : IoU - (|E| - |A u B|) / |E|, E étant la plus petite boîte
    contenant A et B."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    aire_a = (a[2] - a[0]) * (a[3] - a[1])
    aire_b = (b[2] - b[0]) * (b[3] - b[1])

    inter_haut_gauche = np.maximum(a[:2], b[:2])
    inter_bas_droit = np.minimum(a[2:], b[2:])
    largeur_inter = np.maximum(inter_bas_droit[0] - inter_haut_gauche[0], 0.0)
    hauteur_inter = np.maximum(inter_bas_droit[1] - inter_haut_gauche[1], 0.0)
    aire_inter = largeur_inter * hauteur_inter

    aire_union = aire_a + aire_b - aire_inter
    iou = aire_inter / aire_union

    e_haut_gauche = np.minimum(a[:2], b[:2])
    e_bas_droit = np.maximum(a[2:], b[2:])
    aire_e = (e_bas_droit[0] - e_haut_gauche[0]) * (e_bas_droit[1] - e_haut_gauche[1])

    return iou - (aire_e - aire_union) / aire_e


def encode_boxes(boxes, anchors):
    """Écart normalisé (t_u, t_v, t_w, t_h) entre des boîtes et leurs ancres. Retourne (N, 4)."""
    boxes = np.atleast_2d(np.asarray(boxes, dtype=float))
    anchors = np.atleast_2d(np.asarray(anchors, dtype=float))

    u_c, v_c = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]

    u_a, v_a = (anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2
    w_a, h_a = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]

    t_u = (u_c - u_a) / w_a
    t_v = (v_c - v_a) / h_a
    t_w = np.log(w / w_a)
    t_h = np.log(h / h_a)
    return np.stack([t_u, t_v, t_w, t_h], axis=1)


def decode_boxes(t, anchors):
    """Inverse de `encode_boxes` : retourne les boîtes (N, 4) au format coins."""
    t = np.atleast_2d(np.asarray(t, dtype=float))
    anchors = np.atleast_2d(np.asarray(anchors, dtype=float))

    u_a, v_a = (anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2
    w_a, h_a = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]

    u_c = t[:, 0] * w_a + u_a
    v_c = t[:, 1] * h_a + v_a
    w = np.exp(t[:, 2]) * w_a
    h = np.exp(t[:, 3]) * h_a

    return np.stack([u_c - w / 2, v_c - h / 2, u_c + w / 2, v_c + h / 2], axis=1)


def nms(boxes, scores, tau=0.5):
    """Suppression gloutonne des non-maxima. Retourne les indices gardés, par score décroissant."""
    boxes = np.atleast_2d(np.asarray(boxes, dtype=float))
    scores = np.asarray(scores, dtype=float)

    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious = box_iou(boxes[i], boxes[order[1:]])[0]
        order = order[1:][ious <= tau]
    return np.array(keep)
