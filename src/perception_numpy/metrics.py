"""Métriques de segmentation et de détection."""

import numpy as np

from perception_numpy.boxes import box_iou


def precision_recall(vp, fp, fn):
    """Précision et rappel ; 0 lorsque le dénominateur est nul."""
    if vp + fp == 0:
        precision = 0.0
    else:
        precision = vp / (vp + fp)

    if vp + fn == 0:
        recall = 0.0
    else:
        recall = vp / (vp + fn)

    return precision, recall


def class_frequencies(masks, n_classes=19, ignore=255):
    """Fréquence de chaque classe parmi les pixels annotés d'un ensemble de masques (N, H, W)."""
    masks = np.asarray(masks)
    valid = masks[masks != ignore]
    counts = np.bincount(valid.ravel(), minlength=n_classes)[:n_classes]
    return counts / counts.sum()


def class_weights(frequencies):
    """Poids de classe w_c = 1 / log(1,02 + fréquence_c)."""
    frequencies = np.asarray(frequencies, dtype=float)
    return 1.0 / np.log(1.02 + frequencies)


def confusion_matrix(gt, pred, n_classes=19, ignore=255):
    """M[c, c'] = nombre de pixels de vraie classe c prédits c' ; la classe `ignore` est exclue."""
    gt = np.asarray(gt).ravel()
    pred = np.asarray(pred).ravel()

    valid = gt != ignore
    gt, pred = gt[valid], pred[valid]

    index = gt.astype(np.int64) * n_classes + pred.astype(np.int64)
    return np.bincount(index, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def iou_from_confusion(M):
    """IoU par classe, mIoU et précision par pixel, à partir d'une matrice de confusion.

    Une classe absente de la vérité et de la prédiction a une IoU `nan`, exclue de la moyenne.
    """
    M = np.asarray(M, dtype=float)
    diag = np.diag(M)
    gt_count = M.sum(axis=1)
    pred_count = M.sum(axis=0)
    denom = gt_count + pred_count - diag

    iou = np.where(denom > 0, diag / np.where(denom > 0, denom, 1.0), np.nan)
    miou = np.nanmean(iou)
    accuracy = diag.sum() / M.sum()
    return iou, miou, accuracy


def match_detections(boxes, scores, gt_boxes, tau=0.5):
    """Apparie les détections d'une image (une classe) aux boîtes de référence.

    Parcours par score décroissant ; une détection est un vrai positif si son IoU avec une boîte
    de référence non encore appariée atteint `tau`. Retourne (scores triés, indicateurs VP).
    """
    boxes = np.atleast_2d(np.asarray(boxes, dtype=float))
    scores = np.asarray(scores, dtype=float)
    gt_boxes = np.atleast_2d(np.asarray(gt_boxes, dtype=float))

    order = np.argsort(-scores, kind="stable")
    boxes_sorted, scores_sorted = boxes[order], scores[order]

    n_gt = gt_boxes.shape[0]
    matched = np.zeros(n_gt, dtype=bool)
    is_vp = np.zeros(len(boxes_sorted), dtype=bool)

    for i, box in enumerate(boxes_sorted):
        if n_gt == 0:
            continue
        ious = box_iou(box, gt_boxes)[0]
        ious[matched] = -1.0
        j = np.argmax(ious)
        if ious[j] >= tau:
            is_vp[i] = True
            matched[j] = True

    return scores_sorted, is_vp


def precision_recall_curve(scores, is_vp, n_obj):
    """Précision et rappel cumulés sur les détections triées par score décroissant."""
    scores = np.asarray(scores, dtype=float)
    is_vp = np.asarray(is_vp, dtype=bool)

    order = np.argsort(-scores, kind="stable")
    cum_vp = np.cumsum(is_vp[order])
    j = np.arange(1, len(scores) + 1)

    return cum_vp / j, cum_vp / n_obj


def average_precision(P, R):
    """Aire sous l'enveloppe de la courbe précision-rappel."""
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)

    P_env = np.maximum.accumulate(P[::-1])[::-1]
    R_prev = np.concatenate([[0.0], R[:-1]])
    return np.sum((R - R_prev) * P_env)
