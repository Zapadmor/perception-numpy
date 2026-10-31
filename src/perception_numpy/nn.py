"""Briques élémentaires d'un réseau de neurones."""

import numpy as np


def softmax(z, axis=-1):
    """Softmax le long de `axis`, stabilisée par soustraction du maximum."""
    z_max = np.max(z, axis=axis, keepdims=True)
    e_z = np.exp(z - z_max)
    pc = e_z / np.sum(e_z, axis=axis, keepdims=True)

    return pc


def cross_entropy(p, y):
    """Entropie croisée moyenne. p : (n, C) probabilités ; y : (n,) classes vraies."""
    p = np.asarray(p)
    y = np.asarray(y, dtype=int)
    n = p.shape[0]
    return -np.mean(np.log(p[np.arange(n), y]))


def output_size(H_in, k, pad=0, stride=1, dilation=1):
    """Taille de sortie d'une convolution (une dimension)."""
    H_out = (H_in + 2 * pad - dilation * (k - 1) - 1) // stride + 1
    return H_out


def conv2d(image, kernel, stride=1, pad=0, dilation=1):
    """Convolution 2D sans retournement du filtre, sur une image (H, W) et un filtre (k, k)."""
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if pad > 0:
        image = np.pad(image, pad)
    k_h, k_w = kernel.shape
    H_out = output_size(image.shape[0] - 2 * pad, k_h, pad=pad, stride=stride, dilation=dilation)
    W_out = output_size(image.shape[1] - 2 * pad, k_w, pad=pad, stride=stride, dilation=dilation)
    out = np.zeros((H_out, W_out))
    for i in range(k_h):
        for j in range(k_w):
            di, dj = i * dilation, j * dilation
            out += (
                kernel[i, j]
                * image[di : di + stride * H_out : stride, dj : dj + stride * W_out : stride]
            )
    return out


def max_pool2d(image, size=2):
    """Max pooling `size` x `size` de pas `size` ; les lignes et colonnes en trop sont ignorées."""
    image = np.asarray(image)
    H, W = image.shape
    H2, W2 = (H // size) * size, (W // size) * size
    cropped = image[:H2, :W2]
    return cropped.reshape(H2 // size, size, W2 // size, size).max(axis=(1, 3))


def receptive_field(layers):
    """Champ réceptif après une suite de couches [(taille, pas, dilatation), ...]."""
    R = 1
    stride_prod = 1
    for k, s, d in layers:
        R += (k - 1) * d * stride_prod
        stride_prod *= s
    return R


def cross_entropy_and_gradient(X, y, W, b, class_weight=None):
    """Entropie croisée (pondérée) d'une régression softmax et son gradient.

    z = X W + b ; L = somme_i w_{y_i} (-log p_{i, y_i}) / somme_i w_{y_i}.
    X : (n, F) ; y : (n,) ; W : (F, C) ; b : (C,). Retourne (L, dL/dW, dL/db).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n, C = X.shape[0], W.shape[1]

    z = X @ W + b
    p = softmax(z, axis=-1)

    if class_weight is None:
        w = np.ones(n)
    else:
        w = np.asarray(class_weight, dtype=float)[y]
    w_sum = w.sum()

    L = np.sum(w * -np.log(p[np.arange(n), y])) / w_sum

    g = np.zeros((n, C))
    g[np.arange(n), y] = 1.0
    G = (w[:, None] * (p - g)) / w_sum

    dW = X.T @ G
    db = G.sum(axis=0)
    return L, dW, db
