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
