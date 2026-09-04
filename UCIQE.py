"""Python-3-compatible copy of the public JOU-UIP UCIQE implementation.

Source: https://github.com/JOU-UIP/UCIQE/blob/main/UCIQE.py
"""

import cv2
import numpy as np


def uciqe(nargin, loc):
    img_bgr = cv2.imread(loc)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    if nargin == 1:
        coe_metric = [0.4680, 0.2745, 0.2576]
    img_lum = img_lab[..., 0] / 255
    img_a = img_lab[..., 1] / 255
    img_b = img_lab[..., 2] / 255

    img_chr = np.sqrt(np.square(img_a) + np.square(img_b))
    img_sat = img_chr / np.sqrt(np.square(img_chr) + np.square(img_lum))
    aver_sat = np.mean(img_sat)
    aver_chr = np.mean(img_chr)
    var_chr = np.sqrt(np.mean(abs(1 - np.square(aver_chr / img_chr))))

    dtype = img_lum.dtype
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536
    hist, bins = np.histogram(img_lum, nbins)
    cdf = np.cumsum(hist) / np.sum(hist)
    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0] - 1) / (nbins - 1), (ihigh[0][0] - 1) / (nbins - 1)]
    con_lum = tol[1] - tol[0]
    return coe_metric[0] * var_chr + coe_metric[1] * con_lum + coe_metric[2] * aver_sat
