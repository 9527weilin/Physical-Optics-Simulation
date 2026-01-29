# -*- coding: utf-8 -*-
"""
定义不同类型的光源模型，统一返回 (k, S)
"""

import numpy as np
from dataclasses import dataclass
import math

# =========================================================
#                  光源参数结构体
# =========================================================
@dataclass
class LightSource:
    type: str = "gaussian"          # 'gaussian', 'rect', 'laser'
    central_wavelength_nm: float = 515.0
    bandwidth_nm: float = 80.0
    wavelength_samples: int = 64


# =========================================================
#               光源统一接口函数
# =========================================================
def generate_spectrum(light: LightSource):
    """
    根据光源类型生成 (k, S)
    """
    if light.type == "gaussian":
        return gaussian_spectrum_k(
            light.central_wavelength_nm,
            light.bandwidth_nm,
            light.wavelength_samples
        )

    elif light.type == "rect":
        return rectangular_spectrum_k(
            light.central_wavelength_nm,
            light.bandwidth_nm,
            light.wavelength_samples
        )
    else:
        raise ValueError(f"未知光源类型: {light.type}")


# =========================================================
#              高斯白光（LED ）
# =========================================================
def gaussian_spectrum_k(lambda0_nm, bandwidth_nm, N):
    lambda0_m = lambda0_nm * 1e-9
    delta_lambda = bandwidth_nm * 1e-9

    k0 = 2 * np.pi / lambda0_m
    delta_k = 2 * np.pi * delta_lambda / (lambda0_m ** 2)

    N = max(N, 3)

    k = np.linspace(
        k0 - 3 * delta_k,
        k0 + 3 * delta_k,
        N,
        dtype=np.float64
    )

    sigma_k = delta_k / 2.355
    S = np.exp(-0.5 * ((k - k0) / sigma_k) ** 2)

    S /= np.sum(S)
    return k, S


# =========================================================
#              矩形光谱
# =========================================================
def rectangular_spectrum_k(lambda0_nm, bandwidth_nm, N):
    lambda0_m = lambda0_nm * 1e-9
    delta_lambda = bandwidth_nm * 1e-9

    lambda_min = lambda0_m - delta_lambda / 2
    lambda_max = lambda0_m + delta_lambda / 2

    lambda_arr = np.linspace(lambda_min, lambda_max, N)
    k = 2 * np.pi / lambda_arr

    S = np.ones_like(k)
    S /= np.sum(S)
    return k, S



# =========================================================
#            相干长度（通用公式）
# =========================================================
def coherence_length(lambda0_m, bandwidth_m):
    """
    高斯光源近似相干长度
    """
    if bandwidth_m <= 0:
        return np.inf
    return lambda0_m ** 2 / bandwidth_m
