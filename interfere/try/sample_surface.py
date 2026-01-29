# -*- coding: utf-8 -*-
"""
样品表面生成器（支持多种模式）
- 高度随 sample_width / sample_height 缩放
"""

import numpy as np
from scipy.signal import savgol_filter

def sample_surface_selector(X, Y, mode="gaussian_step", sample_width=None, Sample_length=None):
    """
    根据模式生成样品表面
    X, Y: 网格坐标
    mode: 样品模式
    sample_width / sample_height: 样品总尺寸，用于自动缩放高度和瑕疵大小
    """
    if mode == "gaussian_step":
        return sample_surface_gaussian_step(X, Y, sample_width, Sample_length)
    elif mode == "sphere":
        return sample_surface_sphere(X, Y, sample_width, Sample_length)
    elif mode == "tilt":
        return sample_surface_tilt(X, Y, sample_width, Sample_length)
    elif mode == "random":
        return sample_surface_random(X, Y, sample_width, Sample_length)
    elif mode == "multi_step":
        return sample_surface_multistep(X, Y, sample_width, Sample_length)
    else:
        raise ValueError(f"Unknown sample mode: {mode}")

# =========================================================
# 高斯凸台
# =========================================================
def sample_surface_gaussian_step(X, Y, sample_width, sample_length):
    """
    生成带有高斯凸台的样品表面。
    
    X, Y: 网格坐标
    sample_width: 样品宽度 (m)
    sample_length: 样品长度 (m)
    """
    # 高斯凸台占样品宽度和长度的比例
    gaussian_width_ratio = 0.3  # 高斯凸台占样品宽度的比例
    gaussian_length_ratio = 0.3  # 高斯凸台占样品长度的比例
    
    # 根据样品尺寸计算高斯标准差
    sigma_x = gaussian_width_ratio * sample_width
    sigma_y = gaussian_length_ratio * sample_length
    
    # 基础高度随样品尺寸调整，幅度较小，随尺寸增加
    base_height = 0.2e-6 * (sample_width * sample_length / (3e-3 * 3e-3))  # 高度随样品尺寸（面积）缩放

    # 生成高斯形状
    Z = base_height * np.exp(-((X / sigma_x)**2 + (Y / sigma_y)**2))
    
    return Z

# =========================================================
# 球面
# =========================================================
def sample_surface_sphere(X, Y, sample_width, Sample_length):
    R = min(sample_width, Sample_length) / 2  # 半径随样品尺寸
    r2 = X**2 + Y**2
    mask = r2 < R**2
    Z = np.zeros_like(X)
    Z[mask] = R - np.sqrt(R**2 - r2[mask])
    # 高度随 sample_height 缩放到原来的比例
    Z = 1e-3*Z * (3e-3 / R) * (Sample_length / 3e-3)
    return Z - Z.min()

# =========================================================
# 倾斜平面
# =========================================================
def sample_surface_tilt(X, Y, sample_width, Sample_length):
    # 高度随样品尺寸缩放
    slope_x = 2e-3 * sample_width
    slope_y = 2e-3 * Sample_length
    return slope_x * X / sample_width + slope_y * Y / Sample_length

# =========================================================
# 随机粗糙面
# =========================================================
def sample_surface_random(X, Y, sample_width, Sample_length, scale_ratio=0.2):
    """
    scale_ratio: 粗糙度占样品尺寸比例
    """
    scale = scale_ratio * Sample_length
    noise = np.random.randn(*X.shape)
    noise = savgol_filter(noise, 21, 3, axis=0)
    noise = savgol_filter(noise, 21, 3, axis=1)
    return scale * (noise - noise.min())

# =========================================================
# 多层台阶
# =========================================================
def sample_surface_multistep(X, Y, sample_width, Sample_length):
    step_ratio = 0.1  # 台阶占样品宽度比例
    # 高度随 sample_height 缩放
    step1 = 0.3e-6 * (Sample_length / 3e-3)
    step2 = 0.4e-6 * (Sample_length / 3e-3)
    step3 = 0.6e-6 * (Sample_length / 3e-3)

    Z = np.zeros_like(X)
    Z[X > -step_ratio*sample_width] += step1
    Z[X > 0]                  += step2
    Z[X > step_ratio*sample_width] += step3
    return Z
