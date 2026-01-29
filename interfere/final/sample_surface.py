# -*- coding: utf-8 -*-
"""
样品表面生成器
    X, Y: 网格坐标
    sample_width: 样品宽度 (m)
    sample_length: 样品长度 (m)
    sample_height: 样品高度范围 (m)
"""

import numpy as np
from scipy.signal import savgol_filter

def sample_surface_selector(X, Y, mode="gaussian_step", sample_width=None, Sample_length=None, Sample_height=None):
    """
    根据模式生成样品表面
    """
    if mode == "gaussian_step":
        return sample_surface_gaussian_step(X, Y, sample_width, Sample_length, Sample_height)
    elif mode == "sphere":
        return sample_surface_sphere(X, Y, sample_width, Sample_length, Sample_height)
    elif mode == "tilt":
        return sample_surface_tilt(X, Y, sample_width, Sample_length, Sample_height)
    elif mode == "random":
        return sample_surface_random(X, Y, sample_width, Sample_length, Sample_height)
    elif mode == "multi_step":
        return sample_surface_multistep(X, Y, sample_width, Sample_length, Sample_height)
    else:
        raise ValueError(f"Unknown sample mode: {mode}")

# =========================================================
# 高斯凸台
# =========================================================
def sample_surface_gaussian_step(X, Y, sample_width, sample_length, sample_height):
    """
    生成带有高斯凸台的样品表面。

    """
    # 高斯凸台占样品宽度和长度的比例
    gaussian_width_ratio = 0.3  # 高斯凸台占样品宽度的比例
    gaussian_length_ratio = 0.3  # 高斯凸台占样品长度的比例
    
    # 根据样品尺寸计算高斯标准差
    sigma_x = gaussian_width_ratio * sample_width
    sigma_y = gaussian_length_ratio * sample_length
    
    # 基础高度随样品尺寸调整
    base_height = sample_height

    # 生成高斯形状
    Z = base_height * np.exp(-((X / sigma_x)**2 + (Y / sigma_y)**2))
    
    return Z

# =========================================================
# 球面
# =========================================================
def sample_surface_sphere(X, Y, sample_width, sample_length, sample_height):
    """
    生成球面样品表面。
    """
    # 球半径设为样品最小尺寸的1/3
    R = min(sample_width, sample_length) / 3 
    
    # 计算每个点到中心的距离
    r = np.sqrt(X**2 + Y**2)
    
    # 生成球面
    Z = np.zeros_like(X)
    
    # 在半径范围内计算球面高度
    mask = r < R
    Z[mask] = np.sqrt(R**2 - r[mask]**2)
    
    # 确保高度从0开始
    Z[mask] = Z[mask] - R  # 这样中心点高度为0，边缘高度为-R
    
    # 使所有高度为正
    Z = Z + R
    
    # 缩放高度到指定的sample_height范围
    current_height_range = np.max(Z) - np.min(Z)
    if current_height_range > 0:
        Z = (Z - np.min(Z)) / current_height_range * sample_height
    
    # 添加一个小的基础高度以避免边缘为0
    Z = Z + sample_height * 0.01
    
    
    return Z
# =========================================================
# 倾斜平面
# =========================================================
def sample_surface_tilt(X, Y, sample_width, sample_length, sample_height):
    """
    生成倾斜平面样品表面。
    """
    # 倾斜角度与样品高度成比例
    max_tilt = sample_height / 2
    
    # 计算每个方向的最大倾斜
    # 在X方向从-max_tilt到+max_tilt变化
    tilt_x = (X / (sample_width/2)) * max_tilt
    
    # 在Y方向从-max_tilt到+max_tilt变化
    tilt_y = (Y / (sample_length/2)) * max_tilt
    
    # 组合倾斜，使总高度变化在sample_height范围内
    Z = (tilt_x + tilt_y) / 2
    
    # 确保最小高度为0
    Z = Z - np.min(Z)
    
    return Z


# =========================================================
# 随机粗糙面
# =========================================================
def sample_surface_random(X, Y, sample_width, sample_length, sample_height, scale_ratio=0.5):
    """
    生成随机粗糙面。
    
    scale_ratio: 粗糙度占样品高度的比例
    """
    # 生成基础噪声
    np.random.seed(42)  # 固定随机种子以获得可重复的结果
    noise = np.random.randn(*X.shape)
    
    # 应用S-G滤波平滑噪声
    noise = savgol_filter(noise, 21, 3, axis=0)
    noise = savgol_filter(noise, 21, 3, axis=1)
    
    # 将噪声归一化到[0,1]范围
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    
    # 使用sample_height缩放噪声
    Z = noise * sample_height * scale_ratio
    
    return Z


# =========================================================
# 多层台阶
# =========================================================
def sample_surface_multistep(X, Y, sample_width, sample_length, sample_height):
    """
    生成多层台阶样品表面。
    """
    step_ratio = 0.15  # 台阶占样品宽度比例
    
    # 根据sample_height定义台阶高度
    # 将总高度sample_height分配给3个台阶
    step1_height = sample_height * 0.3  # 30% 高度
    step2_height = sample_height * 0.6  # 60% 高度
    step3_height = sample_height * 1.0  # 100% 高度
    
    # 定义台阶边界
    step1_boundary = -step_ratio * sample_width
    step2_boundary = 0
    step3_boundary = step_ratio * sample_width
    
    Z = np.zeros_like(X)
    # 第一层台阶
    mask1 = X > step1_boundary
    Z[mask1] += step1_height
    
    # 第二层台阶
    mask2 = X > step2_boundary
    Z[mask2] += (step2_height - step1_height)
    
    # 第三层台阶
    mask3 = X > step3_boundary
    Z[mask3] += (step3_height - step2_height)
    
    return Z
if __name__ == "__main__":
    # 测试样品表面生成
    import matplotlib.pyplot as plt

    sample_width = 3e-3
    sample_length = 3e-3
    sample_height = 0.5e-6
    Nx = 512
    Ny = 512

    x = np.linspace(-sample_width/2, sample_width/2, Nx)
    y = np.linspace(-sample_length/2, sample_length/2, Ny)
    X, Y = np.meshgrid(x, y)

    modes = ["gaussian_step", "sphere", "tilt", "random", "multi_step"]

    fig, axs = plt.subplots(1, len(modes), figsize=(15, 3))
    for ax, mode in zip(axs, modes):
        Z = sample_surface_selector(X, Y, mode, sample_width, sample_length, sample_height)
        im = ax.imshow(Z * 1e6, extent=(-sample_width/2*1e3, sample_width/2*1e3, -sample_length/2*1e3, sample_length/2*1e3))
        ax.set_title(mode)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        fig.colorbar(im, ax=ax, label="Height (µm)")

    plt.tight_layout()
    plt.show()