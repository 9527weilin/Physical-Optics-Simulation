"""img_data.py: 光学系统参数数据类"""
from dataclasses import dataclass
import numpy as np

@dataclass
class OpticalParams:
    wavelength: float = 671e-9  # 波长 (m)
    dx: float = 6.5e-6            # x方向像素物理尺寸 (m)，
    dy: float = 6.4e-6
    Nx: int = 512
    Ny: int = 512

    imageTolens: float = 150e-3  # 物距 (m)

    f1: float = 100e-3           # 透镜焦距 (m)
    f2: float = 200e-3
    f1Tof2: float = 100e-3

    z: float = 20e-3             # 初始传播距离 (m)
    mode: int = 1                # 0: 无透镜; 1: 1个透镜; 2: 2个透镜
    radius_mm1: float = 10e-3
    radius_mm2: float = 10e-3