import dataclasses
import numpy as np
@dataclasses.dataclass
class Light:
    wavelength: float = 550e-9  # 波长（单位：米）
    EPower: float = 1.0         # 电磁波功率
    Focal_length: float = 120e-3  # 波前传播距离，单位：米
    PingMode: str = 'circular'  # 透射屏模式：'circular', 'rectangular', 'grating'

    R: float = 0.025e-3  # 圆孔半径（单位：米）

    Rect_Width: float = 0.05e-3  # 矩形孔宽度（单位：米）
    Rect_Height: float = 0.05e-3  # 矩形孔高度（单位：米）

    Grating_d1: float = 0.01e-3 # 光栅宽度（单位：米）
    Grating_d2: float = 0.03e-3 # 光栅周期（单位：米）
    Grating_N: int = 5  # 光栅缝数
    
    N: int = 1024  # 分辨率
    obs_range_x: float = 20e-3  # 观察屏x方向范围（单位：米）
    obs_range_y: float = 20e-3  # 观察屏y方向范围（单位：米）

    def generate_aperture_plane(self):
        """
        生成孔径平面坐标和频率坐标
        返回: (X_a, Y_a, fx, fy, X_fft, Y_fft, dx_aperture, dy_aperture)
        """
        lam = self.wavelength
        z = self.Focal_length
        N = self.N
        Lx_obs = self.obs_range_x
        Ly_obs = self.obs_range_y
        
        # 孔径平面尺寸
        L_aperture_x = lam * z * N / Lx_obs
        L_aperture_y = lam * z * N / Ly_obs
        
        # 孔径平面采样间隔
        dx_aperture = L_aperture_x / N
        dy_aperture = L_aperture_y / N
        
        # 孔径平面坐标
        x_aperture = np.linspace(-L_aperture_x/2, L_aperture_x/2, N, endpoint=False)
        y_aperture = np.linspace(-L_aperture_y/2, L_aperture_y/2, N, endpoint=False)
        X_aperture, Y_aperture = np.meshgrid(x_aperture, y_aperture)
        
        # 频率坐标
        fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx_aperture))
        fy = np.fft.fftshift(np.fft.fftfreq(N, d=dy_aperture))
        
        # 观察屏坐标
        x_fft = fx * lam * z
        y_fft = fy * lam * z
        X_fft , Y_fft = np.meshgrid(x_fft, y_fft)
        
        return X_aperture, Y_aperture, fx, fy, X_fft, Y_fft, dx_aperture, dy_aperture