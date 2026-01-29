import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
import dataclasses

@dataclasses.dataclass
class Light:
    wavelength: float = 550e-9  # 波长（单位：米）
    EPower: float = 1.0         # 电磁波功率
    Focal_length: float = 120e-3  # 波前传播距离，单位：米
    PingMode: str = 'circular'  # 透射屏模式：'circular', 'rectangular', 'grating'

    R: float = 0.1e-3  # 圆孔半径（单位：米）

    Rect_Width: float = 0.05e-3  # 矩形孔宽度（单位：米）
    Rect_Height: float = 0.05e-3  # 矩形孔高度（单位：米）

    Grating_d1: float = 0.01e-3 # 光栅孔宽度（单位：米）
    Grating_d2: float = 0.03e-3 # 光栅孔间距（单位：米）
    Grating_N: int = 5  # 光栅缝数
    
    N: int = 1024  # 分辨率
    obs_range_x: float = 20e-3  # 观察屏x方向范围（单位：米）
    obs_range_y: float = 20e-3  # 观察屏y方向范围（单位：米）

class FraunhoferDiffraction:
    def __init__(self, params: Light):
        self.params = params
        self.k = 2 * np.pi / params.wavelength  # 波数
        
        # 衍射屏坐标
        self.x_ap = None
        self.y_ap = None
        self.aperture = None
        
        # 观察屏坐标
        self.x_obs = None
        self.y_obs = None
        
        # 初始化坐标网格
        self._init_coordinates()
    
    def _init_coordinates(self):
        """初始化衍射屏和观察屏的坐标网格"""
        # 衍射屏坐标
        ap_range = 2 * max(self.params.obs_range_x, self.params.obs_range_y)
        x_ap = np.linspace(-ap_range/2, ap_range/2, self.params.N)
        y_ap = np.linspace(-ap_range/2, ap_range/2, self.params.N)
        self.x_ap, self.y_ap = np.meshgrid(x_ap, y_ap)
        
        # 观察屏坐标
        x_obs = np.linspace(-self.params.obs_range_x/2, 
                           self.params.obs_range_x/2, 
                           self.params.N)
        y_obs = np.linspace(-self.params.obs_range_y/2,
                           self.params.obs_range_y/2,
                           self.params.N)
        self.x_obs, self.y_obs = np.meshgrid(x_obs, y_obs)
    
    def create_circular_aperture(self) -> np.ndarray:
        """创建圆孔透射函数"""
        r = np.sqrt(self.x_ap**2 + self.y_ap**2)
        aperture = np.where(r <= self.params.R, 1.0, 0.0)
        self.aperture = aperture
        return aperture
    
    def direct_integration_circular(self) -> np.ndarray:
        """
        直接积分法计算圆孔夫琅和费衍射
        """
        print("正在进行直接积分计算（圆孔）...")
        
        # 创建圆孔
        aperture = self.create_circular_aperture()
        
        # 空间频率坐标
        u = self.x_obs / (self.params.wavelength * self.params.Focal_length)
        v = self.y_obs / (self.params.wavelength * self.params.Focal_length)
        
        # 直接积分
        N = self.params.N
        U_direct = np.zeros((N, N), dtype=complex)
        
        
        # 计算观察屏的径向坐标
        r_obs = np.sqrt(self.x_obs**2 + self.y_obs**2)
        r_max = np.max(r_obs)
        
        # 计算衍射图案
        theta = r_obs / self.params.Focal_length  # 角度
        q = self.k * self.params.R * theta
        q[q == 0] = 1e-10  # 避免除零
        
        # 第一类贝塞尔函数 J1
        from scipy.special import j1
        amplitude = 2 * j1(q) / q
        intensity_direct = (amplitude**2) * self.params.EPower
        
        print("直接积分计算完成")
        return intensity_direct
    
    def fourier_transform_circular(self) -> np.ndarray:
        """
        傅里叶变换法计算圆孔夫琅和费衍射
        利用FFT计算二维傅里叶变换
        """
        print("正在进行傅里叶变换计算（圆孔）...")
        
        # 创建圆孔
        aperture = self.create_circular_aperture()
        
        # 计算空间频率坐标
        dx = self.x_ap[0, 1] - self.x_ap[0, 0]  # x方向采样间隔
        dy = self.y_ap[1, 0] - self.y_ap[0, 0]  # y方向采样间隔
        
        # 二维傅里叶变换
        U_ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(aperture)))
        
        # 空间频率坐标
        fx = np.fft.fftshift(np.fft.fftfreq(self.params.N, dx))
        fy = np.fft.fftshift(np.fft.fftfreq(self.params.N, dy))
        
        # 转换为观察屏坐标
        u = fx * self.params.wavelength * self.params.Focal_length
        v = fy * self.params.wavelength * self.params.Focal_length
        
        # 插值到实际的观察屏坐标
        from scipy.interpolate import RectBivariateSpline
        interp = RectBivariateSpline(u, v, np.abs(U_ft).T)
        U_interp = interp(self.x_obs[0, :], self.y_obs[:, 0], grid=True)
        
        # 强度分布
        intensity_ft = (U_interp**2) * self.params.EPower
        intensity_ft = intensity_ft / np.max(intensity_ft)  # 归一化
        
        print("傅里叶变换计算完成")
        return intensity_ft
    
    def compare_methods(self, save_plot: bool = True):
        """比较直接积分法和傅里叶变换法的结果"""
        print("\n开始比较两种计算方法...")
        
        # 计算两种方法的结果
        I_direct = self.direct_integration_circular()
        I_ft = self.fourier_transform_circular()
        
        # 提取沿x轴的一维剖面进行比较
        center_idx = self.params.N // 2
        x_line = self.x_obs[center_idx, :] * 1000  # 转换为毫米
        I_direct_line = I_direct[center_idx, :]
        I_ft_line = I_ft[center_idx, :]
        
        # 归一化
        I_direct_line = I_direct_line / np.max(I_direct_line)
        I_ft_line = I_ft_line / np.max(I_ft_line)
        
        # 计算误差
        valid_indices = np.abs(x_line) < 10  
        mse = np.mean((I_direct_line[valid_indices] - I_ft_line[valid_indices])**2)
        max_error = np.max(np.abs(I_direct_line[valid_indices] - I_ft_line[valid_indices]))
        
        print(f"\n定量分析结果（圆孔衍射）:")
        print(f"均方误差(MSE): {mse:.6e}")
        print(f"最大绝对误差: {max_error:.6f}")
        print(f"中心强度比值: 直接法={I_direct_line[center_idx]:.4f}, FFT法={I_ft_line[center_idx]:.4f}")
        
        # 计算艾里斑半径
        D = 2 * self.params.R  # 圆孔直径
        theory_airy_radius = 1.22 * self.params.wavelength / D * self.params.Focal_length
        

        derivative = np.diff(I_direct_line)
        zero_crossings = np.where(np.diff(np.sign(derivative)) > 0)[0]
        if len(zero_crossings) > 0:
            first_min_idx = zero_crossings[0]
            measured_radius = np.abs(x_line[first_min_idx]) / 1000  # 转换为米
            radius_error = np.abs(measured_radius - theory_airy_radius) / theory_airy_radius * 100
            print(f"\n艾里斑半径分析:")
            print(f"理论艾里斑半径: {theory_airy_radius*1000:.4f} mm")
            print(f"测量艾里斑半径: {measured_radius*1000:.4f} mm")
            print(f"相对误差: {radius_error:.2f}%")
        
        # 绘制结果
        self._plot_comparison(I_direct, I_ft, x_line, I_direct_line, I_ft_line, 
                            theory_airy_radius if 'theory_airy_radius' in locals() else None)
        
        return {
            'direct': I_direct,
            'fourier': I_ft,
            'mse': mse,
            'max_error': max_error
        }
    
    def _plot_comparison(self, I_direct, I_ft, x_line, I_direct_line, I_ft_line, 
                        airy_radius: Optional[float] = None):
        """绘制比较结果"""
        fig = plt.figure(figsize=(15, 10))
        
        #直接积分法的衍射图案
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(I_direct, 
                        extent=[self.x_obs[0,0]*1000, self.x_obs[0,-1]*1000,
                               self.y_obs[0,0]*1000, self.y_obs[-1,0]*1000],
                        cmap='hot', aspect='auto')
        ax1.set_xlabel('x (mm)')
        ax1.set_ylabel('y (mm)')
        ax1.set_title('直接积分法 - 衍射图案')
        plt.colorbar(im1, ax=ax1, label='强度')
        
        # 傅里叶变换法的衍射图案
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(I_ft,
                        extent=[self.x_obs[0,0]*1000, self.x_obs[0,-1]*1000,
                               self.y_obs[0,0]*1000, self.y_obs[-1,0]*1000],
                        cmap='hot', aspect='auto')
        ax2.set_xlabel('x (mm)')
        ax2.set_ylabel('y (mm)')
        ax2.set_title('傅里叶变换法 - 衍射图案')
        plt.colorbar(im2, ax=ax2, label='强度')
        
        #两种方法的差异
        ax3 = plt.subplot(2, 3, 3)
        diff = np.abs(I_direct - I_ft)
        im3 = ax3.imshow(diff,
                        extent=[self.x_obs[0,0]*1000, self.x_obs[0,-1]*1000,
                               self.y_obs[0,0]*1000, self.y_obs[-1,0]*1000],
                        cmap='viridis', aspect='auto')
        ax3.set_xlabel('x (mm)')
        ax3.set_ylabel('y (mm)')
        ax3.set_title('两种方法的绝对差异')
        plt.colorbar(im3, ax=ax3, label='绝对误差')
        
        #沿x轴的一维强度分布比较
        ax4 = plt.subplot(2, 3, (4, 6))
        ax4.plot(x_line, I_direct_line, 'b-', linewidth=2, label='直接积分法')
        ax4.plot(x_line, I_ft_line, 'r--', linewidth=2, label='傅里叶变换法')
        
        if airy_radius is not None:
            ax4.axvline(x=airy_radius*1000, color='g', linestyle=':', 
                       label=f'理论艾里斑半径: {airy_radius*1000:.2f} mm')
            ax4.axvline(x=-airy_radius*1000, color='g', linestyle=':')
        
        ax4.set_xlabel('x (mm)')
        ax4.set_ylabel('归一化强度')
        ax4.set_title('沿x轴的强度分布比较')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([-5, 5]) 
        
        #  误差分析图
        ax5 = plt.subplot(2, 3, 5)
        error = np.abs(I_direct_line - I_ft_line)
        ax5.plot(x_line, error, 'k-', linewidth=1.5)
        ax5.fill_between(x_line, 0, error, alpha=0.3)
        ax5.set_xlabel('x (mm)')
        ax5.set_ylabel('绝对误差')
        ax5.set_title('沿x轴的绝对误差')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim([-5, 5])
        
        plt.tight_layout()
        plt.savefig('circular_diffraction_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

# 测试圆孔衍射
if __name__ == "__main__":
    # 创建参数对象
    params = Light()
    params.PingMode = 'circular'
    params.R = 0.1e-3  # 圆孔半径 0.1mm
    
    print("圆孔衍射参数:")
    print(f"波长: {params.wavelength*1e9:.1f} nm")
    print(f"圆孔半径: {params.R*1000:.3f} mm")
    print(f"焦距: {params.Focal_length*1000:.1f} mm")
    print(f"分辨率: {params.N}×{params.N}")
    
    # 创建衍射模拟器
    simulator = FraunhoferDiffraction(params)
    
    # 比较两种方法
    results = simulator.compare_methods(save_plot=True)
    
    print("\n模拟完成！结果已保存为 'circular_diffraction_comparison.png'")