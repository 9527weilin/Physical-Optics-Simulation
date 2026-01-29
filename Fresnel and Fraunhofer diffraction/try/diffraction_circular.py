import numpy as np
from scipy.special import j1
from diffraction_data import Light

def circular_aperture(x: np.ndarray, y: np.ndarray, params: Light) -> tuple:
    """
    圆孔夫琅和费衍射
    """
    R = params.R  
    lam = params.wavelength  
    z = params.Focal_length  
    k = 2 * np.pi / lam
    I0 = params.EPower
    
    # 观察屏坐标
    X_obs = x
    Y_obs = y
    
    # ========== 解析解计算 ==========
    r = np.sqrt(X_obs**2 + Y_obs**2)
    u = k * R * r / z
    
    airy_pattern = np.where(u == 0, 1.0, 2 * j1(u) / u)
    #=========面积、常数项、二次相位==========
    area = np.pi * R**2
    prefactor = np.exp(1j * k * z) / (1j * lam * z)
    quadratic_phase = np.exp(1j * k * (X_obs**2 + Y_obs**2) / (2 * z))
    
    E_analytic = area * prefactor * quadratic_phase * airy_pattern
    I_analytic = np.abs(E_analytic)**2 * I0
    
    # ========== FFT方法计算 ==========
    X_aperture, Y_aperture, fx, fy, X_fft, Y_fft, dx_aperture, dy_aperture = params.generate_aperture_plane()
    
    # 圆形孔径函数
    r_aperture = np.sqrt(X_aperture**2 + Y_aperture**2)

    # n = 50  # 阶数，越大边缘越陡
    # aperture_func = np.exp(-(r_aperture / R)**(2*n))

    aperture_func = (r_aperture <= R).astype(float)

    # 傅里叶变换
    F = np.fft.fft2(aperture_func)
    F_shifted = np.fft.fftshift(F)
    
    # 观察屏上的电场（FFT解）
    quadratic_phase_fft = np.exp(1j * k * (X_fft**2 + Y_fft**2) / (2 * z))
    scale_factor = dx_aperture * dy_aperture
    
    E_fft = prefactor * F_shifted * quadratic_phase_fft * scale_factor
    I_fft = np.abs(E_fft)**2 * I0
    
    # 孔径函数信息
    aperture_info = {
        'func': aperture_func,
        'X': X_aperture,
        'Y': Y_aperture,
        'x_range': [-R*1.2, R*1.2],
        'y_range': [-R*1.2, R*1.2]
    }
    
    return E_analytic, I_analytic, E_fft, I_fft, aperture_info