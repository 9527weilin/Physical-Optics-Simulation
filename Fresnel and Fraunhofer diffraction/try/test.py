import numpy as np
import dataclasses
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.interpolate import RectBivariateSpline
from diffraction_circular import circular_aperture
from diffraction_rectangular import rectangular_aperture
from diffraction_grating import grating_aperture
from diffraction_data import Light
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用于显示负号

# 根据孔径类型选择计算函数
def calculate(x: np.ndarray, y: np.ndarray, params: Light) -> tuple:

    if params.PingMode == 'rectangular':
        return rectangular_aperture(x, y, params)
    elif params.PingMode == 'circular':
        return circular_aperture(x, y, params)
    elif params.PingMode == 'grating':
        return grating_aperture(x, y, params)
    else:
        raise ValueError(f"未知模式: {params.PingMode}")
    
def calculate_best_screen_size(params: Light) -> dict:
    best_screen_sizes = {}

    # 对于圆孔径（Circular Aperture）
    best_screen_sizes['circular'] = 2 * (2 * (1.22 * params.wavelength * params.Focal_length) / params.R)

    # 对于矩形孔径（Rectangular Aperture）
    best_screen_sizes['rectangular'] = 6 * (1.22 * params.wavelength * params.Focal_length) / max(params.Rect_Width, params.Rect_Height)

    # 对于光栅（Grating Aperture）
    # 使用光栅的缝隙间距（假设光栅的间距为d，缝数为N）
    best_screen_sizes['grating'] = 2 * 2 * (1.22 * params.wavelength * params.Focal_length) / params.Grating_d2

    return best_screen_sizes

def print_best_screen_size(params: Light):
    best_screen_sizes = calculate_best_screen_size(params)
    
    # 输出各孔径类型的最佳观察屏尺寸
    print("最佳观察屏尺寸（单位：米）：")
    for aperture_type, size in best_screen_sizes.items():
        print(f"{aperture_type.capitalize()} 孔径: {size * 1e3:.2f} mm")

# 计算误差的函数
def compute_errors(I_true: np.ndarray, I_pred: np.ndarray, epsilon: float = 1e-10) -> dict:
   
    # 避免除零，相关误差
    relative_error = (I_pred - I_true) / (I_true + epsilon)
    
    # 计算均方误差（MSE）
    mse = np.mean((I_pred - I_true) ** 2)
    
    # 计算最大误差
    max_error = np.max(np.abs(I_pred - I_true))
    
    # 计算均方根误差（RMS）
    rms_error = np.sqrt(np.mean((I_pred - I_true) ** 2))

    # 返回结果
    errors = {
        'relative_error': relative_error,
        'mse': mse,
        'max_error': max_error,
        'rms_error': rms_error
    }
    return errors


# 绘制一维光强分布
def plot_1d_intensity(x: np.ndarray, I: np.ndarray, title: str, position: int):
    plt.subplot(2, 2, position)
    plt.plot(x, I)
    plt.xlabel('x (mm)')
    plt.ylabel('光强')
    plt.title(title)
    plt.grid(True)

# 绘制2D光强图
def plot_2d_intensity(params: Light, aperture_type: str):

    obs_range_x = params.obs_range_x
    obs_range_y = params.obs_range_y
    x = np.linspace(-obs_range_x/2, obs_range_x/2, params.N)
    y = np.linspace(-obs_range_y/2, obs_range_y/2, params.N)
    X, Y = np.meshgrid(x, y)

    E, I, EFFT, IFFT, aperture_info= calculate(X, Y, params)
    I_norm = I 
    IFFT_norm = np.abs(IFFT) 
    
    # 计算误差
    I_true = I_norm  # 假设FFT结果为真实结果
    errors = compute_errors(I_true, IFFT_norm)

    # 打印误差
    print(f"相对误差：{np.mean(errors['relative_error'])}")
    print(f"均方误差（MSE）：{errors['mse']}")
    print(f"最大误差：{errors['max_error']}")
    print(f"均方根误差（RMS）：{errors['rms_error']}")

    # 创建一个大的图形框架
    plt.figure(figsize=(10, 10))

    # 绘制2D光强图
    plt.subplot(2, 2, 1)
    I_norm_log = np.log(I_norm + 1e-10) 
    plt.imshow(I_norm_log, extent=[x.min() * 1e3, x.max() * 1e3, y.min() * 1e3, y.max() * 1e3], cmap='viridis', origin='lower')
    plt.colorbar(label='归一化光强')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'{aperture_type} 孔径夫琅禾费衍射 - 2D光强分布')

    # 绘制FFT光强图
    plt.subplot(2, 2, 2)
    IFFT_norm_log = np.log(IFFT_norm + 1e-10)
    plt.imshow(IFFT_norm_log, extent=[x.min() * 1e3, x.max() * 1e3, y.min() * 1e3, y.max() * 1e3], cmap='viridis', origin='lower')
    plt.colorbar(label='归一化光强')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'{aperture_type} 孔径夫琅禾费衍射 - FFT光强分布')

    # 绘制1D光强图，取 y=0 处的剖面
    plot_1d_intensity(x * 1e3, I_norm[int(params.N / 2), :], f'{aperture_type} 孔径 - 1D光强分布', 3)
    # 绘制FFT的1D光强图，取 y=0 处的剖面
    plot_1d_intensity(x * 1e3, IFFT_norm[int(params.N / 2), :], f'{aperture_type} 孔径 - FFT 1D光强分布', 4)


    # 显示所有的子图
    plt.tight_layout()
    plt.show()


# 主函数
def main():
    # params = Light(wavelength=550e-9, EPower=1.0, Focal_length=120e-3, PingMode='rectangular', Rect_Width=0.05e-3, Rect_Height=0.05e-3, N=2048, obs_range_x=20e-3, obs_range_y=20e-3)
    # plot_2d_intensity(params, 'rectangular')
    params = Light(wavelength=550e-9, EPower=1.0, Focal_length=120e-3, PingMode='circular', R=0.025e-3, N=1024, obs_range_x=20e-3, obs_range_y=20e-3)
    plot_2d_intensity(params, 'circular')
    # params = Light(wavelength=550e-9, EPower=1.0, Focal_length=120e-3, PingMode='grating', Grating_d1=0.01e-3, Grating_d2=0.03e-3, Grating_N=5, N=2048, obs_range_x=20e-3, obs_range_y=20e-3)
    # plot_2d_intensity(params, 'grating')
    # print_best_screen_size(params)
if __name__ == "__main__":
    main()
