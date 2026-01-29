import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 创建一个简单的信号：线性增长的数据并添加一些噪声
x = np.linspace(0, 10, 100)
y = 2 * x + 5 + np.random.normal(0, 1, size=x.shape)  # 添加噪声

# 使用 Savitzky-Golay 滤波器进行基线去除
smoothed_y = savgol_filter(y, window_length=11, polyorder=3)

# 绘制原始数据和去除基线后的数据
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='原始数据（带噪声）', color='gray', alpha=0.7)
plt.plot(x, smoothed_y, label='去基线后的数据', color='blue', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('基线去除示例：Savitzky-Golay滤波器')
plt.legend()
plt.grid(True)
plt.show()
