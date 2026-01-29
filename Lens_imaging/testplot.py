import matplotlib.pyplot as plt
import numpy as np
from img_data import OpticalParams

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 光学参数类

# 画图函数
def plot_optical_system(params: OpticalParams,z_init=20):
    # 创建图形
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 设置坐标轴范围和比例
    ax.set_xlim(-200, 350)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal', adjustable='box')
    
    # 物体的位置 (红色点), 物点应在 -imageTolens 位置
    ax.scatter(-params.imageTolens, 0, color='r', s=100, zorder=5)
    ax.text(-params.imageTolens-10 , 10, '物点', ha='left', va='center')

    # 添加透镜及其名称
    lens_positions = [0]  # 第一个透镜的位置在0
    if params.mode >= 1:
        lens_positions.append(params.f1)  # 第一个透镜的位置
        ax.plot([params.f1, params.f1], [-5, 5], 'g-', lw=2)  # 透镜1
        ax.text(params.f1 - 10, -15, '透镜1', ha='left', va='center')
    
    if params.mode == 2:
        lens_positions.append(params.f1 + params.f1Tof2)  # 第二个透镜的位置
        ax.plot([params.f1 + params.f1Tof2, params.f1 + params.f1Tof2], [-5, 5], 'b-', lw=2)  # 透镜2
        ax.text(params.f1 + params.f1Tof2-10, -15, '透镜2', ha='left', va='center')

    # 计算观察平面的距离
    if params.mode == 0:
        observation_distance = z_init   # 无透镜，直接是物距
    else:
        observation_distance = lens_positions[-1] + z_init  # 使用最后一个透镜位置加上 z_init

    # 添加观察平面位置
    ax.scatter(observation_distance, 0, color='g', s=100, zorder=5)
    ax.text(observation_distance, 10, '观察平面', ha='left', va='center')

    # 设置坐标轴标签和标题
    ax.set_xlabel('位置 (mm)')
    ax.set_ylabel('高度 (mm)')
    ax.set_title('光路系统模拟')

    # 显示网格
    ax.grid(True)

    # 显示图形
    plt.show()

# 测试代码
if __name__ == "__main__":
    # 设置光学参数
    params = OpticalParams(
        imageTolens=150,  # 物距 (mm)
        f1=100,           # 透镜1焦距 (mm)
        f2=200,           # 透镜2焦距 (mm)
        f1Tof2=100,       # 透镜1和透镜2的间距 (mm)
       # 观察平面初始位置 (mm)
        mode=0,           # 使用2个透镜
    )
    z_init=30
    # 绘制光学系统
    plot_optical_system(params,z_init=20)
