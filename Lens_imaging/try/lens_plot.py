# lens_plot.py
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from PyQt5 import QtCore, QtGui

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_optical_plot(params, z_mm, mag, label_size):
    """
    创建光路图并返回QPixmap
    
    参数:
        params: OpticalParams对象 (单位：米)
        z_mm: 观察平面距离 (mm)
        mag: 放大率
        label_size: QLabel的尺寸
        
    返回:
        QPixmap对象
    """
    # 根据QLabel尺寸计算图形大小，确保充满
    width_inch = label_size.width() / 100  
    height_inch = label_size.height() / 100 
    
    # 确保最小尺寸
    width_inch = max(width_inch, 8)
    height_inch = max(height_inch, 3)
    
    # 创建Matplotlib图形对象，使用计算出的尺寸
    fig = Figure(figsize=(width_inch, height_inch), tight_layout=True)
    ax = fig.add_subplot(111)
    
    # 计算物距和像距
    object_distance = params.imageTolens * 1e3  # 物距 (mm)
    object_position = -object_distance
    # 计算最后一个透镜的位置和像距
    lens_positions = []
    if params.mode >= 1:
        lens1_position = 0 * 1e3  # 第一个透镜的位置
        lens_positions.append(lens1_position)
        
    if params.mode == 2:
        lens2_position = 0 * 1e3 + params.f1Tof2 * 1e3  # 第二个透镜的位置
        lens_positions.append(lens2_position)
    
    # 计算观察平面的距离
    if params.mode == 0:
        observation_distance = -object_position + z_mm  # 无透镜，直接是物距
    else:
        observation_distance = lens_positions[-1] + z_mm  # 使用最后一个透镜位置加上 z_mm
    
    # 计算动态坐标轴范围

    x_min = -object_distance * 5/4
    
    # x轴正方向：显示像距的3/2倍
    x_max = max(observation_distance * 3/2, object_distance * 3/2)
    
    # 确保x_max至少比x_min大一点
    if x_max <= x_min + 10:
        x_max = x_min + 100
    
    # 计算y轴范围，根据x轴范围按比例调整
    x_range = x_max - x_min
    y_range = x_range * (height_inch / width_inch) * 0.6  # 按比例调整
    
    # 设置坐标轴范围和比例
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_range/2, y_range/2)
    ax.set_aspect('equal', adjustable='box')
    
    # 物体的位置 (红色点), 物点应在 -imageTolens 位置
    object_position = -object_distance
    ax.scatter(object_position, 0, color='r', s=100, zorder=5)
    ax.text(object_position - abs(x_range)*0.02, y_range*0.1, '物点', 
            ha='left', va='center', fontsize=10)
    
    # 添加透镜及其名称
    if params.mode >= 1:
        lens1_position = 0 * 1e3
        ax.plot([lens1_position, lens1_position], [-y_range*0.2, y_range*0.2], 
                'g-', lw=3)  # 透镜1
        ax.text(lens1_position - abs(x_range)*0.02, -y_range*0.3, '透镜1', 
                ha='left', va='center', fontsize=10)
    
    if params.mode == 2:
        lens2_position = 0 * 1e3 + params.f1Tof2 * 1e3
        ax.plot([lens2_position, lens2_position], [-y_range*0.2, y_range*0.2], 
                'b-', lw=3)  # 透镜2
        ax.text(lens2_position - abs(x_range)*0.02, -y_range*0.3, '透镜2', 
                ha='left', va='center', fontsize=10)
    
    # 添加观察平面位置
    ax.scatter(observation_distance, 0, color='g', s=100, zorder=5)
    ax.text(observation_distance + abs(x_range)*0.02, y_range*0.1, '观察平面', 
            ha='left', va='center', fontsize=10)
    
    
    # 添加物距标注
    if object_position < 0:
        ax.annotate('', xy=(object_position, y_range*0.25), 
                    xytext=(0, y_range*0.25),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(object_position/2, y_range*0.28, f'物距: {object_distance:.1f} mm', 
                ha='center', va='bottom', color='red', fontsize=9, fontweight='bold')
    
    # 添加像距标注
    if params.mode == 0:
        start_x = object_position
    elif len(lens_positions) > 0:
        start_x = lens_positions[-1]
    else:
        start_x = 0
    
    ax.annotate('', xy=(start_x, y_range*0.25), 
                xytext=(observation_distance, y_range*0.25),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text((start_x + observation_distance)/2, y_range*0.28, 
            f'像距: {observation_distance-start_x:.1f} mm', 
            ha='center', va='bottom', color='blue', fontsize=9, fontweight='bold')
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('位置 (mm)', fontsize=11)
    ax.set_ylabel('高度 (mm)', fontsize=11)
    
    # 根据模式设置标题
    if params.mode == 0:
        title = f'无透镜系统 (物距: {object_distance:.1f} mm, 像距: {observation_distance-start_x:.1f} mm, M = {mag:.3f})'
    elif params.mode == 1:
        title = f'单透镜系统 (物距: {object_distance:.1f} mm, 像距: {observation_distance-start_x:.1f} mm, M = {mag:.3f})'
    else:
        title = f'双透镜系统 (物距: {object_distance:.1f} mm, 像距: {observation_distance-start_x:.1f} mm, M = {mag:.3f})'
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # 显示网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 将图形转换为QPixmap
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    # 获取图形数据
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    
    # 将图形转换为QImage
    buf = canvas.buffer_rgba()
    qimage = QtGui.QImage(buf, width, height, QtGui.QImage.Format_ARGB32)
    pixmap = QtGui.QPixmap.fromImage(qimage)
    
    # 缩放以适应QLabel
    scaled_pixmap = pixmap.scaled(
        label_size,
        QtCore.Qt.IgnoreAspectRatio,  
        QtCore.Qt.SmoothTransformation
    )
    
    plt.close(fig)
    
    return scaled_pixmap
        