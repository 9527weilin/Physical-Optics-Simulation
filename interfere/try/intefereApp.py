# -*- coding: utf-8 -*-
"""
白光干涉仪仿真主程序（最终版）
- 滑块控制高分辨率剖面
- X/Y 剖面显示真实 + 测量曲线
- InterfrerPlot, ObjectPlot, errorplot 显示测量形貌和误差
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator  
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMenu, QAction
from Ui_WhiteInterfere import Ui_WhiteInterfere  
from interfere_cul import InterfereData, mirau_white_light_sim 
from interfereLight import LightSource, generate_spectrum, coherence_length 

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置matplotlib中文显示
plt.rcParams['axes.unicode_minus'] = False  # 禁止显示负号时乱码


class WhiteInterfereApp(QtWidgets.QMainWindow):
    """
    白光干涉仪仿真应用的主程序类，继承自QMainWindow。
    负责初始化界面、绑定按钮事件、运行仿真并显示结果。
    """
    def __init__(self):
        super().__init__()
        self.ui = Ui_WhiteInterfere()
        self.ui.setupUi(self)  # 设置UI界面

        self.result = None
        self.Icube = None
        self.Z_true = None
        self.Z_meas = None
        self.error = None

        # 安装事件过滤器，控件缩放时自动刷新
        for label in [self.ui.InterfrerPlot, self.ui.ObjectPlot, self.ui.errorplot, self.ui.xPlot, self.ui.yPlot]:
            label.installEventFilter(self)  # 监听控件事件

        # 连接按钮和滑块事件
        self.ui.SureButton.clicked.connect(self.run_simulation)  # 点击按钮时运行仿真
        self.ui.xSlider.valueChanged.connect(self.plot_profiles)  # X滑块变动时更新剖面
        self.ui.ySlider.valueChanged.connect(self.plot_profiles)  # Y滑块变动时更新剖面

        self.init_defaults()  # 初始化默认参数
        self.init_samplechoice()  # 初始化样品选择菜单


    def init_defaults(self):
        """
        初始化UI中的默认参数值，设置初始显示值。
        """
        cfg = InterfereData()  # 创建默认的仿真数据对象
        self.ui.Wavelength_start.setText(str(cfg.Wavelength_start))  # 设置起始波长
        self.ui.Wavelength_end.setText(str(cfg.Wavelength_end))  # 设置终止波长
        self.ui.Wavelength_samples.setText(str(cfg.Wavelength_samples))  # 设置波长采样数
        self.ui.Coherence_length.setText(str(cfg.Lc))  # 设置相干长度
        self.ui.Phase_offset.setText(str(cfg.Phase_offset))  # 设置相位偏移
        self.ui.ReflectMirror_distance.setText(str(cfg.Fixed_mirror_distance))  # 设置固定反射镜距离
        self.ui.SampleMirror_distance.setText(str(cfg.Sample_postion_distance))  # 设置样品反射镜距离
        self.ui.Scan_step_size.setText(str(cfg.Scan_steps))  # 设置扫描步数
        self.ui.CCD_Distance.setText(str(cfg.CCD_Distance))  # 设置CCD到样品的距离
        self.ui.Pixel_size.setText(str(cfg.Pixel_size))  # 设置像素尺寸
        self.ui.Screen_Hight.setText(str(cfg.CCD_Pixels_Y))  # 设置屏幕高度
        self.ui.Screen_Weight.setText(str(cfg.CCD_Pixels_X))  # 设置屏幕宽度
        self.ui.sample_length.setText(str(cfg.Sample_length*1e3))  # 设置样品宽度（单位: mm）
        self.ui.sample_width.setText(str(cfg.Sample_width*1e3))  # 设置样品高度（单位: mm）
        self.ui.multipleLc.setText(str(cfg.multipleLc))  # 设置扫描范围为多少个相干长度


    def init_samplechoice(self):
        """
        初始化样品选择菜单，允许用户选择不同的样品表面模式。
        """
        menu = QMenu()
        modes = ["gaussian_step", "sphere", "tilt", "random", "multi_step"]  # 样品模式列表
        for mode in modes:
            action = QAction(mode, self)  # 创建菜单项
            action.triggered.connect(lambda checked, m=mode: self.ui.samplechoice.setText(m))  # 设置菜单项的点击事件
            menu.addAction(action)  # 添加菜单项
        self.ui.samplechoice.setMenu(menu)  # 将菜单添加到样品选择控件
        self.ui.samplechoice.setPopupMode(self.ui.samplechoice.MenuButtonPopup)  # 设置弹出模式
        self.ui.samplechoice.setText(modes[0])  # 默认选择第一个样品模式


    def run_simulation(self):
        """
        运行仿真函数，根据UI中的参数配置进行仿真计算。
        """
        try:
            cfg = InterfereData()  # 创建默认配置对象

            # 获取UI中的输入参数
            cfg.Wavelength_start = float(self.ui.Wavelength_start.text())
            cfg.Wavelength_end = float(self.ui.Wavelength_end.text())
            cfg.Wavelength_samples = int(self.ui.Wavelength_samples.text())
            cfg.Central_wavelength = (cfg.Wavelength_start + cfg.Wavelength_end) / 2  # 计算中心波长
            cfg.Phase_offset = float(self.ui.Phase_offset.text())

            # 计算相干长度
            wavelength_m = cfg.Central_wavelength * 1e-9
            bandwidth_m = (cfg.Wavelength_end - cfg.Wavelength_start) * 1e-9
            Lc = coherence_length(wavelength_m, bandwidth_m)  # 计算相干长度
            self.ui.Coherence_length.setText(f"{Lc * 1e6:.3f}")  # 显示相干长度

            # CCD和扫描参数配置
            cfg.CCD_Pixels_X = int(self.ui.Screen_Weight.text())
            cfg.CCD_Pixels_Y = int(self.ui.Screen_Hight.text())
            cfg.CCD_Distance = float(self.ui.CCD_Distance.text())
            cfg.Pixel_size = float(self.ui.Pixel_size.text())

            # 臂长参数
            cfg.Fixed_mirror_distance = float(self.ui.ReflectMirror_distance.text())
            cfg.Sample_postion_distance = float(self.ui.SampleMirror_distance.text())

            # 扫描参数
            cfg.Scan_steps = int(self.ui.Scan_step_size.text())
            cfg.multipleLc = float(self.ui.multipleLc.text())

            # 样品参数
            cfg.Sample_width = float(self.ui.sample_width.text()) / 1e3  # 直接显示为mm
            cfg.Sample_length = float(self.ui.sample_length.text()) / 1e3  # 直接显示为mm
            cfg.Sample_mode = str(self.ui.samplechoice.text())  # 获取样品表面模式

            # 运行仿真并获取结果
            self.result = mirau_white_light_sim(cfg)
            self.Icube = self.result["Icube"]
            self.Z_true = self.result["Z_true"]
            self.Z_meas = self.result["Z_measured"]
            self.error = self.result["error"]

            # 设置滑块的最大值和初始值（根据仿真结果的分辨率）
            H_true, W_true = self.Z_true.shape
            self.ui.xSlider.setMinimum(0)
            self.ui.xSlider.setMaximum(H_true - 1)
            self.ui.xSlider.setValue(H_true // 2)
            self.ui.ySlider.setMinimum(0)
            self.ui.ySlider.setMaximum(W_true - 1)
            self.ui.ySlider.setValue(W_true // 2)

            # 显示仿真结果：测量形貌、真实形貌、误差
            self.show_image(self.Z_meas * 1e6, self.ui.InterfrerPlot, "jet")
            self.show_image(self.Z_true * 1e6, self.ui.ObjectPlot, "jet")
            self.show_image(self.error * 1e6, self.ui.errorplot, "bwr")

            # 初始绘制 X/Y 剖面
            self.plot_profiles()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"仿真失败：\n{str(e)}")  # 弹出错误对话框


    def plot_profiles(self):
        """
        绘制当前选定位置的 X/Y 剖面。
        根据滑块位置获取该位置的高度数据，并绘制真实和测量的剖面。
        """
        if self.Z_true is None or self.Z_meas is None:
            return

        # 获取高分辨率网格
        X_grid = self.result["X_true"]
        Y_grid = self.result["Y_true"]

        # 使用插值器将Z_meas转换为高分辨率
        px = self.result["px"]
        py = self.result["py"]
        interp = RegularGridInterpolator((py, px), self.Z_meas, bounds_error=False, fill_value=np.nan)

        # 获取滑块位置
        x_pos = self.ui.xSlider.value()
        y_pos = self.ui.ySlider.value()

        H, W = self.Z_true.shape

        # X 剖面：固定 Y = y_pos
        x_points = np.stack([np.full(W, Y_grid[y_pos, 0]), X_grid[y_pos, :]], axis=-1)
        y_meas_x = interp(x_points)
        y_true_x = self.Z_true[y_pos, :]
        self.plot_curve(np.arange(W), y_true_x, self.ui.xPlot, xlabel="X 像素", ylabel="高度/m", y2=y_meas_x)

        # Y 剖面：固定 X = x_pos
        y_points = np.stack([Y_grid[:, x_pos], np.full(H, X_grid[0, x_pos])], axis=-1)
        y_meas_y = interp(y_points)
        y_true_y = self.Z_true[:, x_pos]
        self.plot_curve(np.arange(H), y_true_y, self.ui.yPlot, xlabel="Y 像素", ylabel="高度/m", y2=y_meas_y)


    def show_image(self, img, label, cmap="jet"):
        """
        显示图像，并将图像渲染到给定的label控件中。
        """
        w, h = label.width(), label.height()
        if w <= 10 or h <= 10:
            return
        dpi = 100
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
        im = ax.imshow(img, cmap=cmap, aspect='auto')
        ax.axis("off")
        ax.set_position([0, 0, 0.85, 1])
        cbar_ax = fig.add_axes([0.87, 0.1, 0.03, 0.8])
        plt.colorbar(im, cax=cbar_ax, label="高度 (um)")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        qimg = QtGui.QImage(buf.tobytes(), buf.shape[1], buf.shape[0], 3*buf.shape[1],
                            QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        plt.close(fig)


    def plot_curve(self, x, y, label, xlabel="", ylabel="", y2=None):
        """
        绘制曲线，显示剖面数据。
        """
        w, h = label.width(), label.height()
        if w <= 10 or h <= 10:
            return
        dpi = 100
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
        ax.plot(x, y, 'b-', label='真实')
        if y2 is not None:
            ax.plot(x, y2, 'r--', label='测量')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        ax.set_position([0.15, 0.15, 0.8, 0.8])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        qimg = QtGui.QImage(buf.tobytes(), buf.shape[1], buf.shape[0], 3*buf.shape[1],
                            QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        plt.close(fig)


    def eventFilter(self, obj, event):
        """
        事件过滤器，当控件大小变化时，自动刷新显示内容。
        """
        if event.type() == QtCore.QEvent.Resize and self.result is not None:
            if obj == self.ui.InterfrerPlot:
                self.show_image(self.Z_meas * 1e6, self.ui.InterfrerPlot, "jet")
            elif obj == self.ui.ObjectPlot:
                self.show_image(self.Z_true * 1e6, self.ui.ObjectPlot, "jet")
            elif obj == self.ui.errorplot:
                self.show_image(self.error * 1e6, self.ui.errorplot, "bwr")
            elif obj in (self.ui.xPlot, self.ui.yPlot):
                self.plot_profiles()
            return True
        return super().eventFilter(obj, event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = WhiteInterfereApp()
    win.show()
    sys.exit(app.exec_())
