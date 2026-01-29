"""
imgApp.py - 光学系统成像模拟主应用程序
"""

import sys
import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from Ui_Lens_img import Ui_MainWindow
from img_data import OpticalParams
from img_cal import (
    load_image, aperture_mask, lens_phase, 
    fresnel_propagation_fft, geometric_focus_and_mag
)
from lens_plot import create_optical_plot
from errorcal import calculate_image_metrics


class ImageCanvas(FigureCanvas):
    """带工具栏的图像画布"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(ImageCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    def show_image(self, image, title="", cmap='gray'):
        """显示图像"""
        self.ax.clear()

        image_display = image

        self.ax.imshow(image_display, cmap=cmap)
        self.ax.set_title(title)
        self.ax.axis('off')
        self.fig.tight_layout()
        self.draw()
    
    def clear_canvas(self):
        """清除画布"""
        self.ax.clear()
        self.ax.axis('off')
        self.draw()


class ImgApp(QtWidgets.QMainWindow):
    """主应用程序类"""
    def __init__(self):
        super(ImgApp, self).__init__()
        
        # 初始化UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 初始化变量
        self.image_path = "photo/cat.png"  # 默认图片路径
        self.U0 = None
        self.U_lens = None
        self.params = None
        self.z_init = 20e-3  
        self.mag = 1.0  
        self.simulation_done = False  # 标记模拟是否已完成
        
        # 模拟动画相关
        self.simulation_timer = QtCore.QTimer()
        self.simulation_timer.timeout.connect(self.animate_simulation)
        self.simulation_direction = 1  
        self.simulation_step = 0
        self.simulation_total_steps = 50  
        self.z_sim_min = 0
        self.z_sim_max = 0
        
        # 设置默认值
        self.set_default_values()
        
        # 连接信号与槽
        self.connect_signals_slots()
        
        # 设置窗口标题
        self.setWindowTitle("菲涅尔衍射透镜成像模拟")
        
        # 加载默认图片
        self.load_default_image()
    
    def set_default_values(self):
        """设置默认参数值"""
        # 创建默认参数对象
        default_params = OpticalParams()
        
        # 转换为显示单位并设置到UI
        self.ui.Wavelength.setText(f"{default_params.wavelength * 1e9:.1f}")
        
        # 像素尺寸：米转微米
        self.ui.Psize.setText(f"{default_params.dx * 1e6:.1f}")
        
        # 图像尺寸
        self.ui.imgNx.setText(str(default_params.Nx))
        self.ui.imgNy.setText(str(default_params.Ny))
        
        # 物距
        self.ui.imgToLen.setText(f"{default_params.imageTolens * 1e3:.1f}")
        
        # 透镜焦距
        self.ui.Lensf1.setText(f"{default_params.f1 * 1e3:.1f}")

        self.ui.Lensf2.setText(f"{default_params.f2 * 1e3:.1f}")
        # 透镜间距
        self.ui.f1Tof2.setText(f"{default_params.f1Tof2 * 1e3:.1f}")

        self.ui.Phi1.setText(f"{default_params.radius_mm1 * 1e3:.1f}")
        self.ui.Phi2.setText(f"{default_params.radius_mm2 * 1e3:.1f}")
        
        # 初始Z值范围
        z_min_mm = 0
        z_max_mm = 200
        self.ui.Zmin.setText(str(z_min_mm))
        self.ui.Zmax.setText(str(z_max_mm))
        self.ui.Zvalue.setMinimum(z_min_mm)
        self.ui.Zvalue.setMaximum(z_max_mm)
        self.ui.Zvalue.setValue(20)  # 默认20mm
        
        # 初始化质量指标显示
        self.ui.MES_2.setText("0.000000")
        self.ui.RMSE_2.setText("0.000000")
        self.ui.PSNR_2.setText("0.00")
        self.ui.lineEdit_5.setText("0.000000")
        
        # 设置透镜数量选项
        self.lens_menu = QtWidgets.QMenu(self)
        self.lens_menu.addAction("无透镜", lambda: self.set_lens_mode(0))
        self.lens_menu.addAction("单透镜", lambda: self.set_lens_mode(1))
        self.lens_menu.addAction("双透镜", lambda: self.set_lens_mode(2))
        self.ui.Lensnum.setMenu(self.lens_menu)
        self.ui.Lensnum.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.set_lens_mode(1)  # 默认单透镜
    
    def set_lens_mode(self, mode):
        """设置透镜模式"""
        self.ui.Lensnum.setText(["无透镜", "单透镜", "双透镜"][mode])
    
    def connect_signals_slots(self):
        """连接信号与槽函数"""
        # 按钮信号
        self.ui.imgChoice.clicked.connect(self.load_image_file)
        self.ui.PlotAll.clicked.connect(self.plot_all)  # 绘制所有
        self.ui.ZvalueSim.clicked.connect(self.start_z_simulation)  # 开始Z值模拟
        
        # 滑动条信号
        self.ui.Zvalue.valueChanged.connect(self.update_simulation)
        
        # 参数输入框信号
        self.ui.Wavelength.textChanged.connect(self.update_params)
        self.ui.Psize.textChanged.connect(self.update_params)
        self.ui.imgNx.textChanged.connect(self.update_params)
        self.ui.imgNy.textChanged.connect(self.update_params)
        self.ui.imgToLen.textChanged.connect(self.update_params)
        self.ui.Lensf1.textChanged.connect(self.update_params)
        self.ui.Lensf2.textChanged.connect(self.update_params)
        self.ui.f1Tof2.textChanged.connect(self.update_params)
        self.ui.Phi1.textChanged.connect(self.update_params)
        self.ui.Phi2.textChanged.connect(self.update_params)
        # 透镜数量选择
        self.ui.Lensnum.triggered.connect(self.update_lens_mode)
        self.ui.SetZinit.textChanged.connect(self.update_params)
        self.ui.checkSetZinit.stateChanged.connect(self.update_params)
    
    def load_default_image(self):
        """加载默认图片（只显示原图，不运行模拟）"""
        if os.path.exists(self.image_path):
            try:
                # 加载图像
                self.U0, Nx, Ny = load_image(self.image_path, dx=4e-6, dy=4e-6)
                
                # 更新UI中的图像大小
                self.ui.imgNx.setText(str(Nx))
                self.ui.imgNy.setText(str(Ny))
                # 更新参数
                self.update_params()
                # 重置模拟状态
                self.simulation_done = False
                # 清除结果图像和光路图
                self.clear_simulation_results()
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载默认图像失败: {str(e)}")
        else:
            QMessageBox.warning(self, "警告", f"默认图片不存在: {self.image_path}\n请选择其他图片。")
    
    def show_original_image(self, image):
        """显示原图到Readimg区域（带工具栏）"""
        # 创建画布和工具栏
        canvas = ImageCanvas(self, width=5, height=4)
        canvas.show_image(image, "原图")
        
        # 创建工具栏
        toolbar = NavigationToolbar(canvas, self)
        
        # 创建布局
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        # 清除原Readimg并设置新布局
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        
        # 替换原有的Readimg
        if hasattr(self, 'original_widget'):
            self.ui.horizontalLayout_2.replaceWidget(self.original_widget, widget)
            self.original_widget.deleteLater()
        else:
            self.ui.Readimg.setParent(None)
            self.ui.horizontalLayout_2.insertWidget(0, widget)
        
        self.original_widget = widget
        self.original_canvas = canvas
    
    def show_result_image(self, image, title=""):
        """显示结果图像到LensShowimg区域"""
        # 如果已有结果画布，直接更新
        if hasattr(self, 'result_canvas') and self.result_canvas:
            self.result_canvas.show_image(image, title)
            return
        
        # 创建画布和工具栏
        canvas = ImageCanvas(self, width=5, height=4)
        canvas.show_image(image, title)
        
        # 创建工具栏
        toolbar = NavigationToolbar(canvas, self)
        
        # 创建布局
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        # 清除原LensShowimg并设置新布局
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        
        # 替换原有的LensShowimg
        if hasattr(self, 'result_widget'):
            self.ui.horizontalLayout_2.replaceWidget(self.result_widget, widget)
            self.result_widget.deleteLater()
        else:
            self.ui.LensShowimg.setParent(None)
            self.ui.horizontalLayout_2.insertWidget(1, widget)
        
        self.result_widget = widget
        self.result_canvas = canvas
    
    def update_lens_mode(self, action):
        """更新透镜模式"""
        mode_text = action.text()
        if mode_text == "无透镜":
            self.set_lens_mode(0)
        elif mode_text == "单透镜":
            self.set_lens_mode(1)
        else:  # "双透镜"
            self.set_lens_mode(2)
        self.update_params()
        # 重置模拟状态
        self.simulation_done = False
    
    def load_image_file(self):
        """加载图像文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)", 
            options=options
        )
        
        if file_path:
            self.image_path = file_path
            try:
                # 加载图像
                self.U0, Nx, Ny = load_image(file_path, dx=4e-6, dy=4e-6)
                
                # 更新UI中的图像大小
                self.ui.imgNx.setText(str(Nx))
                self.ui.imgNy.setText(str(Ny))
                
                # 更新参数
                self.update_params()
                
                # 重置模拟状态
                self.simulation_done = False
                
                # 清除结果图像和光路图
                self.clear_simulation_results()
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")
    
    def clear_simulation_results(self):
        """清除模拟结果"""
        # 清除结果图像
        if hasattr(self, 'result_canvas') and self.result_canvas:
            self.result_canvas.clear_canvas()
        
        # 清除光路图
        self.ui.LightroadShow.clear()
        
        # 重置显示信息
        self.ui.imgdistance.setText("0.0")
        self.ui.multi.setText("1.000")
        
        # 重置质量指标
        self.ui.MES_2.setText("0.000000")
        self.ui.RMSE_2.setText("0.000000")
        self.ui.PSNR_2.setText("0.00")
        self.ui.lineEdit_5.setText("0.000000")
        
        # 停止模拟动画
        if self.simulation_timer.isActive():
            self.simulation_timer.stop()
    
    def update_params(self):
        """更新光学参数"""
        try:
            # 获取透镜模式
            mode_text = self.ui.Lensnum.text()
            if mode_text == "无透镜":
                mode = 0
            elif mode_text == "单透镜":
                mode = 1
            else:  # "双透镜"
                mode = 2
            
            # 获取图像尺寸
            Nx = int(self.ui.imgNx.text()) if self.ui.imgNx.text() else 512
            Ny = int(self.ui.imgNy.text()) if self.ui.imgNy.text() else 512
            
            # 波长
            wavelength = float(self.ui.Wavelength.text()) * 1e-9 if self.ui.Wavelength.text() else 671e-9
            
            # 像素尺寸
            psize_m = float(self.ui.Psize.text()) * 1e-6 if self.ui.Psize.text() else 4e-6
            
            # 物距
            imageTolens = float(self.ui.imgToLen.text()) * 1e-3 if self.ui.imgToLen.text() else 150e-3
            
            # 透镜焦距
            f1 = float(self.ui.Lensf1.text()) * 1e-3 if self.ui.Lensf1.text() else 100e-3
            f2 = float(self.ui.Lensf2.text()) * 1e-3 if self.ui.Lensf2.text() else 200e-3
            
            # 透镜间距
            f1Tof2 = float(self.ui.f1Tof2.text()) * 1e-3 if self.ui.f1Tof2.text() else 100e-3
            
            radius_mm1 = float(self.ui.Phi1.text()) * 1e-3 if self.ui.Phi1.text() else 10e-3
            radius_mm2 = float(self.ui.Phi2.text()) * 1e-3 if self.ui.Phi2.text() else 10e-3
            # 创建参数对象
            self.params = OpticalParams(
                wavelength=wavelength,
                dx=psize_m,
                dy=psize_m,
                Nx=Nx,
                Ny=Ny,
                imageTolens=imageTolens,
                f1=f1,
                f2=f2,
                f1Tof2=f1Tof2,
                z=self.z_init,
                mode=mode,
                radius_mm1=radius_mm1,
                radius_mm2=radius_mm2
            )
            
            # 重置模拟状态，因为参数已改变
            self.simulation_done = False
            
        except ValueError as e:
            print(f"参数更新错误: {e}")
            # 设置默认参数
            self.params = OpticalParams()
            self.simulation_done = False
            self.clear_simulation_results()
    
    def plot_all(self):
        """绘制所有（光路图和成像图）"""
        if not hasattr(self, 'original_canvas') or self.original_canvas is None:
            self.show_original_image(self.U0)
        else:
            self.original_canvas.show_image(self.U0, "原图")        
        if self.U0 is None:
            QMessageBox.warning(self, "警告", "请先加载图像！")
            return
        
        if self.params is None:
            self.update_params()
        
        try:
            QtWidgets.QApplication.processEvents()  # 更新UI
            
            # 透镜处理
            if self.params.mode == 0:
                self.U_lens = self.U0
            elif self.params.mode == 1:
                U0_img = fresnel_propagation_fft(self.U0, self.params, self.params.imageTolens)
                self.U_lens = lens_phase(U0_img, self.params, self.params.f1, radius_mm=self.params.radius_mm1)
            elif self.params.mode == 2:
                U1 = fresnel_propagation_fft(self.U0, self.params, self.params.imageTolens)
                U1 = lens_phase(U1, self.params, self.params.f1, radius_mm=self.params.radius_mm1)
                U2 = fresnel_propagation_fft(U1, self.params, self.params.f1Tof2)
                self.U_lens = lens_phase(U2, self.params, self.params.f2, radius_mm=self.params.radius_mm2)
            
            # 几何光学焦点及放大率
            if self.ui.checkSetZinit.isChecked():
                self.z_init = float(self.ui.SetZinit.text()) * 1e-3  
            else:
                self.z_init, self.mag = geometric_focus_and_mag(self.params)
            # 初始传播
            Uz = fresnel_propagation_fft(self.U_lens, self.params, self.z_init)

            I = np.abs(Uz)**2
            # 如果I有值大于1，设为1
            I[I > 1] = 1.0
            # 显示结果图像
            self.show_result_image(I, f"像距 z = {self.z_init*1e3:.0f} mm, 放大率 M = {self.mag:.3f}")
            
            # 更新滑动条范围
            z_init_mm = self.z_init * 1e3
            z_min = max(0, z_init_mm - z_init_mm/2)
            z_max = z_init_mm + z_init_mm/2
            
            self.ui.Zmin.setText(str(int(z_min)))
            self.ui.Zmax.setText(str(int(z_max)))
            self.ui.Zvalue.setMinimum(int(z_min))
            self.ui.Zvalue.setMaximum(int(z_max))
            self.ui.Zvalue.setValue(int(z_init_mm))
            
            # 更新显示信息
            self.ui.imgdistance.setText(f"{z_init_mm:.1f}")
            self.ui.multi.setText(f"{self.mag:.3f}")
            
            # 计算图像质量指标
            metrics = calculate_image_metrics(self.U0, I, self.mag)
            self.ui.MES_2.setText(f"{metrics['mse']:.6f}")
            self.ui.RMSE_2.setText(f"{metrics['rmse']:.6f}")
            self.ui.PSNR_2.setText(f"{metrics['psnr']:.2f}")
            self.ui.lineEdit_5.setText(f"{metrics['corr_coef']:.6f}")
            
            # 绘制光路图
            self.plot_optical_system()
            
            # 标记模拟已完成
            self.simulation_done = True
            
        except Exception as e:
            QMessageBox.critical(self, "模拟错误", f"模拟过程中发生错误: {str(e)}")
            print(f"模拟错误详情: {e}")
            self.ui.LightroadShow.setText(f"模拟失败: {str(e)[:50]}")
    
    def start_z_simulation(self):
        """开始Z值模拟（在像距附近移动）"""
        if not self.simulation_done:
            QMessageBox.warning(self, "警告", "请先点击'绘制所有'进行初始模拟！")
            return
        
        try:
            # 获取当前模式
            mode = self.params.mode
            
            # 根据模式计算模拟范围
            if mode == 0:  # 无透镜模式
                # 无透镜模式：从0到25mm
                self.z_sim_min = 0
                self.z_sim_max = 25
            else:
                # 有透镜模式：在像距附近移动
                z_init_mm = self.z_init * 1e3  # 转换为mm
                self.z_sim_min = max(0, z_init_mm - z_init_mm / 4)
                self.z_sim_max = z_init_mm + z_init_mm / 4
            
            # 确保最小值为正
            self.z_sim_min = max(0, self.z_sim_min)
            
            # 设置初始值
            self.simulation_step = 0
            self.simulation_direction = 1
            
            # 更新滑动条范围
            self.ui.Zmin.setText(str(int(self.z_sim_min)))
            self.ui.Zmax.setText(str(int(self.z_sim_max)))
            self.ui.Zvalue.setMinimum(int(self.z_sim_min))
            self.ui.Zvalue.setMaximum(int(self.z_sim_max))
            
            # 根据模式设置初始值
            if mode == 0:
                self.ui.Zvalue.setValue(int(self.z_sim_max / 2))  # 无透镜模式从中间开始
            else:
                self.ui.Zvalue.setValue(int(self.z_sim_min))  # 有透镜模式从最小值开始
            
            # 开始定时器模拟
            self.simulation_timer.start(10)  # 每10ms更新一次
            
            # 更新按钮文本
            self.ui.ZvalueSim.setText("停止模拟")
            self.ui.ZvalueSim.clicked.disconnect()
            self.ui.ZvalueSim.clicked.connect(self.stop_z_simulation)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"开始Z值模拟失败: {str(e)}")
    
    def stop_z_simulation(self):
        """停止Z值模拟"""
        if self.simulation_timer.isActive():
            self.simulation_timer.stop()
        
        # 恢复按钮状态
        self.ui.ZvalueSim.setText("Z值模拟")
        self.ui.ZvalueSim.clicked.disconnect()
        self.ui.ZvalueSim.clicked.connect(self.start_z_simulation)
    
    def animate_simulation(self):
        """动画模拟更新"""
        if not self.simulation_done:
            return
        
        try:
            # 计算当前Z值
            progress = self.simulation_step / self.simulation_total_steps
            z_mm = self.z_sim_min + (self.z_sim_max - self.z_sim_min) * progress
            
            # 更新滑动条值
            self.ui.Zvalue.setValue(int(z_mm))
            
            # 更新步数
            self.simulation_step += self.simulation_direction
            
            # 如果到达边界，反向
            if self.simulation_step >= self.simulation_total_steps:
                self.simulation_direction = -1
            elif self.simulation_step <= 0:
                self.simulation_direction = 1
            
            # 如果回到起点，停止模拟
            if self.simulation_step == 0 and self.simulation_direction == -1:
                self.stop_z_simulation()
            
        except Exception as e:
            print(f"动画模拟错误: {e}")
            self.stop_z_simulation()
    
    def update_simulation(self):
        """更新模拟结果（滑动条变化时）- 只更新已完成的模拟"""
        if not self.simulation_done:
            return
        
        if self.U_lens is None or self.params is None:
            return
        
        try:
            # 获取当前传播距离
            z_mm = self.ui.Zvalue.value()
            z = z_mm * 1e-3  
            
            # 传播计算
            Uz = fresnel_propagation_fft(self.U_lens, self.params, z)
            I = np.abs(Uz)**2
            # 如果I有值大于1，设为1
            I[I > 1] = 1.0
            # 更新显示结果图像
            self.show_result_image(I, f"传播后强度 z = {z_mm:.0f} mm, 放大率 M = {self.mag:.3f}")
            
            # 更新光路图
            self.plot_optical_system()
            
        except Exception as e:
            print(f"更新模拟错误: {e}")
    
    def plot_optical_system(self):
        """绘制光路图（在LightroadShow中显示）"""
        if self.params is None:
            return
        
        try:
            # 获取当前传播距离
            z_mm = self.ui.Zvalue.value()
            
            # 使用lens_plot模块创建光路图
            pixmap = create_optical_plot(self.params, z_mm, self.mag, self.ui.LightroadShow.size())
            
            if pixmap:
                # 显示在LightroadShow中
                self.ui.LightroadShow.setPixmap(pixmap)
            else:
                self.ui.LightroadShow.setText("光路图绘制失败")
                
        except Exception as e:
            print(f"绘制光路图错误: {e}")
            self.ui.LightroadShow.setText(f"光路图绘制失败: {str(e)}")
    
    def resizeEvent(self, event):
        """窗口大小改变时的事件处理"""
        super(ImgApp, self).resizeEvent(event)


def main():
    """主函数"""
    app = QtWidgets.QApplication(sys.argv)
    window = ImgApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()