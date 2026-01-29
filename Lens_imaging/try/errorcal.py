import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_image_metrics(original, processed, M, show_result=True):
    """
    original: 原始图像 
    processed: 仿真图像
    M: 放大率
    show_result: 是否显示对比图
    返回: 字典，包含 mse, rmse, psnr, corr_coef
    """
    try:
        # 获取仿真图（传感器）的尺寸
        target_h, target_w = processed.shape[:2]
        
        # 处理倒转 (实像 M < 0 时，图像翻转 180 度)
        if M < 0:
            # -1 表示水平和垂直同时翻转
            temp_img = cv2.flip(original, -1)
        else:
            temp_img = original.copy()

        # 物理缩放：按放大率 M 的绝对值计算预测像的像素大小
        abs_M = abs(M)
        # 预测像在传感器坐标系下应该占用的像素数
        scaled_w = int(target_w * abs_M)
        scaled_h = int(target_h * abs_M)
        
        # 执行缩放
        scaled_img = cv2.resize(
            temp_img, 
            (scaled_w, scaled_h), 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # 尺寸匹配 (裁剪或填充回 target_h x target_w)
        transformed_original = np.zeros((target_h, target_w), dtype=scaled_img.dtype)
        
        if abs_M >= 1.0:
            # 放大情况：从放大后的图中截取中心部分 
            y_start = (scaled_h - target_h) // 2
            x_start = (scaled_w - target_w) // 2
            # 切片确保不会超出索引范围
            transformed_original = scaled_img[
                y_start : y_start + target_h, 
                x_start : x_start + target_w
            ]
        else:
            # 缩小情况：将缩小后的图贴在黑色画布中心
            y_start = (target_h - scaled_h) // 2
            x_start = (target_w - scaled_w) // 2
            transformed_original[
                y_start : y_start + scaled_h, 
                x_start : x_start + scaled_w
            ] = scaled_img

        #绘图显示
        if show_result:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(transformed_original, cmap='gray')
            plt.title(f"预测像 (M={M:.2f}, 已处理裁剪/填充)")
            plt.axis('on')
            
            plt.subplot(1, 2, 2)
            plt.imshow(processed, cmap='gray')
            plt.title("仿真结果 (Processed)")
            plt.axis('on')
            
            plt.tight_layout()
            plt.show()

        # 归一化处理
        def normalize(img):
            v_max = np.max(img)
            return img / v_max if v_max > 0 else img

        original_norm = normalize(transformed_original)
        processed_norm = normalize(processed)
        
        # 计算指标
        mse = np.mean((original_norm - processed_norm) ** 2)
        rmse = np.sqrt(mse)
        
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 20 * np.log10(1.0 / rmse)
        
        # 计算相关系数
        if original_norm.size > 1:
            corr_matrix = np.corrcoef(original_norm.flatten(), processed_norm.flatten())
            corr_coef = corr_matrix[0, 1]
        else:
            corr_coef = 0
        
        return {
            'mse': mse, 
            'rmse': rmse, 
            'psnr': psnr, 
            'corr_coef': corr_coef
        }
        
    except Exception as e:
        print(f"计算图像质量指标错误: {e}")
        return {k: float('nan') for k in ['mse', 'rmse', 'psnr', 'corr_coef']}