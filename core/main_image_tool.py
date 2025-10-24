import cv2
import numpy as np
from PIL import Image

def imread_unicode(image_path):
    """
    支持中文路径的图片读取函数
    
    参数:
        image_path: 图片路径（支持中文）
    
    返回:
        image: 读取的图片，如果失败返回None
    """
    try:
        # 使用PIL读取图片，然后转换为OpenCV格式
        pil_image = Image.open(image_path)
        # 转换为RGB（PIL默认是RGB，OpenCV是BGR）
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        # 转换为numpy数组
        cv_image = np.array(pil_image)
        # 转换为BGR格式（OpenCV标准）
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        return cv_image
    except Exception as e:
        print(f"无法读取图片 {image_path}: {str(e)}")
        return None

def imwrite_unicode(image_path, img):
    """
    支持中文路径的图片保存函数
    """
    try:
        # 将OpenCV图片（BGR）转换为PIL图片（RGB）
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
        else:
            pil_image = Image.fromarray(img)
        
        # 使用PIL保存，支持中文路径
        pil_image.save(image_path)
        return True
    except Exception as e:
        print(f"无法保存图片到 {image_path}: {str(e)}")
        return False

def detect_yellow_region(image_path, sensitivity='medium'):
    """
    检测图片中的黄色区域
    
    参数:
        image_path: 输入图片路径
        sensitivity: 检测灵敏度 'low'(宽松), 'medium'(中等), 'high'(严格)
    
    返回:
        image: 原始图片
        mask: 黄色区域的掩码
        contours: 黄色区域的轮廓
    """
    # 读取图片（支持中文路径）
    img = imread_unicode(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None, None, []
    
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 改进的黄色检测 - 扩大HSV范围，提高检测覆盖率
    if sensitivity == 'low':
        # 宽松模式 - 检测更多黄色变体
        lower_yellow = np.array([10, 60, 60])    # 更宽的色调范围
        upper_yellow = np.array([50, 255, 255])
    elif sensitivity == 'high':
        # 严格模式 - 但仍然比原来宽松一些
        lower_yellow = np.array([18, 120, 120])
        upper_yellow = np.array([42, 255, 255])
    else:
        # 中等模式（默认）- 扩大范围以提高检测率
        lower_yellow = np.array([15, 80, 80])    # 扩大色调和饱和度范围
        upper_yellow = np.array([45, 255, 255])
    
    # 创建黄色掩码
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 改进的形态学操作 - 减少区域丢失
    # 使用更小的核和更少的迭代次数
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # 先进行轻微的闭运算填充小孔
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    # 可选的开运算去除噪点（更保守）
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 最后进行一次闭运算连接断开的区域
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    
    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return img, mask, contours


def detect_green_region(image_path, sensitivity='medium'):
    """
    检测图片中的绿色区域
    
    参数:
        image_path: 输入图片路径
        sensitivity: 检测灵敏度 'low'(宽松), 'medium'(中等), 'high'(严格)
    
    返回:
        image: 原始图片
        mask: 绿色区域的掩码
        contours: 绿色区域的轮廓
    """
    # 读取图片（支持中文路径）
    img = imread_unicode(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None, None, []
    
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 绿色检测的HSV范围
    if sensitivity == 'low':
        # 宽松模式 - 检测更多绿色变体
        lower_green = np.array([35, 40, 40])    # 更宽的色调范围
        upper_green = np.array([85, 255, 255])
    elif sensitivity == 'high':
        # 严格模式 - 检测纯绿色
        lower_green = np.array([45, 100, 100])
        upper_green = np.array([75, 255, 255])
    else:
        # 中等模式（默认）
        lower_green = np.array([40, 60, 60])    # 平衡的绿色范围
        upper_green = np.array([80, 255, 255])
    
    # 创建绿色掩码
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 改进的形态学操作 - 减少区域丢失
    # 使用更小的核和更少的迭代次数
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # 先进行轻微的闭运算填充小孔
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    # 可选的开运算去除噪点（更保守）
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 最后进行一次闭运算连接断开的区域
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    
    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return img, mask, contours


def get_quadrilateral_corners(contour, expand_pixels=2):
    """
    获取轮廓的四个角点（用于透视变换）
    
    参数:
        contour: 轮廓
        expand_pixels: 向外扩展的像素数，用于减少边缘遗留
    
    返回:
        corners: 四个角点坐标 [左上, 右上, 右下, 左下]
    """
    # 使用多边形逼近，减小epsilon以获得更精确的近似
    epsilon = 0.015 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # 如果逼近结果是四边形，直接使用
    if len(approx) == 4:
        points = approx.reshape(4, 2)
    else:
        # 否则使用最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        points = np.int32(box)
    
    # 对点进行排序：左上、右上、右下、左下
    # 通过计算点的和与差来排序
    sum_pts = points.sum(axis=1)
    diff_pts = np.diff(points, axis=1)
    
    corners = np.zeros((4, 2), dtype=np.float32)
    corners[0] = points[np.argmin(sum_pts)]      # 左上 (x+y最小)
    corners[1] = points[np.argmin(diff_pts)]     # 右上 (x-y最小)
    corners[2] = points[np.argmax(sum_pts)]      # 右下 (x+y最大)
    corners[3] = points[np.argmax(diff_pts)]     # 左下 (x-y最大)
    
    # 向外扩展像素以获得更完整的边界，减少边缘遗留
    corners[0] -= [expand_pixels, expand_pixels]  # 左上角向左上扩展
    corners[1] += [expand_pixels, -expand_pixels] # 右上角向右上扩展
    corners[2] += [expand_pixels, expand_pixels]  # 右下角向右下扩展
    corners[3] += [-expand_pixels, expand_pixels] # 左下角向左下扩展
    
    return corners


def create_enhanced_mask(contour, img_shape, expand_pixels=2):
    """
    创建增强的掩码，包含边缘扩展和平滑处理，用于减少边缘遗留颜色
    
    参数:
        contour: 轮廓
        img_shape: 图像形状
        expand_pixels: 扩展像素数
    
    返回:
        enhanced_mask: 增强的掩码
    """
    # 创建基础掩码
    base_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(base_mask, [contour], 255)
    
    # 轻微膨胀以扩展边缘
    kernel = np.ones((expand_pixels*2+1, expand_pixels*2+1), np.uint8)
    expanded_mask = cv2.dilate(base_mask, kernel, iterations=1)
    
    # 高斯模糊以平滑边缘
    blurred_mask = cv2.GaussianBlur(expanded_mask.astype(np.float32), (5, 5), 1.0)
    
    # 归一化到0-255范围
    enhanced_mask = (blurred_mask / np.max(blurred_mask) * 255).astype(np.uint8)
    
    return enhanced_mask


def get_region_direction(yellow_contour, green_contour):
    """
    判断绿色区域相对于黄色区域的方向
    
    参数:
        yellow_contour: 黄色区域轮廓
        green_contour: 绿色区域轮廓
    
    返回:
        direction: 'top', 'bottom', 'left', 'right'
        distance: 两个区域中心点之间的距离
    """
    # 计算两个轮廓的中心点
    yellow_moments = cv2.moments(yellow_contour)
    green_moments = cv2.moments(green_contour)
    
    if yellow_moments['m00'] == 0 or green_moments['m00'] == 0:
        return 'unknown', 0
    
    yellow_center_x = int(yellow_moments['m10'] / yellow_moments['m00'])
    yellow_center_y = int(yellow_moments['m01'] / yellow_moments['m00'])
    
    green_center_x = int(green_moments['m10'] / green_moments['m00'])
    green_center_y = int(green_moments['m01'] / green_moments['m00'])
    
    # 计算相对位置
    dx = green_center_x - yellow_center_x
    dy = green_center_y - yellow_center_y
    
    # 计算距离
    distance = np.sqrt(dx**2 + dy**2)
    
    # 判断主要方向（基于绝对值较大的轴）
    if abs(dx) > abs(dy):
        # 水平方向为主
        if dx > 0:
            return 'right', distance
        else:
            return 'left', distance
    else:
        # 垂直方向为主
        if dy > 0:
            return 'bottom', distance
        else:
            return 'top', distance


def crop_image_by_direction(image, green_contour, direction):
    """
    根据绿色区域的方向和尺寸裁剪图片，并进行相应的翻转
    
    参数:
        image: 要裁剪的图片
        green_contour: 绿色区域轮廓
        direction: 绿色区域相对于黄色区域的方向
    
    返回:
        processed_image: 裁剪和翻转后的图片
    """
    h, w = image.shape[:2]
    
    # 获取绿色区域的边界框
    x, y, green_w, green_h = cv2.boundingRect(green_contour)
    
    if direction in ['top', 'bottom']:
        # 上下方向：按绿色区域高度裁剪并翻转
        if direction == 'top':
            # 绿色在上方：从顶部裁剪并上下翻转
            cropped_image = image[0:green_h, :]
            processed_image = cv2.flip(cropped_image, 0)  # 0表示上下翻转
        else:  # bottom
            # 绿色在下方：从底部裁剪并上下翻转
            cropped_image = image[h-green_h:h, :]
            processed_image = cv2.flip(cropped_image, 0)  # 0表示上下翻转
    else:  # left or right
        # 左右方向：按绿色区域宽度裁剪并翻转
        if direction == 'left':
            # 绿色在左侧：从左侧裁剪并左右翻转
            cropped_image = image[:, 0:green_w]
            processed_image = cv2.flip(cropped_image, 1)  # 1表示左右翻转
        else:  # right
            # 绿色在右侧：从右侧裁剪并左右翻转
            cropped_image = image[:, w-green_w:w]
            processed_image = cv2.flip(cropped_image, 1)  # 1表示左右翻转
    
    return processed_image


def fill_yellow_region(original_img_path, fill_img_path, output_path, 
                       mode='perspective', show_corners=False,
                       anti_aliasing=True, edge_blur=3, sensitivity='high'):
    """
    将一张图片填充到黄色和绿色区域
    
    参数:
        original_img_path: 包含黄色和/或绿色区域的原始图片路径
        fill_img_path: 要填充的图片路径
        output_path: 输出图片路径
        mode: 填充模式
              'perspective' - 透视变换（角对角）
              'stretch' - 拉伸填充
              'tile' - 平铺填充
        show_corners: 是否在结果图上显示检测到的角点（黄色区域用红点，绿色区域用绿点）
        anti_aliasing: 是否启用抗锯齿（边缘羽化）
        edge_blur: 边缘羽化程度，数值越大越平滑（建议1-5之间的奇数，默认3）
        sensitivity: 颜色检测灵敏度 'low'(宽松), 'medium'(中等), 'high'(严格)，同时应用于黄色和绿色检测
    
    返回:
        result: 处理后的图片
    """
    # 检测黄色区域
    img, yellow_mask, yellow_contours = detect_yellow_region(original_img_path, sensitivity)
    
    # 检测绿色区域
    _, green_mask, green_contours = detect_green_region(original_img_path, sensitivity)
    
    # 合并所有检测到的区域
    all_contours = []
    region_types = []  # 记录每个轮廓的类型
    
    if len(yellow_contours) > 0:
        all_contours.extend(yellow_contours)
        region_types.extend(['yellow'] * len(yellow_contours))
        print(f"检测到 {len(yellow_contours)} 个黄色区域")
    
    if len(green_contours) > 0:
        all_contours.extend(green_contours)
        region_types.extend(['green'] * len(green_contours))
        print(f"检测到 {len(green_contours)} 个绿色区域")
    
    if len(all_contours) == 0:
        print("未检测到黄色或绿色区域！")
        return None
    
    # 合并黄色和绿色掩码
    combined_mask = cv2.bitwise_or(yellow_mask, green_mask)
    
    # 读取要填充的图片（支持中文路径）
    fill_img = imread_unicode(fill_img_path)
    
    if fill_img is None:
        print(f"无法读取图片: {fill_img_path}")
        return None
    
    # 创建结果图片的副本
    result = img.copy()
    
    if mode == 'perspective':
        # 透视变换模式 - 处理每个检测到的区域
        print("使用透视变换模式填充")
        
        # 填充图片的四个角点（按原图尺寸）
        h, w = fill_img.shape[:2]
        src_corners = np.array([
            [0, 0],           # 左上
            [w - 1, 0],       # 右上
            [w - 1, h - 1],   # 右下
            [0, h - 1]        # 左下
        ], dtype=np.float32)
        
        # 处理每个检测到的区域
        for i, contour in enumerate(all_contours):
            region_type = region_types[i]
            
            # 获取当前区域的四个角点，使用更大的扩展像素数以减少边缘遗留
            dst_corners = get_quadrilateral_corners(contour, expand_pixels=4)
            
            print(f"处理{region_type}区域 {i+1}:")
            print(f"  左上: {dst_corners[0]}")
            print(f"  右上: {dst_corners[1]}")
            print(f"  右下: {dst_corners[2]}")
            print(f"  左下: {dst_corners[3]}")
            
            # 准备填充图片
            current_fill_img = fill_img.copy()
            
            # 如果是绿色区域，需要进行智能裁剪
            if region_type == 'green' and len(yellow_contours) > 0:
                # 找到最大的黄色区域作为参考
                yellow_ref_contour = max(yellow_contours, key=cv2.contourArea)
                
                # 判断绿色区域相对于黄色区域的方向
                direction, distance = get_region_direction(yellow_ref_contour, contour)
                print(f"  绿色区域位于黄色区域的{direction}方向，距离: {distance:.1f}像素")
                
                # 根据方向裁剪和翻转填充图片
                current_fill_img = crop_image_by_direction(fill_img, contour, direction)
                
                # 显示详细的处理信息
                process_info = ""
                if direction == 'top':
                    process_info = "从顶部裁剪 + 上下翻转"
                elif direction == 'bottom':
                    process_info = "从底部裁剪 + 上下翻转"
                elif direction == 'left':
                    process_info = "从左侧裁剪 + 左右翻转"
                else:  # right
                    process_info = "从右侧裁剪 + 左右翻转"
                
                print(f"  智能处理: {process_info}")
                print(f"  处理后尺寸: {current_fill_img.shape[:2]}")
                
                # 更新源角点（基于裁剪后的图片尺寸）
                h_crop, w_crop = current_fill_img.shape[:2]
                src_corners_current = np.array([
                    [0, 0],                    # 左上
                    [w_crop - 1, 0],          # 右上
                    [w_crop - 1, h_crop - 1], # 右下
                    [0, h_crop - 1]           # 左下
                ], dtype=np.float32)
            else:
                # 黄色区域或没有黄色参考时，使用原始源角点
                src_corners_current = src_corners
            
            # 计算透视变换矩阵
            matrix = cv2.getPerspectiveTransform(src_corners_current, dst_corners)
            
            # 应用透视变换（使用高质量插值）
            h_img, w_img = img.shape[:2]
            warped = cv2.warpPerspective(current_fill_img, matrix, (w_img, h_img), 
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=[0, 0, 0])
            
            # 创建增强掩码以减少边缘遗留
            current_mask = create_enhanced_mask(contour, img.shape, expand_pixels=3)
            
            # 边缘抗锯齿处理
            if anti_aliasing and edge_blur > 0:
                # 对掩码进行轻微模糊，实现边缘羽化
                mask_float = current_mask.astype(np.float32) / 255.0
                # 确保edge_blur是奇数
                blur_size = edge_blur if edge_blur % 2 == 1 else edge_blur + 1
                # 使用较小的sigma值，让羽化更自然
                sigma = blur_size / 3.0
                mask_blurred = cv2.GaussianBlur(mask_float, (blur_size, blur_size), sigma)
                
                # 增强边缘对比度，减少过渡区域
                # 使用gamma校正让过渡更陡峭
                mask_blurred = np.power(mask_blurred, 1.5)
                
                mask_3ch = cv2.merge([mask_blurred, mask_blurred, mask_blurred])
                
                # 使用羽化掩码进行alpha混合
                result = (warped * mask_3ch + result * (1 - mask_3ch)).astype(np.uint8)
            else:
                # 不使用抗锯齿，直接替换
                mask_3ch = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)
                result = np.where(mask_3ch == 255, warped, result)
            
            # 如果需要显示角点
            if show_corners:
                # 为不同类型的区域使用不同颜色的角点
                color = (0, 0, 255) if region_type == 'yellow' else (0, 255, 0)  # 黄色区域用红点，绿色区域用绿点
                for j, corner in enumerate(dst_corners):
                    cv2.circle(result, tuple(corner.astype(int)), 10, color, -1)
                    cv2.putText(result, f"{region_type[0]}{i+1}-{j}", tuple(corner.astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        print(f"未知的填充模式: {mode}")
        return None
    
    # 后处理：清理残留的边缘颜色
    # result = post_process_edge_cleanup(result, img, combined_mask)
    
    # 保存结果（支持中文路径）
    if imwrite_unicode(output_path, result):
        print(f"结果已保存到: {output_path}")
        return result
    else:
        print(f"保存失败: {output_path}")
        return None


def post_process_edge_cleanup(processed_img, original_img, mask, threshold=25):
    """
    后处理清理残留的边缘颜色
    
    参数:
        processed_img: 处理后的图像
        original_img: 原始图像
        mask: 处理区域的掩码
        threshold: 颜色差异阈值
    
    返回:
        cleaned_img: 清理后的图像
    """
    # 创建更精确的边缘区域掩码
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=3)
    eroded_mask = cv2.erode(mask, kernel, iterations=3)
    edge_mask = dilated_mask - eroded_mask
    
    # 在边缘区域查找颜色差异较小的像素（可能是残留的原始颜色）
    diff = np.abs(processed_img.astype(np.float32) - original_img.astype(np.float32))
    color_diff = np.sqrt(np.sum(diff**2, axis=2))
    
    # 找到边缘区域中颜色差异小于阈值的像素
    residual_pixels = (edge_mask > 0) & (color_diff < threshold)
    
    if np.any(residual_pixels):
        # 对这些像素进行修复
        cleaned_img = processed_img.copy()
        
        # 使用更智能的修复策略
        residual_coords = np.where(residual_pixels)
        
        for i in range(len(residual_coords[0])):
            y, x = residual_coords[0][i], residual_coords[1][i]
            
            # 扩大搜索范围，寻找周围的非残留像素
            found_replacement = False
            for radius in range(1, 6):  # 逐渐扩大搜索半径
                y_min = max(0, y - radius)
                y_max = min(processed_img.shape[0], y + radius + 1)
                x_min = max(0, x - radius)
                x_max = min(processed_img.shape[1], x + radius + 1)
                
                # 获取搜索区域
                search_region = processed_img[y_min:y_max, x_min:x_max]
                mask_region = mask[y_min:y_max, x_min:x_max]
                residual_region = residual_pixels[y_min:y_max, x_min:x_max]
                
                # 找到掩码内且非残留的像素
                valid_mask = (mask_region > 128) & (~residual_region)
                
                if np.any(valid_mask):
                    # 使用距离加权平均
                    valid_coords = np.where(valid_mask)
                    if len(valid_coords[0]) > 0:
                        # 计算距离权重
                        center_y, center_x = radius, radius
                        distances = np.sqrt((valid_coords[0] - center_y)**2 + (valid_coords[1] - center_x)**2)
                        weights = 1.0 / (distances + 1e-6)  # 避免除零
                        weights = weights / np.sum(weights)
                        
                        # 加权平均颜色
                        valid_pixels = search_region[valid_mask]
                        weighted_color = np.average(valid_pixels, axis=0, weights=weights)
                        cleaned_img[y, x] = weighted_color.astype(np.uint8)
                        found_replacement = True
                        break
            
            # 如果找不到合适的替换像素，使用简单的邻域平均
            if not found_replacement:
                y_min = max(0, y - 1)
                y_max = min(processed_img.shape[0], y + 2)
                x_min = max(0, x - 1)
                x_max = min(processed_img.shape[1], x + 2)
                
                neighborhood = processed_img[y_min:y_max, x_min:x_max]
                mask_neighborhood = mask[y_min:y_max, x_min:x_max]
                
                valid_pixels = neighborhood[mask_neighborhood > 128]
                if len(valid_pixels) > 0:
                    cleaned_img[y, x] = np.mean(valid_pixels, axis=0).astype(np.uint8)
        
        print(f"  检测到 {np.sum(residual_pixels)} 个遗留像素，进行清理")
        return cleaned_img
    
    return processed_img


# 使用示例
if __name__ == "__main__":
    print("使用示例：")
    print("1. 透视变换（默认）: fill_yellow_region('moban.jpg', 'shitu.jpg', 'shuchu.jpg', mode='perspective')")
    fill_yellow_region('moban.jpg', 'shitu.jpg', 'shuchu.jpg', mode='perspective')
