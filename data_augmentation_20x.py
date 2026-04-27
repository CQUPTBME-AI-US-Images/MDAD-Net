"""
数据增强脚本：将数据集扩充至20倍
实现以下增强操作：
1. 随机旋转（[0°,20°]和[340°,357°]）
2. 翻转（沿x轴和y轴）
3. 弹性变换（α=10，σ=2，α-affine=2，random_state=None）
4. 高斯噪声添加（均值=0，δ=[5,10]）
5. 模糊处理（模糊核尺寸为3×3）
6. 随机伽马变换（γ=1.0）
"""

import os
import cv2
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from tqdm import tqdm
import random


def random_rotation(image, label, angle_ranges):
    """
    随机旋转图像和标签
    
    Args:
        image: 输入图像 (numpy array)
        label: 输入标签 (numpy array)
        angle_ranges: 角度范围列表，如 [[0, 20], [340, 357]]
    
    Returns:
        旋转后的图像和标签
    """
    # 随机选择一个角度范围
    angle_range = random.choice(angle_ranges)
    # 在该范围内随机选择角度
    angle = random.uniform(angle_range[0], angle_range[1])
    
    # 获取图像中心
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 旋转图像（使用双线性插值）
    if len(image.shape) == 3:
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, 
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, 
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 旋转标签（使用最近邻插值，保持标签值不变）
    rotated_label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST, 
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return rotated_image, rotated_label


def random_flip(image, label):
    """
    随机翻转图像和标签（沿x轴或y轴）
    
    Args:
        image: 输入图像
        label: 输入标签
    
    Returns:
        翻转后的图像和标签
    """
    flip_code = random.choice([-1, 0, 1])  # -1: 同时翻转x和y, 0: 垂直翻转, 1: 水平翻转
    
    flipped_image = cv2.flip(image, flip_code)
    flipped_label = cv2.flip(label, flip_code)
    
    return flipped_image, flipped_label


def elastic_transform(image, label, alpha=10, sigma=2, alpha_affine=2, random_state=None):
    """
    弹性变换（Elastic Transform）
    
    Args:
        image: 输入图像
        label: 输入标签
        alpha: 变形强度
        sigma: 高斯核标准差
        alpha_affine: 仿射变换强度
        random_state: 随机种子
    
    Returns:
        变换后的图像和标签
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image.shape
    shape_size = shape[:2]
    
    # 随机仿射变换
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, 
                       [center_square[0] + square_size, center_square[1] - square_size], 
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    
    # 对图像进行仿射变换
    if len(image.shape) == 3:
        image_affine = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    else:
        image_affine = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    
    # 对标签进行仿射变换
    label_affine = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    
    # 生成随机位移场
    if len(image.shape) == 3:
        dx = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha
    else:
        dx = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha
    
    # 创建坐标网格
    x, y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    # 应用弹性变换
    if len(image.shape) == 3:
        transformed_image = np.zeros_like(image_affine)
        for i in range(image.shape[2]):
            transformed_image[:, :, i] = map_coordinates(image_affine[:, :, i], indices, order=1, mode='constant').reshape(shape_size)
    else:
        transformed_image = map_coordinates(image_affine, indices, order=1, mode='constant').reshape(shape_size)
    
    # 标签使用最近邻插值
    transformed_label = map_coordinates(label_affine.astype(np.float32), indices, order=0, mode='constant').reshape(shape_size)
    transformed_label = transformed_label.astype(label.dtype)
    
    return transformed_image, transformed_label


def add_gaussian_noise(image, mean=0, std_range=[5, 10]):
    """
    添加高斯噪声（仅对图像，不对标签）
    
    Args:
        image: 输入图像
        mean: 噪声均值
        std_range: 噪声标准差范围
    
    Returns:
        添加噪声后的图像
    """
    std = random.uniform(std_range[0], std_range[1])
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    
    if len(image.shape) == 3:
        noisy_image = image.astype(np.float32) + noise
    else:
        noisy_image = image.astype(np.float32) + noise
    
    # 限制像素值范围
    noisy_image = np.clip(noisy_image, 0, 255).astype(image.dtype)
    
    return noisy_image


def apply_blur(image, kernel_size=3):
    """
    应用高斯模糊（仅对图像，不对标签）
    
    Args:
        image: 输入图像
        kernel_size: 模糊核尺寸（必须是奇数）
    
    Returns:
        模糊后的图像
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return blurred_image


def gamma_correction(image, gamma=1.0):
    """
    伽马变换（仅对图像，不对标签）
    
    Args:
        image: 输入图像
        gamma: 伽马值
    
    Returns:
        变换后的图像
    """
    # 归一化到[0, 1]
    if image.dtype != np.float32:
        image_normalized = image.astype(np.float32) / 255.0
    else:
        image_normalized = image / 255.0
    
    # 应用伽马变换
    gamma_corrected = np.power(image_normalized, gamma)
    
    # 转换回原始范围
    gamma_corrected = (gamma_corrected * 255.0).astype(image.dtype)
    
    return gamma_corrected


def augment_image_pair(image, label, augmentation_type):
    """
    对图像-标签对应用指定的数据增强
    
    Args:
        image: 输入图像
        label: 输入标签
        augmentation_type: 增强类型字符串
    
    Returns:
        增强后的图像和标签
    """
    aug_image = image.copy()
    aug_label = label.copy()
    
    # 几何变换（同时应用于图像和标签）
    if 'rotation' in augmentation_type:
        angle_ranges = [[0, 20], [340, 357]]
        aug_image, aug_label = random_rotation(aug_image, aug_label, angle_ranges)
    
    if 'flip' in augmentation_type:
        aug_image, aug_label = random_flip(aug_image, aug_label)
    
    if 'elastic' in augmentation_type:
        aug_image, aug_label = elastic_transform(aug_image, aug_label, 
                                                alpha=10, sigma=2, alpha_affine=2, random_state=None)
    
    # 仅对图像应用的变换
    if 'noise' in augmentation_type:
        aug_image = add_gaussian_noise(aug_image, mean=0, std_range=[5, 10])
    
    if 'blur' in augmentation_type:
        aug_image = apply_blur(aug_image, kernel_size=3)
    
    if 'gamma' in augmentation_type:
        aug_image = gamma_correction(aug_image, gamma=1.0)
    
    return aug_image, aug_label


def generate_augmentation_combinations():
    """
    生成19种不同的增强组合（加上原始图像共20倍）
    
    Returns:
        增强组合列表
    """
    augmentations = []
    
    # 单个增强操作
    augmentations.extend([
        ['rotation'],
        ['flip'],
        ['elastic'],
        ['noise'],
        ['blur'],
        ['gamma'],
    ])
    
    # 两个增强操作的组合
    augmentations.extend([
        ['rotation', 'flip'],
        ['rotation', 'elastic'],
        ['rotation', 'noise'],
        ['rotation', 'blur'],
        ['flip', 'elastic'],
        ['flip', 'noise'],
        ['flip', 'blur'],
        ['elastic', 'noise'],
        ['noise', 'blur'],
        ['blur', 'gamma'],
    ])
    
    # 三个增强操作的组合
    augmentations.extend([
        ['rotation', 'flip', 'noise'],
        ['rotation', 'flip', 'blur'],
        ['rotation', 'elastic', 'noise'],
    ])
    
    return augmentations


def augment_dataset(
    input_images_dir,
    input_labels_dir,
    output_images_dir,
    output_labels_dir,
    augmentation_factor=20
):
    """
    对数据集进行数据增强
    
    Args:
        input_images_dir: 输入图像目录
        input_labels_dir: 输入标签目录
        output_images_dir: 输出图像目录
        output_labels_dir: 输出标签目录
        augmentation_factor: 增强倍数（默认20倍）
    """
    # 创建输出目录
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_images_dir) 
                   if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    print(f"找到 {len(image_files)} 个原始图像文件")
    print(f"将生成 {len(image_files) * augmentation_factor} 个增强图像文件")
    
    # 生成增强组合
    augmentation_combinations = generate_augmentation_combinations()
    # 确保有足够的组合
    while len(augmentation_combinations) < augmentation_factor - 1:
        augmentation_combinations.extend(augmentation_combinations)
    augmentation_combinations = augmentation_combinations[:augmentation_factor - 1]
    
    # 处理每个图像
    total_files = len(image_files) * augmentation_factor
    with tqdm(total=total_files, desc="数据增强进度") as pbar:
        for img_file in image_files:
            # 读取原始图像和标签
            img_path = os.path.join(input_images_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            img_ext = os.path.splitext(img_file)[1]
            
            # 查找对应的标签文件
            label_file = None
            for f in os.listdir(input_labels_dir):
                if os.path.splitext(f)[0] == base_name:
                    label_file = f
                    break
            
            if label_file is None:
                print(f"警告: 未找到 {img_file} 对应的标签文件，跳过")
                continue
            
            label_path = os.path.join(input_labels_dir, label_file)
            
            # 读取图像和标签
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or label is None:
                print(f"警告: 无法读取 {img_file} 或对应的标签文件，跳过")
                continue
            
            # 保存原始图像（第1个，不增强）
            original_img_path = os.path.join(output_images_dir, f"{base_name}_aug_00{img_ext}")
            original_label_path = os.path.join(output_labels_dir, f"{base_name}_aug_00{os.path.splitext(label_file)[1]}")
            cv2.imwrite(original_img_path, image)
            cv2.imwrite(original_label_path, label)
            pbar.update(1)
            
            # 生成增强版本
            for aug_idx, aug_combo in enumerate(augmentation_combinations, start=1):
                try:
                    aug_image, aug_label = augment_image_pair(image.copy(), label.copy(), aug_combo)
                    
                    # 生成文件名
                    aug_img_path = os.path.join(output_images_dir, 
                                               f"{base_name}_aug_{aug_idx:02d}{img_ext}")
                    aug_label_path = os.path.join(output_labels_dir, 
                                                 f"{base_name}_aug_{aug_idx:02d}{os.path.splitext(label_file)[1]}")
                    
                    # 保存增强后的图像和标签
                    cv2.imwrite(aug_img_path, aug_image)
                    cv2.imwrite(aug_label_path, aug_label)
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"警告: 增强 {img_file} (组合 {aug_combo}) 时出错: {str(e)}")
                    continue
    
    print(f"\n数据增强完成！")
    print(f"输出目录: {output_images_dir}")
    print(f"输出目录: {output_labels_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据增强脚本：将数据集扩充至20倍")
    parser.add_argument(
        "--input-images-dir",
        type=str,
        default=r"D:\rr\dataset\hos\Test_Images",
        help="输入图像目录 (默认: us_B_data/Training_Images)"
    )
    parser.add_argument(
        "--input-labels-dir",
        type=str,
        default=r"D:\rr\dataset\hos\Test_Labels",
        help="输入标签目录 (默认: us_B_data/Training_Labels)"
    )
    parser.add_argument(
        "--output-images-dir",
        type=str,
        default=r"D:\rr\dataset\hos\Test_Images_Augmented",
        help="输出图像目录 (默认: us_B_data/Training_Images_Augmented)"
    )
    parser.add_argument(
        "--output-labels-dir",
        type=str,
        default=r"D:\rr\dataset\hos\Test_Labels_Augmented",
        help="输出标签目录 (默认: us_B_data/Training_Labels_Augmented)"
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=20,
        help="增强倍数 (默认: 20)"
    )
    
    args = parser.parse_args()
    
    # 执行数据增强
    augment_dataset(
        input_images_dir=args.input_images_dir,
        input_labels_dir=args.input_labels_dir,
        output_images_dir=args.output_images_dir,
        output_labels_dir=args.output_labels_dir,
        augmentation_factor=args.factor
    )

