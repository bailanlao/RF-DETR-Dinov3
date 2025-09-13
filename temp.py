import numpy as np
from PIL import Image
import os
import sys
import glob

def process_npy_file(npy_file_path, output_parent_dir):
    """处理单个npy文件，提取图片并直接保存到输出文件夹（无分子文件夹）"""
    try:
        # 获取npy文件的“基础名”（不含扩展名），用于图片命名
        base_name = os.path.splitext(os.path.basename(npy_file_path))[0]
        output_dir = output_parent_dir  # 直接使用输出父目录，不创建子文件夹
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
        
        print(f"\n===== 开始处理文件: {npy_file_path} =====")
        
        # 加载npy文件
        data = np.load(npy_file_path, allow_pickle=True)
        
        # 打印基本信息
        print(f"npy文件加载成功")
        print(f"数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"数据类型（Python类型）: {type(data)}")
        
        # 处理空形状的object类型数据
        if data.dtype == 'object' and data.shape == ():
            print("检测到空形状的object类型数据，解析内部结构...")
            data_content = data.item()  # 获取object中的实际内容
            print(f"内部数据类型: {type(data_content)}")
            
            # 处理字典类型
            if isinstance(data_content, dict):
                print(f"检测到字典结构，包含键: {list(data_content.keys())}")
                
                # 打印structures内容（如果存在）
                if 'structures' in data_content:
                    print("\n===== 开始打印 structures 内容 =====")
                    structures_data = data_content['structures']
                    print(f"structures 数据类型: {type(structures_data)}")
                    
                    # 根据数据类型打印不同信息
                    if isinstance(structures_data, np.ndarray):
                        print(f"structures 形状: {structures_data.shape}")
                        print(f"structures 数据类型: {structures_data.dtype}")
                        print("structures 内容预览（前5个元素）:")
                        print(structures_data[:5] if len(structures_data.shape) > 0 else structures_data)
                    elif isinstance(structures_data, (list, tuple)):
                        print(f"structures 长度: {len(structures_data)}")
                        print("structures 内容预览（前5个元素）:")
                        print(structures_data[:5])
                    elif isinstance(structures_data, dict):
                        print(f"structures 包含键: {list(structures_data.keys())}")
                        print("structures 部分内容预览:")
                        # 只打印前3个键值对，避免内容过多
                        for i, (k, v) in enumerate(structures_data.items()):
                            if i < 3:
                                print(f"  {k}: {str(v)[:100]}...")  # 每个值只显示前100字符
                    else:
                        print("structures 内容预览:")
                        print(str(structures_data)[:500] + "...")  # 限制显示长度
                    print("===== 结束打印 structures 内容 =====\n")
                
                # 尝试从字典中提取图片数据
                image_keys = ['image', 'images', 'data', 'img', 'imgs']
                found_key = None
                
                for key in image_keys:
                    if key in data_content:
                        found_key = key
                        break
                
                if found_key:
                    print(f"从键 '{found_key}' 中提取图片数据")
                    data = data_content[found_key]
                    print(f"提取的数据类型: {type(data)}")
                else:
                    print(f"未找到图片相关键名，请检查字典结构")
                    return
            
            # 直接使用内容作为数据
            else:
                data = data_content
        
        print(f"图片将保存到目录: {os.path.abspath(output_dir)}")
        
        # 处理图片数据（传递base_name用于命名）
        process_image_collection(data, output_dir, show_images=False, base_name=base_name)
        print(f"===== 完成处理文件: {npy_file_path} =====")
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {npy_file_path}")
    except Exception as e:
        print(f"处理文件 {npy_file_path} 时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def process_image_collection(collection, output_dir, show_images, base_name):
    """处理图片集合，批量保存图片（使用base_name+序号命名）"""
    # 确保集合是可迭代的
    if not isinstance(collection, (list, tuple, np.ndarray)):
        print(f"图片集合不是可识别的列表或数组类型: {type(collection)}")
        return
    
    # 转换为数组以便统一处理
    if isinstance(collection, (list, tuple)):
        collection = np.array(collection)
    
    # 确定图片数量
    if len(collection.shape) == 4:  # (num, h, w, c)
        num_images = collection.shape[0]
    elif len(collection.shape) == 3:  # 单张彩色图 (h, w, c)
        num_images = 1
    elif len(collection.shape) == 2:  # 单张灰度图 (h, w)
        num_images = 1
    else:
        print(f"不支持的图片集合形状: {collection.shape}")
        return
    
    print(f"检测到 {num_images} 张图片，开始导出...")
    
    # 批量处理每张图片（传递base_name）
    for i in range(num_images):
        # 获取单张图片数据
        if num_images > 1:
            image_data = collection[i]
        else:
            image_data = collection
        
        # 处理并保存图片
        process_image(image_data, i, output_dir, show_images, base_name)
    
    print(f"图片导出完成，共处理 {num_images} 张图片")

def process_image(image_data, index, output_dir, show_images, base_name):
    """处理单张图片并保存（文件名：base_name_image_序号.png）"""
    try:
        # 确保数据在0-255范围内，并转换为uint8类型
        if image_data.dtype != np.uint8:
            # 处理可能的归一化数据（0-1范围）
            if np.max(image_data) <= 1.0 and np.min(image_data) >= 0:
                image_data = (image_data * 255).astype(np.uint8)
            else:
                # 归一化到0-255，添加微小值防止除零
                min_val = image_data.min()
                max_val = image_data.max()
                if max_val > min_val:
                    image_data = ((image_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    image_data = np.zeros_like(image_data, dtype=np.uint8)
        
        # 转换为PIL Image对象
        if len(image_data.shape) == 2:  # 灰度图
            img = Image.fromarray(image_data, mode='L')
        else:  # 彩色图
            # 处理不同通道数
            if image_data.shape[-1] == 3:
                img = Image.fromarray(image_data)
            elif image_data.shape[-1] == 1:  # 单通道图转换为灰度图
                img = Image.fromarray(image_data.squeeze(), mode='L')
            else:
                print(f"警告: 不支持的通道数 {image_data.shape[-1]}，尝试转换为灰度图")
                img = Image.fromarray(image_data.mean(axis=-1).astype(np.uint8), mode='L')
        
        # 显示图片（可选）
        if show_images:
            img.show(title=f"Image {index}")
        
        # 保存图片：使用“基础名_image_序号.png”命名
        img_path = os.path.join(output_dir, f"{base_name}_image_{index:04d}.png")
        img.save(img_path)
        # 每10张图片打印一次进度
        if index % 10 == 0:
            print(f"已保存 {index + 1} 张图片...")
            
    except Exception as e:
        print(f"处理图片 {index} 时出错: {str(e)}")

def batch_process_npy_files(input_dir, output_dir):
    """批量处理文件夹中的所有npy文件"""
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在 - {input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有npy文件
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    
    if not npy_files:
        print(f"在目录 {input_dir} 中未找到任何npy文件")
        return
    
    print(f"找到 {len(npy_files)} 个npy文件，开始批量处理...")
    
    # 逐个处理npy文件
    for i, npy_file in enumerate(npy_files):
        print(f"\n----- 处理文件 {i+1}/{len(npy_files)} -----")
        process_npy_file(npy_file, output_dir)
    
    print(f"\n批量处理完成！所有图片已保存到 {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # 支持命令行参数指定输入和输出文件夹
    # 使用方式: python 脚本名.py 输入文件夹路径 输出文件夹路径
    if len(sys.argv) == 3:
        input_directory = sys.argv[1]
        output_directory = sys.argv[2]
    else:
        # 默认路径（可根据需要修改）
        input_directory = "ARRAY_FORMAT"    # 存放npy文件的文件夹
        output_directory = "extracted_images"  # 图片输出文件夹
    
    # 执行批量处理
    batch_process_npy_files(input_directory, output_directory)