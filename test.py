import tensorflow_hub as hub #谷歌开源的预训练模型库，可直接加载如风格迁移、图像分类等现成模型,a hub给库指定别名
import tensorflow as tf #负责处理复杂计算，比如让模型跑起来、处理图像数据。
import matplotlib.pyplot as plt #把原始图片、处理后的图片显示出来，方便我们看效果，可视化
import numpy as np #图片数据变成电脑能看懂的数字格式
from PIL import Image #图片处理，打开、保存图片，还能简单修改图片（比如调整大小）
import os #找到电脑里图片的位置
import ssl#网上下载模型，它能保证下载过程安全
#os 找图片路径 → PIL 打开并处理图片 → numpy 转换图片为数字格式 → tensorflow 作为 “发动机”，配合 tensorflow_hub 加载的模型进行风格迁移计算 → 
# 计算结果再通过 numpy 转换 → matplotlib 显示或保存图片，而 ssl 确保模型能顺利下载。

# 输入原始图片的路径 - 使用原始字符串或正斜杠 使用原始字符串
content_image_path = r"c:\Users\24514\Desktop\7bc3df755242894bdcf58ff1fd444238.jpg"  # 使用原始字符串
style_image_path = r"c:\Users\24514\Desktop\49c1c5cfd4a82c322a19658621782f48.jpg"  # 使用原始字符串
# 或者使用正斜杠（推荐）
# content_image_path = "test_picture/girl.jpg"
# style_image_path = "test_picture/MonaLisa.jpg"


plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等，解决中文显示乱码，指定支持中文的字体。
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

# 检查文件是否存在
if not os.path.exists(content_image_path):
    print(f"错误: 内容图像文件不存在: {content_image_path}")
    print("请确保文件存在并修改 content_image_path 变量")
    exit(1)

if not os.path.exists(style_image_path):
    print(f"错误: 风格图像文件不存在: {style_image_path}")
    print("请确保文件存在并修改 style_image_path 变量")
    exit(1)

print(f"内容图像路径: {content_image_path}")
print(f"风格图像路径: {style_image_path}")

# 定义load_image函数用于加载内容和风格图像
def load_image(image_path, max_dim=512):#输入图像文件的路径（字符串），输入图像调整后的最大边长（默认 512 像素），用于限制图像尺寸
    """加载并预处理图像"""
    try:
        img = tf.io.read_file(image_path)#使用 TensorFlow 的 tf.io.read_file 读取图像文件的原始字节数据，输入的文件路径 输出的二进制字符串未解码
        img = tf.image.decode_image(img, channels=3)#TensorFlow 中用于解码图像文件数据，输出解码后的 tf.Tensor，
                                                     # 表示图像的像素值，形状为 (height, width, channels)。3 通道（RGB）：用“红+绿+蓝”三种颜色的混合表示颜色
        img = tf.image.convert_image_dtype(img, tf.float32)#将图像像素值从 [0, 255] 的整数转换为 [0, 1] 的浮点数（tf.float32 类型）
        
        # 调整图像大小，保持宽高比
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img
    except Exception as e:
        print(f"加载图像时出错: {e}")
        return None

# 加载图像
print("加载图像...")
content_image = load_image(content_image_path)
style_image = load_image(style_image_path)

if content_image is None or style_image is None:
    print("图像加载失败，程序退出")
    exit(1)

# 调整风格图像大小为推荐尺寸 (256x256)
style_image = tf.image.resize(style_image, (256, 256))

print(f"内容图像形状: {content_image.shape}")
print(f"风格图像形状: {style_image.shape}")

# 加载图像风格迁移模块
print("加载风格迁移模型...")

def load_model_with_retry():#定义了一个名为 load_model_with_retry 的函数，用于尝试多种方式加载模型，并在失败时进行重试
    """尝试多种方式加载模型"""
    model_urls = [#定义了一个包含两个模型下载 URL（统一资源定位符） 的列表：
        'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2',# TensorFlow Hub 的官方模型地址
        'https://storage.googleapis.com/tfhub-modules/google/magenta/arbitrary-image-stylization-v1-256/2.tar.gz',# Google Cloud Storage 的直接下载链接,
    ]
    
    for i, url in enumerate(model_urls):#for 循环遍历 model_urls 列表
        try:
            print(f"尝试方法 {i+1}: {url}")
            hub_module = hub.load(url)
            print("模型加载成功!")
            return hub_module
        except Exception as e:
            print(f"方法 {i+1} 失败: {e}")
    
    # 如果在线方法都失败，尝试使用本地模型（如果存在）  
    local_model_path = "./new_model"
    if os.path.exists(local_model_path):
        try:
            print("尝试加载本地模型...")
            hub_module = hub.load(local_model_path)
            print("本地模型加载成功!")
            return hub_module
        except Exception as e:
            print(f"本地模型加载失败: {e}")
    
    return None


# 设置SSL上下文以避免SSL错误（临时解决方案）
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

hub_module = load_model_with_retry()

if hub_module is None:
    print("所有模型加载方法都失败！")
    print("请尝试以下解决方案：")
    print("1. 检查网络连接")
    print("2. 手动下载模型并保存到 ./model/ 目录")
    print("3. 使用VPN或更换网络环境")
    exit(1)
# 风格迁移
print("进行风格迁移...")
try:
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    print("风格迁移完成!")
except Exception as e:
    print(f"风格迁移失败: {e}")
    exit(1)

# 显示结果
def show_images(content, style, stylized):
    """显示内容图像、风格图像和风格迁移结果"""
    plt.figure(figsize=(15, 5))
    
    # 内容图像
    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(content))
    plt.title('内容图像')
    plt.axis('off')
    
    # 风格图像
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(style))
    plt.title('风格图像')
    plt.axis('off')
    
    # 风格迁移结果
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(stylized))
    plt.title('风格迁移结果')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 显示图像
print("显示结果...")
show_images(content_image, style_image, stylized_image)

# 保存结果图像
def save_stylized_image(stylized_image, filename='stylized_result.jpg'):
    """保存风格迁移结果"""
    try:
        # 将张量转换为numpy数组
        stylized_np = stylized_image.numpy()
        
        # 调整值范围
        if stylized_np.max() <= 1.0:
            stylized_np = (stylized_np * 255).astype(np.uint8)
        else:
            stylized_np = stylized_np.astype(np.uint8)
        
        # 移除批次维度
        if len(stylized_np.shape) == 4:
            stylized_np = stylized_np[0]
        
        # 保存图像
        image = Image.fromarray(stylized_np)
        image.save(filename)
        print(f"结果已保存为: {filename}")
        return True
    except Exception as e:
        print(f"保存图像时出错: {e}")
        return False

# 保存结果
save_stylized_image(stylized_image)

print("程序执行完成!")
# 测试