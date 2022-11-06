import os
import json
import random
import pandas as pd
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机划分结果一致
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('R5/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".jpeg", ".JPEG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.\n{} for training, {} for validation".format(sum(every_class_num),
                                                                                            len(train_images_path),
                                                                                            len(val_images_path)
                                                                                            ))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def generate_ds(data_root: str,
                train_im_height: int = 224,
                train_im_width: int = 224,
                val_im_height: int = None,
                val_im_width: int = None,
                batch_size: int = 8,
                val_rate: float = 0.1,
                cache_data: bool = False):
    """
    读取划分数据集，并生成训练集和验证集的迭代器
    :param data_root: 数据根目录
    :param train_im_height: 训练输入网络图像的高度
    :param train_im_width:  训练输入网络图像的宽度
    :param val_im_height: 验证输入网络图像的高度
    :param val_im_width:  验证输入网络图像的宽度
    :param batch_size: 训练使用的batch size
    :param val_rate:  将数据按给定比例划分到验证集
    :param cache_data: 是否缓存数据
    :return:
    """
    assert train_im_height is not None
    assert train_im_width is not None
    if val_im_width is None:
        val_im_width = train_im_width
    if val_im_height is None:
        val_im_height = train_im_height

    # Read Training Data
    train_data = pd.read_csv(data_root + 'Training_data.CSV', header=None)
    train_data = np.array(train_data).astype('float32')
    train_data = tf.reshape(tensor=train_data, shape=[-1, 640, 18, 1])

    # Read Training Labels
    train_labels = pd.read_csv(data_root + 'Training_labels2.CSV', header=None)
    train_labels = np.array(train_labels).astype('float32')
    train_labels2 = []
    for i in range(len(train_labels)):
        if(i%18 == 0):
            train_labels2.append(train_labels[i])
    train_labels2 = tf.reshape(tensor=train_labels2, shape=[-1])


    # Read Testing Data
    test_data = pd.read_csv(data_root + 'Test_data.CSV', header=None)
    test_data = np.array(test_data).astype('float32')
    test_data = tf.reshape(tensor=test_data, shape=[-1, 640, 18, 1])


    # Read Testing Labels
    test_labels = pd.read_csv(data_root + 'Test_labels2.CSV', header=None)
    test_labels = np.array(test_labels).astype('float32')
    test_labels2 = []
    for i in range(len(test_labels)):
        if (i % 18 == 0):
            test_labels2.append(test_labels[i])
    test_labels2 = tf.reshape(tensor=test_labels2, shape=[-1])

    # train_img_path, train_img_label, val_img_path, val_img_label = read_split_data(data_root, val_rate=val_rate)
    AUTOTUNE = tf.data.experimental.AUTOTUNE


    # Configure dataset for performance
    def configure_for_performance(ds,
                                  shuffle_size: int,
                                  shuffle: bool = False,
                                  cache: bool = False):
        if cache:
            ds = ds.cache()  # 读取数据后缓存至内存
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_size)  # 打乱数据顺序
        ds = ds.batch(batch_size)                      # 指定batch size
        ds = ds.prefetch(buffer_size=AUTOTUNE)         # 在训练的同时提前准备下一个step的数据
        return ds

    # 从路径读取图片
    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_data),
                                                   tf.constant(train_labels2)))
    total_train = len(train_data[0])

    # Use Dataset.map to create a dataset of image, label pairs
    # train_ds = train_ds.map(process_train_info, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds, total_train, shuffle=True, cache=cache_data)

    val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(test_data),
                                                 tf.constant(test_labels2)))
    total_val = len(test_data[0])
    # Use Dataset.map to create a dataset of image, label pairs
    # val_ds = val_ds.map(process_val_info, num_parallel_calls=AUTOTUNE)
    val_ds = configure_for_performance(val_ds, total_val, cache=False)



    return train_ds, val_ds