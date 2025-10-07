try:
    import cupy as np
    import numpy as np_cpu
    print("🚀 GPU加速已启用")
except ImportError:
    import numpy as np
    np_cpu = np
    print("⚠️ 使用CPU模式")
SEED = 2023
np.random.seed(SEED)

train_images_list = []
train_labels_list = []
test_images_list = []
test_labels_list = []
for i in range(5):
    with open(f'data/cifar-10-batches-bin/data_batch_{i+1}.bin','rb') as f: #读取二进制文件
        raw_data = f.read()
    arr_cpu = np_cpu.frombuffer(raw_data,dtype=np_cpu.uint8).reshape(-1,3073)
    arr = np.asarray(arr_cpu)
    labels = arr[:,0]  #分离图像和标签
    pixels = arr[:,1:]
    pixels = pixels.astype(np.float32)/255.0  #归一化
    train_images_list.append(pixels)
    train_labels_list.append(labels)
train_images = np.concatenate(train_images_list,axis=0)  #合并训练集
train_labels = np.eye(10)[np.concatenate(train_labels_list,axis=0)]
indices = np.random.permutation(train_images.shape[0])  #打乱训练集
train_images = train_images[indices]
train_labels = train_labels[indices]

#划分验证集
val_images = train_images[45000:]
val_labels = train_labels[45000:]
train_images = train_images[:45000]
train_labels = train_labels[:45000]

with open('data/cifar-10-batches-bin/test_batch.bin','rb') as f:
    raw_data = f.read()
    arr_cpu = np_cpu.frombuffer(raw_data,dtype=np_cpu.uint8).reshape(-1,3073)
    arr = np.asarray(arr_cpu)
    labels = arr[:,0]
    pixels = arr[:,1:]
    pixels = pixels.astype(np.float32)/255.0
    test_images_list.append(pixels)
    test_labels_list.append(labels)
test_images = np.concatenate(test_images_list,axis=0)
test_labels = np.eye(10)[np.concatenate(test_labels_list,axis=0)]


