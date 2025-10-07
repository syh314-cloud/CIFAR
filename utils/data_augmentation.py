try:
    import cupy as np
    import numpy as np_cpu
except ImportError:
    import numpy as np
    np_cpu = np
from scipy.ndimage import rotate

SEED = 2023
def random_flip(images, prob=0.5):
    mask = np.random.rand(images.shape[0]) < prob
    images[mask] = images[mask, :, :, ::-1]  
    return images

def random_crop(images, crop_size=32, padding=4):
    padded = np.pad(images, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='reflect')
    _, _, H, W = padded.shape
    x_starts = np.random.randint(0, W - crop_size, size=images.shape[0])
    y_starts = np.random.randint(0, H - crop_size, size=images.shape[0])
    
    cropped = np.zeros_like(images)
    for i in range(images.shape[0]):
        cropped[i] = padded[i, :, y_starts[i]:y_starts[i]+crop_size, x_starts[i]:x_starts[i]+crop_size]
    return cropped

def random_rotate(images, max_angle=15):
    rotated = np.zeros_like(images)
    for i in range(images.shape[0]):
        angle = float(np.random.uniform(-max_angle, max_angle))
        for c in range(3):  
            img_cpu = np_cpu.asarray(images[i, c]) if hasattr(images, 'get') else images[i, c]
            rotated_cpu = rotate(img_cpu, angle, reshape=False, mode='reflect')
            rotated[i, c] = np.asarray(rotated_cpu)
    return rotated

def random_noise(images, std=0.02):
    noise = np.random.normal(0, std, images.shape)
    noisy_images = np.clip(images + noise, 0.0, 1.0)
    return noisy_images

def augment_images(images, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    images = random_flip(images)
    images = random_crop(images)
    #images = random_rotate(images)
    #images = random_noise(images)
    return images