"""
GPU工具模块
统一管理CuPy/NumPy的切换，提供GPU加速支持
"""

import sys
import warnings

# 尝试导入CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 CuPy已加载，GPU加速已启用！")
    
    # 检查GPU设备
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        current_device = cp.cuda.Device()
        print(f"📱 检测到 {device_count} 个GPU设备")
        print(f"🎯 当前使用设备: GPU {current_device.id}")
        
        # 显示GPU信息
        meminfo = cp.cuda.MemoryInfo()
        total_mem = meminfo.total / (1024**3)  # GB
        print(f"💾 GPU显存: {total_mem:.1f} GB")
        
    except Exception as e:
        print(f"⚠️ GPU信息获取失败: {e}")
        
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("⚠️ CuPy未安装，使用NumPy (CPU模式)")
    print("💡 安装CuPy以启用GPU加速: pip install cupy-cuda11x")

# 导入numpy作为备选
import numpy as np

class GPUManager:
    """GPU管理器，统一处理CPU/GPU数据转换"""
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        if self.use_gpu:
            print("✅ 启用GPU模式")
        else:
            print("📱 使用CPU模式")
    
    def to_gpu(self, array):
        """将数组转移到GPU"""
        if self.use_gpu and GPU_AVAILABLE:
            if isinstance(array, np.ndarray):
                return cp.asarray(array)
            return array
        return array
    
    def to_cpu(self, array):
        """将数组转移到CPU"""
        if self.use_gpu and GPU_AVAILABLE:
            if hasattr(array, 'get'):  # CuPy数组
                return array.get()
            return array
        return array
    
    def asarray(self, array):
        """转换为当前设备的数组"""
        if self.use_gpu:
            return cp.asarray(array)
        else:
            return np.asarray(array)
    
    def zeros(self, shape, dtype=None):
        """创建零数组"""
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        """创建全1数组"""
        return self.xp.ones(shape, dtype=dtype)
    
    def random_randn(self, *shape):
        """创建随机数组"""
        return self.xp.random.randn(*shape)
    
    def random_rand(self, *shape):
        """创建0-1随机数组"""
        return self.xp.random.rand(*shape)
    
    def eye(self, n):
        """创建单位矩阵"""
        return self.xp.eye(n)
    
    def set_seed(self, seed):
        """设置随机种子"""
        if self.use_gpu:
            cp.random.seed(seed)
        np.random.seed(seed)
    
    def get_memory_info(self):
        """获取显存信息"""
        if self.use_gpu and GPU_AVAILABLE:
            try:
                meminfo = cp.cuda.MemoryInfo()
                used = meminfo.used / (1024**3)
                total = meminfo.total / (1024**3)
                return f"GPU显存: {used:.1f}/{total:.1f} GB"
            except:
                return "显存信息获取失败"
        else:
            return "CPU模式"
    
    def clear_cache(self):
        """清理GPU缓存"""
        if self.use_gpu and GPU_AVAILABLE:
            try:
                cp.cuda.MemoryPool().free_all_blocks()
                print("🧹 GPU缓存已清理")
            except:
                pass

# 创建全局GPU管理器
gpu_manager = GPUManager(use_gpu=True)

# 导出常用函数和变量，方便其他模块使用
xp = gpu_manager.xp  # 当前使用的数组库 (cupy或numpy)
to_gpu = gpu_manager.to_gpu
to_cpu = gpu_manager.to_cpu
asarray = gpu_manager.asarray
set_seed = gpu_manager.set_seed
get_memory_info = gpu_manager.get_memory_info
clear_cache = gpu_manager.clear_cache

# 数学函数的统一接口
def sqrt(x):
    return xp.sqrt(x)

def exp(x):
    return xp.exp(x)

def log(x):
    return xp.log(x)

def maximum(x, y):
    return xp.maximum(x, y)

def sum(x, axis=None, keepdims=False):
    return xp.sum(x, axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=False):
    return xp.mean(x, axis=axis, keepdims=keepdims)

def dot(a, b):
    return xp.dot(a, b)

def argmax(x, axis=None):
    return xp.argmax(x, axis=axis)

def clip(x, a_min, a_max):
    return xp.clip(x, a_min, a_max)

def concatenate(arrays, axis=0):
    return xp.concatenate(arrays, axis=axis)

def permutation(n):
    return xp.random.permutation(n)

def frombuffer(buffer, dtype=None):
    # frombuffer在CuPy中可能不直接支持，先用numpy再转换
    arr = np.frombuffer(buffer, dtype=dtype)
    return asarray(arr)

def eye(n):
    return xp.eye(n)

# 兼容性函数
def ensure_numpy(array):
    """确保数组是numpy格式（用于保存、可视化等）"""
    return to_cpu(array)

def ensure_gpu(array):
    """确保数组在GPU上（用于计算）"""
    return to_gpu(array)

# 性能监控
class GPUTimer:
    """GPU计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        if gpu_manager.use_gpu:
            cp.cuda.Stream.null.synchronize()
        self.start_time = self._get_time()
    
    def stop(self):
        if gpu_manager.use_gpu:
            cp.cuda.Stream.null.synchronize()
        self.end_time = self._get_time()
        return self.elapsed()
    
    def elapsed(self):
        if self.start_time is None or self.end_time is None:
            return 0
        return self.end_time - self.start_time
    
    def _get_time(self):
        import time
        return time.time()

def print_gpu_status():
    """打印GPU状态信息"""
    print("\n" + "="*50)
    print("🖥️  GPU状态信息")
    print("="*50)
    print(f"GPU可用: {'✅' if GPU_AVAILABLE else '❌'}")
    print(f"当前模式: {'🚀 GPU' if gpu_manager.use_gpu else '💻 CPU'}")
    print(f"数组库: {xp.__name__}")
    print(f"显存信息: {get_memory_info()}")
    print("="*50)

if __name__ == "__main__":
    # 测试GPU功能
    print_gpu_status()
    
    # 测试基本操作
    print("\n🧪 测试基本操作...")
    
    # 创建测试数组
    a = xp.random.randn(1000, 1000)
    b = xp.random.randn(1000, 1000)
    
    # 计时测试
    timer = GPUTimer()
    timer.start()
    c = dot(a, b)
    elapsed = timer.stop()
    
    print(f"矩阵乘法 (1000x1000): {elapsed:.4f}秒")
    print(f"结果形状: {c.shape}")
    print(f"结果类型: {type(c)}")
    
    # 内存测试
    print(f"\n{get_memory_info()}")
    
    print("\n✅ GPU工具模块测试完成！")
