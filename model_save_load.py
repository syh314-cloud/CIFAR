#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型保存和加载功能
解决eval.py需要重新训练的问题
"""

import numpy as np
import pickle
import os

def save_model(model, filepath):
    """
    保存模型权重和结构
    """
    model_data = {
        'layers': [],
        'dropout_params': {
            'dropout1_p': model.dropout1.p,
            'dropout2_p': model.dropout2.p, 
            'dropout3_p': model.dropout3.p,
        }
    }
    
    # 保存每层的权重和偏置
    for i, layer in enumerate(model.layers):
        layer_data = {
            'w': layer.w.copy(),
            'b': layer.b.copy(),
        }
        model_data['layers'].append(layer_data)
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存到文件
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ 模型已保存到: {filepath}")

def load_model(model, filepath):
    """
    加载模型权重
    """
    if not os.path.exists(filepath):
        print(f"❌ 模型文件不存在: {filepath}")
        return False
    
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # 恢复权重
        for i, layer_data in enumerate(model_data['layers']):
            if i < len(model.layers):
                model.layers[i].w = layer_data['w'].copy()
                model.layers[i].b = layer_data['b'].copy()
        
        # 恢复dropout参数
        if 'dropout_params' in model_data:
            model.dropout1.p = model_data['dropout_params']['dropout1_p']
            model.dropout2.p = model_data['dropout_params']['dropout2_p']
            model.dropout3.p = model_data['dropout_params']['dropout3_p']
        
        print(f"✅ 模型已从 {filepath} 加载")
        return True
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return False

def save_training_history(history, filepath):
    """
    保存训练历史
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"✅ 训练历史已保存到: {filepath}")

def load_training_history(filepath):
    """
    加载训练历史
    """
    if not os.path.exists(filepath):
        print(f"❌ 训练历史文件不存在: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            history = pickle.load(f)
        print(f"✅ 训练历史已从 {filepath} 加载")
        return history
    except Exception as e:
        print(f"❌ 加载训练历史失败: {e}")
        return None
