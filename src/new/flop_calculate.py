import argparse
import os
import torch
from ptflops import get_model_complexity_info

from utils.model import get_model, load_model_weights

def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """
    计算PyTorch模型的FLOPs和参数量
    
    参数:
    model: PyTorch模型
    input_size: 输入张量的尺寸，默认为(1, 3, 224, 224)对应ImageNet图像
    
    返回:
    flops: 模型的FLOPs (浮点运算次数)
    params: 模型的参数量
    """
    # 设置为评估模式
    model.eval()
    
    # 计算复杂度
    flops, params = get_model_complexity_info(
        model, 
        input_size[1:],  # 去除批次维度
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    
    return flops, params

# 使用示例
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="model infer for vtk file input.")  
    # parser.add_argument('-model_name', help='Model type for inference.')
    # parser.add_argument('-model_weight_path', help='Saved model weight for inference.')
    # parser.add_argument('--stream', type=int, default=15, required=False, help='stream size of data.')
    # args = parser.parse_args()


    base = '/path/to/your/base'
    model_weight_path='/path/to/your/model_weight_path'
    model_weight_path = os.path.join(base, model_weight_path,'1/best_f1_model.pth')
    model_name  = 'SigNet_tr_single_inception'
    # num_classes = 801
    num_classes = 73
    stream = 15
    # 假设Model是你已经加载的模型
    # model = load_model_weights(model_weight_path=args.model_weight_path, model_name=args.model_name, num_classes=num_classes, stream=args.stream)
    model = load_model_weights(model_weight_path=model_weight_path, model_name=model_name, num_classes=num_classes, stream=stream)
    
    # 计算FLOPs和参数量
    flops, params = calculate_flops(model, input_size=(1, stream, 3))
    # flops, params = calculate_flops(model, input_size=(1, 3, 2*stream, 2*stream)) # DeepWMA
    
    print(f"模型参数量: {params/1e6:.2f} M params")
    print(f"模型FLOPs: {2*flops/1e6:.2f} MFLOPs")
        