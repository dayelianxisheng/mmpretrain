from mmpretrain.models import build_classifier
import torch
from mmengine.config import Config

def print_model_info(model_config, model_name="Model"):
    """打印详细的模型信息"""
    print(f"\n{'='*60}")
    print(f"{model_name} 详细信息")
    print(f"{'='*60}\n")

    # 构建模型
    model = build_classifier(model_config)
    model.eval()

    # 1. 打印模型结构
    print("模型结构:")
    print(model)
    print(f"\n{'='*60}\n")

    # 2. 打印参数统计
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    print("参数详细信息:")
    print("-" * 60)
    print(f"{'层名称':<40} {'输出形状':<20} {'参数数量':>10}")
    print("-" * 60)

    # 创建测试输入
    if hasattr(model, 'backbone'):
        # 根据模型类型选择输入大小
        if 'VGG' in model_config['backbone'].get('type', ''):
            input_size = (1, 3, 224, 224)
        elif 'AlexNet' in model_config['backbone'].get('type', ''):
            input_size = (1, 3, 224, 224)
        elif 'ResNet' in model_config['backbone'].get('type', ''):
            input_size = (1, 3, 224, 224)
        else:
            input_size = (1, 3, 224, 224)

        test_input = torch.randn(input_size)

        # 前向传播获取输出
        with torch.no_grad():
            try:
                output = model(test_input)
                if isinstance(output, (list, tuple)):
                    output_shape = output[0].shape if len(output) > 0 else "N/A"
                else:
                    output_shape = output.shape
                print(f"{'最终输出':<40} {str(output_shape):<20} {'N/A':>10}")
            except Exception as e:
                print(f"前向传播出错: {e}")

    print("-" * 60)

    # 统计参数
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        else:
            non_trainable_params += param_count

        # 打印每层的参数
        if param.requires_grad:
            print(f"{name:<40} {str(list(param.shape)):>20} {param_count:>10,}")

    print(f"\n{'='*60}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {non_trainable_params:,}")
    print(f"参数大小 (MB): {total_params * 4 / (1024 * 1024):.2f}")
    print(f"{'='*60}\n")

    return model

# 测试不同的模型配置
if __name__ == '__main__':

    print("正在构建 VGG16_BN 模型...")
    sevgg16_config = dict(
        type='ImageClassifier',
        backbone=dict(
            type='SEVGG',
            depth=16,
            norm_cfg=dict(type='BN'),
            num_classes=100
        ),
        neck=None,
        head=dict(
            type='ClsHead',
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        ))

    model_vgg16 = print_model_info(sevgg16_config, "SEVGG")

