"""使用 MobileNetV2 VOC 多标签分类模型进行推理。

用法:
    python my/tools/infer_mobilenetv2_voc.py <image_path_or_url> [--checkpoint <path>] [--device cuda|cpu]

示例:
    python my/tools/infer_mobilenetv2_voc.py demo/demo.JPEG
    python my/tools/infer_mobilenetv2_voc.py demo/demo.JPEG --device cuda
    python my/tools/infer_mobilenetv2_voc.py data/VOC2012/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg
"""

import argparse
import os
import sys

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from PIL import Image

from mmpretrain.models import build_classifier

# Pascal VOC 2012 的 20 个类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

# 与训练配置一致的预处理参数
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_SIZE = 224


def preprocess_image(image_path: str, device: str) -> torch.Tensor:
    """加载并预处理图像（与训练时的 test_pipeline 保持一致：ResizeEdge(256) + CenterCrop(224)）。"""
    if image_path.startswith(('http://', 'https://')):
        from urllib.request import urlopen
        img = Image.open(urlopen(image_path)).convert('RGB')
    else:
        img = Image.open(image_path).convert('RGB')

    # ResizeEdge: 短边缩放到 256
    w, h = img.size
    scale = 256.0 / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # CenterCrop: 中心裁剪 224x224
    left = (new_w - INPUT_SIZE) // 2
    top = (new_h - INPUT_SIZE) // 2
    img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))

    # 转为 numpy 数组并归一化
    img_array = np.array(img, dtype=np.float32)  # (H, W, C), RGB
    img_array = (img_array - IMG_MEAN) / IMG_STD

    # 转为 (C, H, W) 并添加 batch 维度
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def load_model(config_path: str, checkpoint_path: str, device: str):
    """构建模型并加载权重。"""
    cfg = Config.fromfile(config_path)
    model = build_classifier(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def inference(model, image_tensor: torch.Tensor, threshold: float = 0.5):
    """推理并返回多标签预测结果。"""
    with torch.no_grad():
        outputs = model(image_tensor)
        # MultiLabelLinearClsHead 输出 logits
        if isinstance(outputs, (list, tuple)):
            scores = outputs[0]
        elif isinstance(outputs, torch.Tensor):
            scores = outputs
        else:
            scores = outputs

        probs = torch.sigmoid(scores).cpu().numpy()[0]

    # 获取概率大于阈值的类别
    pred_indices = np.where(probs >= threshold)[0]
    pred_indices = sorted(pred_indices, key=lambda i: probs[i], reverse=True)

    results = []
    for idx in pred_indices:
        results.append({
            'class': VOC_CLASSES[idx],
            'score': float(probs[idx]),
        })

    return results, probs


def main():
    parser = argparse.ArgumentParser(description='MobileNetV2 VOC 多标签分类推理')
    parser.add_argument('image', nargs='+', help='图像路径或 URL（支持多个）')
    parser.add_argument(
        '--config', default='my/models/mobilenet/mobilenetv2_voc.py',
        help='模型配置文件路径')
    parser.add_argument(
        '--checkpoint', default='my/work_dirs/mobilenetv2_voc/best_multi-label_mAP_epoch_48.pth',
        help='模型权重文件路径')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='推理设备 (cuda/cpu)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='多标签分类的置信度阈值')
    parser.add_argument('--topk', type=int, default=5,
                        help='打印每个类别的概率时显示前 K 个')
    args = parser.parse_args()

    print(f'配置文件: {args.config}')
    print(f'权重文件: {args.checkpoint}')
    print(f'设备: {args.device}')

    # 检查文件存在
    for path in args.image:
        if not path.startswith(('http://', 'https://')) and not os.path.exists(path):
            print(f'[警告] 文件不存在: {path}')

    model = load_model(args.config, args.checkpoint, args.device)

    for image_path in args.image:
        print(f"\n{'=' * 60}")
        print(f'图像: {image_path}')
        print(f'{"=" * 60}')

        try:
            tensor = preprocess_image(image_path, args.device)
            results, all_probs = inference(model, tensor, args.threshold)

            # 输出预测结果
            if results:
                print(f'\n预测类别 (sigmoid >= {args.threshold}):')
                for r in results:
                    print(f'  {r["class"]:<15}  {r["score"]:.4f}')
            else:
                print(f'\n无类别超过阈值 {args.threshold}')

            # 输出 top-k 概率
            topk_idx = np.argsort(all_probs)[::-1][:args.topk]
            print(f'\nTop-{args.topk} 类别概率:')
            for i in topk_idx:
                marker = ' *' if all_probs[i] >= args.threshold else ''
                print(f'  {VOC_CLASSES[i]:<15}  {all_probs[i]:.4f}{marker}')

        except Exception as e:
            print(f'推理失败: {e}')


if __name__ == '__main__':
    main()
