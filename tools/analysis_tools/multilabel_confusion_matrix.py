# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-label Confusion Matrix Visualization Tool"""
import argparse
import tempfile
from typing import List, Optional

import matplotlib.pyplot as plt
import mmengine
import numpy as np
import seaborn as sns
from mmengine.config import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner

from mmpretrain.registry import DATASETS
from mmpretrain.utils import register_all_modules


class MultiLabelConfusionMatrix:
    """Multi-label confusion matrix visualization.

    For multi-label classification, we create a confusion matrix for each class.
    Each 2x2 matrix shows:
    - True Positive (TP): Correctly predicted as positive
    - False Positive (FP): Incorrectly predicted as positive
    - False Negative (FN): Incorrectly predicted as negative
    - True Negative (TN): Correctly predicted as negative
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
        """Calculate confusion matrices for all classes.

        Args:
            y_pred: Prediction scores, shape (N, num_classes)
            y_true: Ground truth labels, shape (N, num_classes)
            threshold: Classification threshold

        Returns:
            Confusion matrices, shape (num_classes, 2, 2)
            Format: [[TN, FP], [FN, TP]]
        """
        if isinstance(y_pred, np.ndarray):
            y_pred = (y_pred >= threshold).astype(int)
        else:
            y_pred = (y_pred.numpy() >= threshold).astype(int)

        if isinstance(y_true, np.ndarray):
            y_true = y_true.astype(int)
        else:
            y_true = y_true.numpy().astype(int)

        cms = []
        for i in range(self.num_classes):
            pred_i = y_pred[:, i]
            true_i = y_true[:, i]

            tp = np.sum((pred_i == 1) & (true_i == 1))
            fp = np.sum((pred_i == 1) & (true_i == 0))
            fn = np.sum((pred_i == 0) & (true_i == 1))
            tn = np.sum((pred_i == 0) & (true_i == 0))

            # Format: [[TN, FP], [FN, TP]]
            cm = np.array([[tn, fp], [fn, tp]])
            cms.append(cm)

        return np.array(cms)

    def plot(self, cms: np.ndarray, show: bool = False, save_path: Optional[str] = None,
             figsize: tuple = (15, 10), cmap: str = 'Blues'):
        """Plot confusion matrices for all classes.

        Args:
            cms: Confusion matrices, shape (num_classes, 2, 2)
            show: Whether to display the plot
            save_path: Path to save the plot
            figsize: Figure size
            cmap: Color map

        Returns:
            matplotlib figure
        """
        n_classes = len(cms)
        n_cols = min(5, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_classes == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, cm in enumerate(cms):
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                       xticklabels=['Pred Neg', 'Pred Pos'],
                       yticklabels=['True Neg', 'True Pos'],
                       cbar=i == 0)  # Only show colorbar for first plot
            ax.set_title(f'{self.class_names[i]}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        # Hide extra subplots
        for i in range(n_classes, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Confusion matrix saved to {save_path}')

        if show:
            plt.show()

        return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix for multi-label classification')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--show', action='store_true',
                       help='Whether to display the confusion matrix')
    parser.add_argument('--show-path', type=str, default=None,
                       help='Path to save the confusion matrix image')
    parser.add_argument('--figsize', type=int, nargs=2, default=[15, 10],
                       help='Figure size (width, height)')
    parser.add_argument('--cmap', type=str, default='Blues',
                       help='Color map for heatmap')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                       help='override some settings in the config')
    return parser.parse_args()


def main():
    args = parse_args()

    # Register all modules
    register_all_modules(init_default_scope=False)

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Load checkpoint and get predictions
    cfg.load_from = args.checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.work_dir = tmpdir
        runner = Runner.from_cfg(cfg)

        # Get dataset info
        dataset = runner.test_loop.dataloader.dataset
        class_names = dataset.get_metainfo().get('classes', None)
        num_classes = len(class_names) if class_names else 20

        # Run test and get predictions
        print('Running inference...')
        predictions = runner.test_loop.run()
        print(f'Processed {len(predictions)} samples')

    # Extract predictions and ground truth
    y_pred = []
    y_true = []

    for pred in predictions:
        if 'pred_score' in pred:
            y_pred.append(pred['pred_score'].numpy())
        elif 'pred_label' in pred:
            # Convert label to score
            score = np.zeros(num_classes)
            score[pred['pred_label']] = 1.0
            y_pred.append(score)

        if 'gt_score' in pred:
            y_true.append(pred['gt_score'].numpy())
        elif 'gt_label' in pred:
            # Convert label to one-hot
            gt_onehot = np.zeros(num_classes)
            for label in pred['gt_label'].numpy():
                if label >= 0:  # Ignore difficult labels (-1)
                    gt_onehot[int(label)] = 1
            y_true.append(gt_onehot)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    print(f'Predictions shape: {y_pred.shape}')
    print(f'Ground truth shape: {y_true.shape}')

    # Calculate confusion matrices
    print('Calculating confusion matrices...')
    cm_calculator = MultiLabelConfusionMatrix(num_classes, class_names)
    cms = cm_calculator.calculate(y_pred, y_true, threshold=args.threshold)

    # Print statistics
    print('\n=== Per-class Statistics ===')
    for i, cm in enumerate(cms):
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f'{class_names[i] if class_names else f"Class {i}"}: '
              f'TP={tp}, FP={fp}, FN={fn}, TN={tn}, '
              f'Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}')

    # Plot confusion matrices
    print('Plotting confusion matrices...')
    fig = cm_calculator.plot(
        cms,
        show=args.show,
        save_path=args.show_path,
        figsize=tuple(args.figsize),
        cmap=args.cmap
    )

    print('Done!')


if __name__ == '__main__':
    main()
