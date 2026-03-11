import argparse
import os
from glob import glob
import cv2
import numpy as np
from src import evaluation, utils


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation labels against ground truth')
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    args = parser.parse_args()

    pred_files = sorted(glob(os.path.join(args.pred_dir, '*.*')))
    metrics = []

    for pred in pred_files:
        name = os.path.basename(pred)
        gt_path = os.path.join(args.gt_dir, name)
        if not os.path.exists(gt_path):
            continue

        img_pred = utils.read_image(pred)
        img_gt = utils.read_image(gt_path)
        if img_pred is None or img_gt is None:
            continue

        # 处理为灰度标签
        labels = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
        gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)

        res = evaluation.evaluate(labels, gt)
        metrics.append((name, res))
        print(name, res)

    if metrics:
        avg = {
            'boundary_recall': np.mean([m['boundary_recall'] for _, m in metrics]),
            'undersegmentation_error': np.mean([m['undersegmentation_error'] for _, m in metrics]),
            'achievable_segmentation_accuracy': np.mean([m['achievable_segmentation_accuracy'] for _, m in metrics]),
        }
        print('avg', avg)


if __name__ == '__main__':
    main()
