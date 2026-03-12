import argparse
import os
import sys
from pathlib import Path
from glob import glob

# 确保可以在 scripts 目录中直接执行时导入 src 包
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.slic import slic
from src import utils, visualization


def main():
    parser = argparse.ArgumentParser(description='Batch run SLIC over a folder')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_segments', type=int, default=100)
    parser.add_argument('--compactness', type=float, default=10.0)
    parser.add_argument('--max_iter', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images = sorted(glob(os.path.join(args.input_dir, '*.*')))

    for img_path in images:
        image = utils.read_image(img_path)
        if image is None:
            continue
        labels = slic.slic(image, num_segments=args.num_segments, compactness=args.compactness, max_iter=args.max_iter)
        vis = visualization.draw_boundaries(image, labels)
        out_path = os.path.join(args.output_dir, os.path.basename(img_path))
        utils.save_image(out_path, vis)
        print('processed', img_path, '->', out_path)


if __name__ == '__main__':
    main()
