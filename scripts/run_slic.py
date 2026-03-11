import argparse
import os
import numpy as np
from src import slic, utils, visualization


def main():
    parser = argparse.ArgumentParser(description='Run SLIC superpixel segmentation on one image')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--num_segments', type=int, default=100)
    parser.add_argument('--compactness', type=float, default=10.0)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--save_labels', action='store_true', help='Save labels as numpy file')
    args = parser.parse_args()

    image = utils.read_image(args.input)
    if image is None:
        raise FileNotFoundError('Input image not found: %s' % args.input)

    labels = slic.slic(image, num_segments=args.num_segments, compactness=args.compactness, max_iter=args.max_iter)
    vis = visualization.draw_boundaries(image, labels)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    utils.save_image(args.output, vis)

    if args.save_labels:
        np.save(args.output + '.npy', labels)

    print('Saved', args.output)


if __name__ == '__main__':
    main()
