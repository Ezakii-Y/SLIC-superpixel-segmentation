import argparse
from src import slic, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    image = utils.read_image(args.input)
    labels = slic.slic(image)
    # TODO: 保存结果到 output
