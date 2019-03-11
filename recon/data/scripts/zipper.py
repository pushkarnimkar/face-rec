from argparse import ArgumentParser
from os import path, listdir, mkdir
from zipfile import ZipFile

import numpy as np


def permute_image_names(image_names: list, policy: str) -> np.ndarray:
    if policy == "random":
        return np.random.permutation(image_names)
    elif policy == "none":
        return np.ndarray(image_names)
    else:
        raise ValueError(f"unknown policy: {policy}")


def write_block_zip(block_zip_name: str, block: np.ndarray, imei_in_path: str):
    with ZipFile(block_zip_name, mode="w") as archive:
        for image_name in block:
            imei = path.basename(path.dirname(imei_in_path))
            archive_path = path.join(imei, image_name)
            image_path = path.join(imei_in_path, image_name)
            with archive.open(archive_path, mode="w") as archive_file, \
                    open(image_path, "rb") as image:
                archive_file.write(image.read())


def main(data_dir: str, include: list, exclude: list,
         policy: str, block_size: int):

    assert len(include) == 0 or len(exclude) == 0
    assert path.exists(data_dir)

    image_path = path.join(data_dir, "images")
    zipped_path = path.join(data_dir, "zipped")
    if not path.exists(zipped_path):
        mkdir(zipped_path)

    imeis = set(listdir(image_path))
    if len(include) != 0:
        selected = imeis.intersection(set(include))
    elif len(exclude) != 0:
        selected = imeis.difference(set(exclude))
    else:
        selected = set(imeis)

    for imei in selected:
        imei_in_path = path.join(image_path, imei, "collected")
        image_names = listdir(imei_in_path)
        permutation = permute_image_names(image_names, policy)

        split_indices = np.arange(block_size, permutation.shape[0], block_size)
        blocks = np.split(permutation, split_indices)

        for index, block in enumerate(blocks):
            block_name = f"{imei}-{index:03d}"
            print(f"zipping {block_name} {block.shape[0]: 4d} files ...")
            block_zip_name = path.join(zipped_path, f"{block_name}.zip")
            write_block_zip(block_zip_name, block, imei_in_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_dir", help="contains root of data directory")
    parser.add_argument("--include", nargs="+", default=[],
                        help="list of imeis to be included")
    parser.add_argument("--exclude", nargs="+", default=[],
                        help="list of imeis to be excluded")
    parser.add_argument("--policy", default="random",
                        help="policy for selection of images in block")
    # better go like 10 30 100 300 1000 3000 and so on
    parser.add_argument("--block-size", default=300, type=int,
                        help="number of images per block")
    args = parser.parse_args()
    main(args.data_dir, args.include, args.exclude, args.policy, args.block_size)
