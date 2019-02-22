from argparse import ArgumentParser
from collections import defaultdict
from functools import partial

import numpy as np
import os
import shutil


def list_images(base: str):
    join, isdir, isfile, listdir = (os.path.join, os.path.isdir, 
                                    os.path.isfile, os.listdir)
    dirs = filter(isdir, listdir(base))
    files = defaultdict(list)
    for dirname in map(lambda name: join(base, name), dirs):
        files[dirname].extend(filter(isfile, map(
            lambda name: join(dirname, name), listdir(dirname))))
    return files


def criteria(filename, start_time):
    return int(os.path.basename(filename).split(".")[0]) >= start_time


def random_select(file_list, count, start_time=0):
    picked = list(filter(partial(criteria, start_time=start_time), file_list))
    permutation = np.random.permutation(len(picked))[:count]
    return np.array(picked)[permutation].tolist()


def make_sample(selected, base):
    join, isdir, basename = os.path.join, os.path.isdir, os.path.basename

    sample_dir = join(base, "sample")
    for dirname, file_list in selected.items():
        for filename in file_list:
            shutil.copyfile(filename, join(sample_dir, basename(filename)))


def main(args):
    im_len, st_len = len(args.imei), len(args.start_time)
    assert len(args.imei) == len(args.start_time), \
        f"length mismatch {st_len}, {im_len}"

    sample_dir = os.path.join(args.base, "sample")
    if os.path.isdir(sample_dir):
        shutil.rmtree(sample_dir)
    os.mkdir(sample_dir)

    files = list_images(args.base)
    difference = set(args.imei).difference(
        set(map(os.path.basename, files.keys())))

    assert len(difference) == 0, f"invalid imei inputs {difference}"
    _start_times = {imei: start_time for imei, start_time in 
                    zip(args.imei, args.start_time)}
    start_times = defaultdict(lambda : 0, _start_times)

    sizes = {key: len(val) for key, val in files.items()}
    factor = args.n / sum(sizes.values())
    pick_size = {key: round(val * factor) for key, val in sizes.items()}
    selected = {dirname: random_select(file_list, pick_size[dirname],
                start_time=start_times[os.path.basename(dirname)]) 
                for dirname, file_list in files.items()}
    make_sample(selected, args.base)


if __name__ == "__main__":
    parser = ArgumentParser(description="""make sample image set from synchronized s3 bucket of driver-image-store""")
    parser.add_argument("base", help="""directory where data from the bucket was loaded""")
    parser.add_argument("n", type=int, help="""number of samples required""")
    parser.add_argument("-im", "--imei", nargs="*", default=[])
    parser.add_argument("-st", "--start-time", type=int, nargs="*", default=[])
    main(parser.parse_args())

