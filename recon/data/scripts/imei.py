from matplotlib import dates as mdate
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List, Optional
from recon.data.scripts.utils import read_vehicle_frame
from recon.data.scripts.mappers import VehicleMapper, IMEIMapper, SpeedMapper

import asyncio
import argparse
import numpy as np
import os
import pandas as pd


def make_query_frame(live_dir: str):
    basename, join, listdir = os.path.basename, os.path.join, os.listdir

    image_dir, paths = join(live_dir, "images"), []
    dirs = map(lambda d: join(image_dir, d), listdir(image_dir))
    imeis = filter(lambda dname: basename(dname).isnumeric(), dirs)
    _index, queries = [], pd.DataFrame(columns=["file_name"])

    try:
        vehicles = read_vehicle_frame(live_dir)
    except FileNotFoundError:
        vehicles = None

    for _imei in imeis:
        files = listdir(join(_imei, "collected"))
        epochs = filter(lambda _: _.endswith(".jpeg"), files)
        for epoch in map(lambda ts: int(ts.split(".")[0]), epochs):
            imei = basename(_imei)
            if vehicles is not None:
                if (imei, epoch) in vehicles.index:
                    continue
            _index.append((imei, epoch))
            paths.append(join(_imei, "collected", f"{epoch}.jpeg"))

    index = pd.MultiIndex.from_tuples(_index, names=["imei", "epoch"])
    queries["file_name"] = paths
    queries.index = index
    return queries, vehicles


def plot_attachments(queries: pd.DataFrame):
    _plates = queries.groupby("plate").apply(lambda g: g["vehicle"].iloc[0])
    plates, idx = {vehicle: plate for plate, vehicle in
                   _plates.items() if plate != "None"}, 0

    fig, ax = plt.subplots()
    for idx, (vehicle, group) in enumerate(queries.groupby("vehicle")):
        _epoch = group.index.get_level_values(1)
        epoch = mdate.epoch2num(_epoch // 1000)
        value = np.ones_like(epoch) * (idx + 1)
        ax.plot(epoch, value, 'o', ms=2,
                label=plates[vehicle] if vehicle in plates else vehicle)
    date_fmt = '%d-%m-%y %H:%M:%S'
    date_formatter = mdate.DateFormatter(date_fmt)
    ax.xaxis.set_major_formatter(date_formatter)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 2 * idx)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    plt.show()


async def load_imei_logs(live_dir: str, imeis: Optional[List[str]]=None,
                         start: Optional[int]=None, end: Optional[int]=None,
                         silent: Optional[bool]=False):
    if imeis is None:
        imeis = []

    var_dir = os.path.join(live_dir, "var")
    _queries, vehicles = make_query_frame(live_dir)
    if len(imeis) != 0:
        _queries = _queries.loc[imeis]
    if start > 0:
        times = _queries.index.get_level_values(1)
        _queries = _queries[times >= start]
    if end > 0:
        times = _queries.index.get_level_values(1)
        _queries = _queries[times < end]

    imei_mapper = await IMEIMapper.create(var_dir)
    _queries = await imei_mapper.get_idevice(_queries)

    vehicle_mapper = await VehicleMapper.create(var_dir)
    _queries = await vehicle_mapper.get_vehicle(_queries)

    if vehicles is not None:
        queries = pd.concat((_queries, vehicles), sort=True)
    else:
        queries = _queries

    queries.sort_values("epoch", inplace=True)
    queries.to_csv(os.path.join(var_dir, "vehicles.csv"))

    speed_mapper = await SpeedMapper.create(var_dir)
    __queries, _ = await speed_mapper.get_speed(queries)
    print(__queries.head())

    if not silent:
        plot_attachments(queries)


COMMAND_DESCRIPTION = \
    """Script for loading changelog of IMEI to vehicle id imei_mapping"""
LIVE_DIR_HELP = """Location of live directory"""
IMEI_HELP = """List of IMEIs for loading changelog"""
START_HELP = """Start time for loading changelog"""
END_HELP = """End time for loading changelog"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=COMMAND_DESCRIPTION)
    parser.add_argument("live_dir", help=LIVE_DIR_HELP)
    parser.add_argument("imei", type=str, nargs="*", help=IMEI_HELP)
    parser.add_argument("--start", type=int, help=START_HELP, default=-1)
    parser.add_argument("--end", type=int, help=END_HELP, default=-1)
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(load_imei_logs(
        args.live_dir, args.imei, args.start, args.end, args.silent))
