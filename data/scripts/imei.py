from matplotlib import dates as mdate
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Tuple, Optional
from data.scripts.utils import extract_from_two, read_vehicle_frame

import aiofiles
import aiohttp
import asyncio
import argparse
import numpy as np
import os
import pandas as pd
import time


class IMEIMapper:
    IDEVICE_INFO_TEMPL = \
        "http://internal-apis.intangles.com/idevice/{0}/infoV2"
    CSV_ROW_TEMPL = "{imei},{idevice},{created}\n"

    def __init__(self, live_dir: str):
        if not os.path.exists(live_dir):
            raise FileNotFoundError(live_dir)
        self.live_dir = live_dir
        self.imei_mapping = {}

    async def __fetch_idevice_info(self, imeis: List[str]):
        async with aiohttp.ClientSession() as session:
            for imei in imeis:
                idevice_info_url = \
                    self.IDEVICE_INFO_TEMPL.format(imei)
                async with session.get(idevice_info_url) as response:
                    _response = await response.json()
                    yield imei, _response

    async def _fetch_idevice_imei_map(self, imeis: List[str]):
        mapping = {}
        async with aiofiles.open(self.mapping_file, "a") as out:
            async_iter = self.__fetch_idevice_info(imeis)
            async for imei, info_object in async_iter:
                tracker = info_object["result"]["tracker"]
                if "id" not in tracker:
                    mapping[imei] = None
                imei, idevice, created = \
                    tracker["imei"], tracker["id"], \
                    tracker["__lastcreatedtime"]
                csv_row = self.CSV_ROW_TEMPL.format(
                    imei=imei, idevice=idevice, created=created)
                mapping[imei] = (idevice, created)
                await out.write(csv_row)
        self.imei_mapping.update(mapping)
        return mapping

    async def _load_idevice_imei_map(self):
        try:
            async with aiofiles.open(self.mapping_file, "r") as inp:
                async for line in inp:
                    imei, idevice, created = line[:-1].split(",")
                    self.imei_mapping[imei] = (idevice, int(created))
        except FileNotFoundError:
            pass

    async def get_idevice(self, imeis: List[str]):
        unknowns, mapping = [], {}
        for imei in imeis:
            try:
                mapping[imei] = self.imei_mapping[imei]
            except KeyError:
                unknowns.append(imei)
        else:
            if len(unknowns) != 0:
                _mapping = \
                    await self._fetch_idevice_imei_map(unknowns)
                mapping.update(_mapping)
            return mapping

    @classmethod
    async def create(cls, live_dir: str):
        manager = IMEIMapper(live_dir)
        await manager._load_idevice_imei_map()
        return manager

    @property
    def mapping_file(self):
        return os.path.join(self.live_dir, "idevice.csv")


# (imei, timestamp)
IDEVICE_QUERY = Tuple[str, int]

# attributes for get request of idevice log
IDEVICE_FETCH = Tuple[str, int, int]


def _parse_logs(logs: List[dict], idevice: str):
    _attachments, _buffer = [], None
    for log_obj in reversed(logs):
        if "did_attach" not in log_obj:
            raise ValueError("malformed log object")
        if log_obj["did_attach"]:
            if _buffer is not None:
                raise ValueError("unexpected attach log")
            _buffer = log_obj
        else:
            if _buffer is None:
                raise ValueError("unexpected detach log")
            _attachments.append((_buffer, log_obj))
            _buffer = None
    else:
        attachments = []
        for log0, log1 in _attachments:
            vehicle = extract_from_two(log0, log1, "vehicle_id")
            plate = extract_from_two(log0, log1, "vehicle_plate")
            account = extract_from_two(log0, log1, "account_id")
            attachments.append((idevice, log0["timestamp"],
                                log1["timestamp"], vehicle, plate, account))
        if _buffer is not None:
            current_time = int(time.time() * 1000)
            vehicle, plate, account = (
                _buffer["vehicle_id"], _buffer["vehicle_plate"],
                _buffer["account_id"])
            _entry = (idevice, _buffer["timestamp"],
                      current_time, vehicle, plate, account)
            attachments.append(_entry)
            return attachments, True
    return attachments, False


class VehicleMapper:
    IDEVICE_LOG_TEMPL = \
        "http://apis.intangles.com/idevice_attach_detach_logs/" \
        "{start}/{end}?idevice_ids={idevice}&token={token}"
    TOKEN = "Sb11ZE7miFpKRDc5q_sv0DYmIurK5v1lFPr4FRKvg4-spMBt4WeDAU9C32iw1Vcs"
    CSV_ROW_TEMPL = "{idevice},{start},{end},{vehicle},{plate},{account}\n"

    def __init__(self, live_dir: str):
        if not os.path.exists(live_dir):
            raise FileNotFoundError(live_dir)
        self.live_dir = live_dir
        self.idevice_mapping: Dict[str, pd.DataFrame] = {}
        self.imei_mapper = IMEIMapper(live_dir)

    async def __fetch_idevice_log(
            self, unknowns: List[IDEVICE_FETCH]):
        async with aiohttp.ClientSession() as session:
            for idevice, start, end in unknowns:
                idevice_log_url = self.IDEVICE_LOG_TEMPL.format(
                    idevice=idevice, start=start, end=end, token=self.TOKEN)
                async with session.get(idevice_log_url) as response:
                    _response = await response.json()
                    attachments, attached = \
                        _parse_logs(_response["result"]["logs"], idevice)
                    if len(attachments) == 0:
                        return
                    for attachment in attachments[:-1]:
                        self._update_idevice_mapping(attachment)
                        yield attachment
                    self._update_idevice_mapping(attachments[-1])
                    if not attached:
                        yield attachments[-1]

    def _update_idevice_mapping(self, entry: tuple):
        idevice, start, end, vehicle, plate, account = entry
        vals = (vehicle, plate, account)
        interval = pd.Interval(int(start), int(end))
        _columns = ("vehicle", "plate", "account")

        if idevice in self.idevice_mapping:
            _idevice_frame = self.idevice_mapping[idevice]
            _idevice_frame.loc[interval] = vals
        else:
            index = pd.IntervalIndex([interval])
            self.idevice_mapping[idevice] = \
                pd.DataFrame([vals], index=index, columns=_columns)

    async def _fetch_vehicle_idevice_map(
            self, queries: List[IDEVICE_QUERY]):
        unknowns = []
        for idevice, start, end, created in queries:
            if idevice in self.idevice_mapping:
                _log = self.idevice_mapping[idevice]
                _start, _end = _log.index[-1][1], int(time.time() * 1000)
                unknowns.append((idevice, _start, _end))

    async def _load_vehicle_idevice_map(self):
        try:
            async with aiofiles.open(self.mapping_file, "r") as inp:
                async for line in inp:
                    self._update_idevice_mapping(line[:-1].split(","))
        except FileNotFoundError:
            pass

    async def get_vehicle(self, queries: pd.DataFrame):
        _imeis = queries.index.get_level_values(0)
        _idevice_mapping = await self.imei_mapper.get_idevice(_imeis.unique())
        idevice_mapping = \
            {imei: _idevice[0] for imei, _idevice in _idevice_mapping.items()}
        created_mapping = \
            {imei: _idevice[1] for imei, _idevice in _idevice_mapping.items()}
        queries["idevice"], queries["created"] = \
            _imeis.map(idevice_mapping), _imeis.map(created_mapping)

        _unknowns = []
        iterator = queries[["idevice", "created"]].iterrows()
        for (imei, epoch), (idevice, created) in iterator:
            try:
                queries.loc[(imei, epoch), ["vehicle", "plate", "account"]] = \
                    self.idevice_mapping[idevice].loc[epoch]
            except KeyError:
                _unknowns.append(((imei, epoch), idevice, created))
        else:
            _columns = ["idx", "idevice", "created"]
            unknowns = pd.DataFrame(_unknowns, columns=_columns)
            _created = unknowns.groupby("idevice")\
                .apply(lambda x: x["created"].unique()[0]).to_dict()
            fetch_query = []
            for idevice in _created.keys():
                _current_time = int(time.time() * 1000)
                if idevice in self.idevice_mapping:
                    _idevice_frame = self.idevice_mapping[idevice]
                    _start = _idevice_frame.index[-1].right + 1
                    fetch_query.append((idevice, _start, _current_time))
                else:
                    query = (idevice, _created[idevice], _current_time)
                    fetch_query.append(query)
            else:
                attachments = self.__fetch_idevice_log(fetch_query)
                async with aiofiles.open(self.mapping_file, "a") as out:
                    async for attach in attachments:
                        formatter = dict(
                            idevice=attach[0], start=attach[1], end=attach[2],
                            vehicle=attach[3], plate=attach[4], account=attach[5])
                        row = self.CSV_ROW_TEMPL.format(**formatter)
                        await out.write(row)

            for _, ((imei, epoch), idevice, __) in unknowns.iterrows():
                try:
                    queries.loc[(imei, epoch), ["vehicle", "plate", "account"]] = \
                        self.idevice_mapping[idevice].loc[epoch]
                except KeyError:
                    continue

        return queries

    @classmethod
    async def create(cls, live_dir: str):
        mapper = VehicleMapper(live_dir)
        mapper.imei_mapper = await IMEIMapper.create(live_dir)
        await mapper._load_vehicle_idevice_map()
        return mapper

    @property
    def mapping_file(self):
        return os.path.join(self.live_dir, "attachment.csv")


def make_query_frame(live_dir: str):
    basename, join, listdir = os.path.basename, os.path.join, os.listdir

    image_dir, paths = join(live_dir, "images"), []
    dirs = map(lambda d: join(image_dir, d), listdir(image_dir))
    imeis = filter(lambda dname: basename(dname).isnumeric(), dirs)
    _index, queries = [], pd.DataFrame(columns=[
        "file_name", "idevice", "created", "vehicle", "plate", "account"])

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


async def load_imei_logs(live_dir: str, imeis: Optional[List[str]]=[],
                         start: Optional[int]=None, end: Optional[int]=None,
                         silent: Optional[bool]=False):
    _queries, vehicles = make_query_frame(live_dir)
    if len(imeis) != 0:
        _queries = _queries.loc[imeis]
    if start > 0:
        times = _queries.index.get_level_values(1)
        _queries = _queries[times >= start]
    if end > 0:
        times = _queries.index.get_level_values(1)
        _queries = _queries[times < end]
    vehicle_mapper = await VehicleMapper.create(live_dir)
    _queries = await vehicle_mapper.get_vehicle(_queries)

    if vehicles is not None:
        queries = pd.concat((_queries, vehicles), sort=True)
    else:
        queries = _queries

    queries.sort_values("epoch", inplace=True)
    queries.to_csv(os.path.join(live_dir, "vehicles.csv"))

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
