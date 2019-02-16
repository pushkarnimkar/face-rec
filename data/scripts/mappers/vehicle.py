from typing import Tuple, List, Dict
from data.scripts.utils import extract_from_two

import aiofiles
import aiohttp
import numpy as np
import os
import pandas as pd
import time


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
    EXPECTED_COLS = ["idevice", "created"]

    def __init__(self, live_dir: str):
        if not os.path.exists(live_dir):
            raise FileNotFoundError(live_dir)
        self.live_dir = live_dir
        self.idevice_mapping: Dict[str, pd.DataFrame] = {}

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

    async def _load_vehicle_idevice_map(self):
        try:
            async with aiofiles.open(self.mapping_file, "r") as inp:
                async for line in inp:
                    self._update_idevice_mapping(line[:-1].split(","))
        except FileNotFoundError:
            pass

    async def get_vehicle(self, queries: pd.DataFrame):
        assert np.setdiff1d(self.EXPECTED_COLS, queries.columns).shape[0] == 0

        _required_columns = ["vehicle", "plate", "account"]
        for _column_name in np.setdiff1d(_required_columns, queries.columns):
            queries[_column_name] = np.nan

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
        await mapper._load_vehicle_idevice_map()
        return mapper

    @property
    def mapping_file(self):
        return os.path.join(self.live_dir, "attachment.csv")
