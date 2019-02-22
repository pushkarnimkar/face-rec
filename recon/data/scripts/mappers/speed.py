from sklearn.cluster import DBSCAN
from typing import List, Optional, Tuple

import aiofiles
import aiohttp
import asyncio
import os
import numpy as np
import pandas as pd


SPEED_FETCH = Tuple[str, int, int]


def _parse_speed(response):
    assert type(response) == list

    __times, _speeds, times, speeds = [], [], [], []
    for packet in response:
        if "multi_sp" not in packet or "timestamp" not in packet:
            continue
        __times.append(packet["timestamp"])
        _speeds.append(packet["multi_sp"])

    _times = np.array(__times).astype(np.int)
    if _times.shape[0] <= 1:
        return pd.DataFrame()

    for index in range(len(_times) - 1):
        ndiv, diff = len(_speeds[index]), _times[index + 1] - _times[index]
        for _index, __speed in enumerate(_speeds[index]):
            times.append(_times[index] + _index * diff // ndiv)
            speeds.append(__speed["sp"])

    return pd.DataFrame(np.array(speeds).reshape(-1, 1),
                        index=times, columns=["speed"])


class SpeedMapper:
    EXPECTED_COLS = ["vehicle"]
    NOISE_LOG_TEMPL = "for vehicle {vehicle} discarding " \
                      "{noise_perc:7.2f}% data as noise"
    SPEED_QUERY_TEMP = "http://data-download.intangles.com:1883" \
                       "/download/location/{vehicle}/{start}/{end}"
    CSV_ROW_TEMPL = "{vehicle},{epoch},{speed}\n"

    def __init__(self, live_dir: str, time_eps: Optional[int]=1800000,
                 time_min_sample: Optional[int]=3,
                 sample_margin: Optional[int]=10000):

        if not os.path.exists(live_dir):
            raise FileNotFoundError(live_dir)
        self.live_dir = live_dir
        self.time_eps = time_eps
        self.time_min_sample = time_min_sample
        self.sample_margin = sample_margin
        self.speed_map = pd.DataFrame()

    async def __fetch_idevice_log(self, unknowns: List[SPEED_FETCH]):
        async with aiohttp.ClientSession() as session:
            for vehicle, start, end in unknowns:
                speed_url = self.SPEED_QUERY_TEMP.format(
                    vehicle=vehicle, start=start, end=end)
                async with session.get(speed_url) as _response:
                    response = await _response.json()
                    yield vehicle, _parse_speed(response)
        return

    def _update_speed_mapping(self, entries: pd.DataFrame):
        vehicle, epoch, _speed = \
            entries["vehicle"], entries["epoch"], entries["speed"]
        epoch, speed = epoch.astype(np.int), _speed.astype(np.float)
        index = pd.MultiIndex.from_arrays([vehicle, epoch])
        _new_map = pd.DataFrame(speed.values.reshape(-1, 1),
                                columns=["speed"], index=index)

        if self.speed_map.shape != (0, 0):
            self.speed_map = self.speed_map.append(_new_map)
        else:
            self.speed_map = _new_map
        # self.speed_map.sort_index(level="epoch", inplace=True)

    async def _load_vehicle_speed_map(self):
        try:
            async with aiofiles.open(self.mapping_file, "r") as inp:
                vehicle, epoch, speed = [], [], []
                async for line in inp:
                    _vehicle, _epoch, _speed = line[:-1].split(",")
                    vehicle.append(_vehicle)
                    epoch.append(_epoch)
                    speed.append(_speed)
                else:
                    _columns = ["vehicle", "epoch", "speed"]
                    _data = np.stack([vehicle, epoch, speed]).T
                    _frame = pd.DataFrame(_data, columns=_columns)
                    self._update_speed_mapping(_frame)
        except FileNotFoundError:
            pass

    def _make_clusters(self, vehicle: str, epochs: pd.Series):
        model = DBSCAN(eps=self.time_eps, min_samples=self.time_min_sample)
        clusters = model.fit_predict(epochs.reshape(-1, 1))
        cluster_limits = []
        for cluster in np.unique(clusters):
            _data = epochs[clusters == cluster]
            if cluster == -1:
                noise_perc = _data.shape[0] / clusters.shape[0] * 100
                formatter = dict(vehicle=vehicle, noise_perc=noise_perc)
                print(self.NOISE_LOG_TEMPL.format(**formatter))
                continue
            cluster_limits.append((_data.min() - self.sample_margin,
                                   _data.max() + self.sample_margin))
        return cluster_limits

    async def get_speed(self, queries: pd.DataFrame):
        assert np.setdiff1d(self.EXPECTED_COLS, queries.columns).shape[0] == 0
        _required_columns = ["speed"]
        for _column_name in np.setdiff1d(_required_columns, queries.columns):
            queries[_column_name] = np.nan

        _unknowns = []
        iterator = queries.dropna(subset=["vehicle"])["vehicle"].items()
        for (imei, epoch), vehicle in iterator:
            try:
                speed = self.speed_map.loc[vehicle].loc[epoch]
                queries.loc[(imei, epoch), ["speed"]] = speed
            except KeyError:
                _unknowns.append((imei, epoch, vehicle))
        else:
            _columns = ["imei", "epoch", "vehicle"]
            unknowns = pd.DataFrame(_unknowns, columns=_columns)
            fetch_queries = []
            for vehicle, group in unknowns.groupby("vehicle"):
                _queries = self._make_clusters(vehicle, group["epoch"].values)
                for time_limits in _queries:
                    fetch_queries.append(
                        (vehicle, time_limits[0], time_limits[1]))
                    # formatter = dict(vehicle=vehicle, start=_query[1],
                    #                  minutes=(_query[2] - _query[1]) / 60000)
                    # print("{vehicle:19s}{start:14d}{minutes:10.4f}"
                    #       .format(**formatter))

            speed_logs = self.__fetch_idevice_log(fetch_queries)
            vehicle_groups = unknowns.groupby("vehicle")
            merge_kws = dict(right_index=True, left_on="epoch",
                             tolerance=10000, direction="nearest")
            async for vehicle, speed_log in speed_logs:
                if speed_log.shape == (0, 0):
                    continue
                try:
                    vehicle_frame = vehicle_groups.get_group(vehicle)
                    _merged = pd.merge_asof(
                        vehicle_frame, speed_log, **merge_kws)
                    merged = _merged[np.all(~_merged.isna(), axis=1)]
                    self._update_speed_mapping(merged)
                except Exception as exp:
                    print("exception", exp)
            else:
                async with aiofiles.open(self.mapping_file, "w") as out:
                    iterator, writes = self.speed_map.iterrows(), []
                    for (vehicle, epoch), (speed,) in iterator:
                        formatter = dict(
                            vehicle=vehicle, epoch=epoch, speed=speed)
                        writes.append(
                            out.write(self.CSV_ROW_TEMPL.format(**formatter)))

                    status = await asyncio.gather(*writes)

            for _, (imei, epoch, vehicle) in unknowns.iterrows():
                try:
                    speed = self.speed_map.loc[vehicle].loc[epoch]
                    queries.loc[(imei, epoch), ["speed"]] = speed
                except KeyError:
                    continue

        return queries, status

    @classmethod
    async def create(cls, live_dir: str):
        mapper = SpeedMapper(live_dir)
        await mapper._load_vehicle_speed_map()
        return mapper

    @property
    def mapping_file(self):
        return os.path.join(self.live_dir, "speed_map.csv")
