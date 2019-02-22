from typing import List

import aiofiles
import aiohttp
import os
import numpy as np
import pandas as pd


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

    async def _get_idevice(self, imeis: List[str]):
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

    async def get_idevice(self, queries: pd.DataFrame):
        _required_columns = ["idevice", "created"]
        for _column_name in np.setdiff1d(_required_columns, queries.columns):
            queries[_column_name] = np.nan

        _imeis = queries.index.get_level_values(0)
        _idevice_mapping = await self._get_idevice(_imeis.unique())
        idevice_mapping = \
            {imei: _idevice[0] for imei, _idevice in _idevice_mapping.items()}
        created_mapping = \
            {imei: _idevice[1] for imei, _idevice in _idevice_mapping.items()}
        queries["idevice"], queries["created"] = \
            _imeis.map(idevice_mapping), _imeis.map(created_mapping)
        return queries

    @classmethod
    async def create(cls, live_dir: str):
        manager = IMEIMapper(live_dir)
        await manager._load_idevice_imei_map()
        return manager

    @property
    def mapping_file(self):
        return os.path.join(self.live_dir, "idevice.csv")

