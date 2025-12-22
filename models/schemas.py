'''
  Pydantic schema enforcement
'''
from pydantic import BaseModel, conint, confloat
from datetime import datetime, date
from typing import Optional, List


class DataInput(BaseModel):
    user_id: str
    data_payload: str  # placeholder, for more complex payloads


class DataOutput(BaseModel):
    message: str
    task_id: str  # this is the root cause for 'task_id' missing in api response


class IngestRequest(BaseModel):
    data: List[dict]
    data_type: str = "reserve"
    table_name: str = "air_reserve"


class AirReserve(BaseModel):
    air_store_id: str
    visit_datetime: datetime
    reserve_datetime: datetime
    reserve_visitors: conint(ge=0)  # non-negative int


class AirStoreInfo(BaseModel):
    air_store_id: str
    air_genre_name: str
    air_area_name: str
    latitude: confloat(ge=-90, le=90)
    longitude: confloat(ge=-180, le=180)


class AirStoreVisitData(BaseModel):
    air_store_id: str
    visit_date: date
    visitors: conint(ge=0)


class DateInfo(BaseModel):
    calendar_date: datetime
    day_of_week: str
    holiday_flg: conint(ge=0, le=1)


class HpgStoreInfo(BaseModel):
    hpg_store_id: str
    hpg_genre_name: str
    hpg_area_name: str
    latitude: confloat(ge=-90, le=90)
    longitude: confloat(ge=-180, le=180)


class SampleSubmission(BaseModel):
    id: str
    visitors: conint(ge=0)


class StoreIDRelation(BaseModel):
    air_store_id: str
    hpg_store_id: str