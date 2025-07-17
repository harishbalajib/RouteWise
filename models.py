from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import time
import re

class Place(BaseModel):
    place_name: str = Field(..., min_length=1, max_length=100)
    address: str = Field(..., min_length=1, max_length=200)
    open_time: str = Field(..., pattern=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    close_time: str = Field(..., pattern=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    visit_duration: int = Field(..., ge=1, le=480)  # 1 minute to 8 hours
    preference: int = Field(..., ge=1, le=5)
    preferred_start_time: Optional[str] = Field(None, pattern=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    preferred_end_time: Optional[str] = Field(None, pattern=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    timing_importance: int = Field(1, ge=1, le=5)  # How important is the preferred timing
    
    @validator('open_time', 'close_time', 'preferred_start_time', 'preferred_end_time')
    def validate_time_format(cls, v):
        if v is None:
            return v
        if not re.match(r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$", v):
            raise ValueError("Time must be in HH:MM format")
        return v
    
    @validator('close_time')
    def validate_close_after_open(cls, v, values):
        if 'open_time' in values and v:
            open_time = time.fromisoformat(values['open_time'])
            close_time = time.fromisoformat(v)
            if close_time <= open_time:
                raise ValueError("Close time must be after open time")
        return v
    
    @validator('preferred_end_time')
    def validate_preferred_end_after_start(cls, v, values):
        if 'preferred_start_time' in values and v and values['preferred_start_time']:
            start_time = time.fromisoformat(values['preferred_start_time'])
            end_time = time.fromisoformat(v)
            if end_time <= start_time:
                raise ValueError("Preferred end time must be after preferred start time")
        return v
    
    @validator('preferred_start_time', 'preferred_end_time')
    def validate_preferred_times_within_opening_hours(cls, v, values):
        if v and 'open_time' in values and 'close_time' in values:
            if values['open_time'] and values['close_time']:
                open_time = time.fromisoformat(values['open_time'])
                close_time = time.fromisoformat(values['close_time'])
                preferred_time = time.fromisoformat(v)
                
                if preferred_time < open_time or preferred_time > close_time:
                    raise ValueError(f"Preferred time {v} must be within opening hours ({values['open_time']} - {values['close_time']})")
        return v

class TravelRequest(BaseModel):
    places: List[Place] = Field(..., min_items=1, max_items=20)
    start_time: str = Field(..., pattern=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    end_time: str = Field(..., pattern=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    num_days: int = Field(..., ge=1, le=14)
    travel_mode: str = Field(..., pattern=r"^(driving|walking)$")
    algorithm: str = Field(..., pattern=r"^(greedy|ortools|brute|dynamic_programming|optimal|genetic_annealing)$")
    
    @validator('end_time')
    def validate_end_after_start(cls, v, values):
        if 'start_time' in values:
            start_time = time.fromisoformat(values['start_time'])
            end_time = time.fromisoformat(v)
            if end_time <= start_time:
                raise ValueError("End time must be after start time")
        return v
    
    @validator('algorithm')
    def validate_algorithm_for_places(cls, v, values):
        if 'places' in values and v == 'brute':
            if len(values['places']) > 7:
                raise ValueError("Brute force algorithm can only handle up to 7 places")
        return v

class ItineraryItem(BaseModel):
    place_name: str
    address: str
    arrival_time: str
    departure_time: str
    visit_order: int
    visit_duration: int

class DailyPlan(BaseModel):
    day: int
    itinerary: List[ItineraryItem]
    total_duration: int
    map_html: str
    legend: List[str]

class TravelResponse(BaseModel):
    algorithm_used: str
    total_route_duration: int
    execution_time: float
    daily_plans: List[DailyPlan] 