import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    # Updated to use newer Google Maps APIs
    GOOGLE_DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
    GOOGLE_STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"
    GOOGLE_PLACES_AUTOCOMPLETE_URL = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
    GOOGLE_PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
    
    # Alternative: Use Routes API for distance matrix (newer API)
    GOOGLE_ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
    
    # Cache settings
    CACHE_DURATION = 3600  # 1 hour in seconds
    
    # Algorithm settings
    MAX_BRUTE_FORCE_PLACES = 7
    
    # Time settings
    DEFAULT_START_TIME = "09:00"
    DEFAULT_END_TIME = "18:00"
    
    # Map settings
    MAP_WIDTH = 800
    MAP_HEIGHT = 600
    MAP_ZOOM = 12 