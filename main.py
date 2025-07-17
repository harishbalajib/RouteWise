from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from typing import List
import requests
from config import Config

from models import Place, TravelRequest, TravelResponse
from travel_planner import TravelPlanner
from utils import DistanceMatrix, RouteOptimizer

app = FastAPI(title="Smart US Travel Planner", version="1.0.0")

templates = Jinja2Templates(directory="templates")
planner = TravelPlanner()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/places/search")
async def search_places(query: str):
    if not Config.GOOGLE_API_KEY:
        return JSONResponse({"error": "Google Maps API key required"}, status_code=400)
    
    try:
        params = {
            'input': query,
            'types': 'establishment|geocode',
            'key': Config.GOOGLE_API_KEY
        }
        
        response = requests.get(Config.GOOGLE_PLACES_AUTOCOMPLETE_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'OK':
            predictions = []
            for prediction in data['predictions']:
                predictions.append({
                    'place_id': prediction['place_id'],
                    'description': prediction['description'],
                    'main_text': prediction['structured_formatting']['main_text'],
                    'secondary_text': prediction['structured_formatting'].get('secondary_text', '')
                })
            return JSONResponse({"predictions": predictions})
        else:
            error_msg = f"Google API error: {data['status']}"
            if 'error_message' in data:
                error_msg += f" - {data['error_message']}"
            
            if "legacy API" in error_msg or data['status'] == 'REQUEST_DENIED':
                error_msg = (
                    "Google Places API error: The legacy Places API is no longer enabled by default. "
                    "Please enable the Places API in your Google Cloud Console, "
                    "or upgrade to Places API (New)."
                )
            
            return JSONResponse({"error": error_msg}, status_code=400)
            
    except Exception as e:
        return JSONResponse({"error": f"Error searching places: {str(e)}"}, status_code=500)

@app.get("/api/places/details/{place_id}")
async def get_place_details(place_id: str):
    if not Config.GOOGLE_API_KEY:
        return JSONResponse({"error": "Google Maps API key required"}, status_code=400)
    
    try:
        params = {
            'place_id': place_id,
            'fields': 'name,formatted_address,opening_hours,geometry,photos,rating,user_ratings_total,types,business_status,price_level,website,formatted_phone_number',
            'key': Config.GOOGLE_API_KEY
        }
        
        response = requests.get(Config.GOOGLE_PLACE_DETAILS_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'OK':
            place = data['result']
            
            opening_hours_info = {}
            if 'opening_hours' in place:
                hours = place['opening_hours']
                opening_hours_info = {
                    'open_now': hours.get('open_now', False),
                    'periods': hours.get('periods', []),
                    'weekday_text': hours.get('weekday_text', [])
                }
            
            photo_url = None
            if 'photos' in place and place['photos']:
                photo_reference = place['photos'][0]['photo_reference']
                photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={Config.GOOGLE_API_KEY}"
            
            return JSONResponse({
                'name': place.get('name', ''),
                'address': place.get('formatted_address', ''),
                'opening_hours': opening_hours_info,
                'location': place.get('geometry', {}).get('location', {}),
                'rating': place.get('rating'),
                'user_ratings_total': place.get('user_ratings_total'),
                'types': place.get('types', []),
                'business_status': place.get('business_status'),
                'price_level': place.get('price_level'),
                'website': place.get('website'),
                'phone': place.get('formatted_phone_number'),
                'photo_url': photo_url
            })
        else:
            error_msg = f"Google API error: {data['status']}"
            if 'error_message' in data:
                error_msg += f" - {data['error_message']}"
            
            if "legacy API" in error_msg or data['status'] == 'REQUEST_DENIED':
                error_msg = (
                    "Google Places API error: The legacy Places API is no longer enabled by default. "
                    "Please enable the Places API in your Google Cloud Console, "
                    "or upgrade to Places API (New)."
                )
            
            return JSONResponse({"error": error_msg}, status_code=400)
            
    except Exception as e:
        return JSONResponse({"error": f"Error getting place details: {str(e)}"}, status_code=500)

@app.post("/plan", response_class=HTMLResponse)
async def plan_travel(request: Request):
    try:
        print("DEBUG: Received form data:", await request.form())
        form_data = await request.form()
        places = []
        place_names = form_data.getlist("place_name")
        addresses = form_data.getlist("address")
        open_times = form_data.getlist("open_time")
        close_times = form_data.getlist("close_time")
        visit_durations = form_data.getlist("visit_duration")
        preferences = form_data.getlist("preference")
        preferred_start_times = form_data.getlist("preferred_start_time")
        preferred_end_times = form_data.getlist("preferred_end_time")
        timing_importances = form_data.getlist("timing_importance")
        for i in range(len(place_names)):
            if place_names[i].strip():
                preferred_start = preferred_start_times[i].strip() if preferred_start_times[i].strip() else None
                preferred_end = preferred_end_times[i].strip() if preferred_end_times[i].strip() else None
                timing_importance = int(timing_importances[i]) if timing_importances[i] else 1
                place = Place(
                    place_name=place_names[i].strip(),
                    address=addresses[i].strip(),
                    open_time=open_times[i].strip(),
                    close_time=close_times[i].strip(),
                    visit_duration=int(visit_durations[i]),
                    preference=int(preferences[i]),
                    preferred_start_time=preferred_start,
                    preferred_end_time=preferred_end,
                    timing_importance=timing_importance
                )
                places.append(place)
        if not places:
            raise HTTPException(status_code=400, detail="At least one place is required")
        addresses = [p.address for p in places]
        mode = form_data.get("travel_mode", "driving")
        start_time = form_data.get("start_time", "")
        end_time = form_data.get("end_time", "")
        num_days = form_data.get("num_days", "")
        # Build the full travel time and distance matrices
        n = len(addresses)
        matrix_time = [[0 for _ in range(n)] for _ in range(n)]
        matrix_distance = [[0 for _ in range(n)] for _ in range(n)]
        distance_matrix = DistanceMatrix()
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix_time[i][j] = 0
                    matrix_distance[i][j] = 0
                else:
                    t, d = distance_matrix.get_travel_time_and_distance(addresses[i], addresses[j], mode)
                    matrix_time[i][j] = t
                    matrix_distance[i][j] = d
        matrix = distance_matrix.get_full_matrix(addresses, mode)
        # Run all algorithms and compute both time and distance
        def compute_metrics(route):
            total_time = 0
            total_distance = 0
            for i in range(len(route) - 1):
                t, d = distance_matrix.get_travel_time_and_distance(
                    places[route[i]].address,
                    places[route[i+1]].address,
                    mode
                )
                total_time += t
                total_distance += d
            return total_time, total_distance

        results = []
        # Greedy
        greedy_route = RouteOptimizer(distance_matrix).optimize_route(places, 'greedy', mode)
        greedy_time, greedy_distance = compute_metrics(greedy_route)
        results.append({
            'algorithm': 'Greedy',
            'route': greedy_route,
            'route_names': [places[i].place_name for i in greedy_route],
            'total_time': greedy_time,
            'total_distance': greedy_distance
        })
        # Dynamic Programming
        if len(places) <= 12:
            dp_route = RouteOptimizer(distance_matrix).optimize_route(places, 'dynamic_programming', mode)
            dp_time, dp_distance = compute_metrics(dp_route)
            results.append({
                'algorithm': 'Dynamic Programming',
                'route': dp_route,
                'route_names': [places[i].place_name for i in dp_route],
                'total_time': dp_time,
                'total_distance': dp_distance
            })
        # OR-Tools
        ortools_route = RouteOptimizer(distance_matrix).optimize_route(places, 'ortools', mode)
        ortools_time, ortools_distance = compute_metrics(ortools_route)
        results.append({
            'algorithm': 'OR-Tools',
            'route': ortools_route,
            'route_names': [places[i].place_name for i in ortools_route],
            'total_time': ortools_time,
            'total_distance': ortools_distance
        })
        # Brute Force (last)
        if len(places) <= Config.MAX_BRUTE_FORCE_PLACES:
            brute_route = RouteOptimizer(distance_matrix).optimize_route(places, 'brute', mode)
            brute_time, brute_distance = compute_metrics(brute_route)
            results.append({
                'algorithm': 'Brute Force',
                'route': brute_route,
                'route_names': [places[i].place_name for i in brute_route],
                'total_time': brute_time,
                'total_distance': brute_distance
            })
        
        # Genetic Annealing
        genetic_annealing_route = RouteOptimizer(distance_matrix).optimize_route(places, 'genetic_annealing', mode)
        genetic_annealing_time, genetic_annealing_distance = compute_metrics(genetic_annealing_route)
        results.append({
            'algorithm': 'Genetic Annealing',
            'route': genetic_annealing_route,
            'route_names': [places[i].place_name for i in genetic_annealing_route],
            'total_time': genetic_annealing_time,
            'total_distance': genetic_annealing_distance
        })
        # Sort by total_time
        results.sort(key=lambda x: x['total_time'])
        print("DEBUG: form_data =", form_data)
        print("DEBUG: place_names =", place_names)
        print("DEBUG: addresses =", addresses)
        print("DEBUG: places =", places)
        print("DEBUG: results =", results)
        if not results:
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "results": [],
                    "places": places,
                    "start_time": start_time,
                    "end_time": end_time,
                    "travel_mode": mode,
                    "num_days": num_days,
                    "error": "No results found. Please check your input and API key."
                }
            )
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "results": results,
                "places": places,
                "start_time": start_time,
                "end_time": end_time,
                "travel_mode": mode,
                "num_days": num_days,
                "matrix_time": matrix_time,
                "matrix_distance": matrix_distance,
                "Config": Config,
                "enumerate": enumerate
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": str(e)}
        )

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def devtools_config():
    return {"status": "not configured"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Smart US Travel Planner"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 