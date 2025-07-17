import time as time_module
from typing import List, Tuple
from models import Place, TravelRequest, ItineraryItem, DailyPlan, TravelResponse
from utils import TimeUtils, DistanceMatrix, RouteOptimizer, MapGenerator

class TravelPlanner:
    def __init__(self):
        self.distance_matrix = DistanceMatrix()
        self.route_optimizer = RouteOptimizer(self.distance_matrix)
        self.map_generator = MapGenerator()
    
    def plan_travel(self, request: TravelRequest) -> TravelResponse:
        """Main method to plan travel itinerary"""
        start_time = time_module.time()
        
        # Optimize the overall route
        optimized_route = self.route_optimizer.optimize_route(
            request.places, 
            request.algorithm, 
            request.travel_mode
        )
        print(f"DEBUG: Overall optimized route indices: {optimized_route}")
        print(f"DEBUG: Overall optimized route names: {[request.places[i].place_name for i in optimized_route]}")
        
        # Split into daily plans
        daily_plans = self._split_into_daily_plans(
            request.places,
            optimized_route,
            request.start_time,
            request.end_time,
            request.num_days,
            request.travel_mode
        )
        
        # Calculate total route duration
        total_duration = sum(plan.total_duration for plan in daily_plans)
        
        execution_time = time_module.time() - start_time
        
        return TravelResponse(
            algorithm_used=request.algorithm,
            total_route_duration=total_duration,
            execution_time=execution_time,
            daily_plans=daily_plans
        )
    
    def _split_into_daily_plans(
        self, 
        places: List[Place], 
        optimized_route: List[int], 
        start_time: str, 
        end_time: str, 
        num_days: int, 
        travel_mode: str
    ) -> List[DailyPlan]:
        """Split optimized route into daily plans based on time constraints"""
        
        # Calculate available time per day
        available_minutes = TimeUtils.time_to_minutes(end_time) - TimeUtils.time_to_minutes(start_time)
        
        # Calculate total time needed for each place
        place_times = []
        for i, place_idx in enumerate(optimized_route):
            place = places[place_idx]
            visit_time = place.visit_duration
            
            # Add travel time to next place (if not last)
            if i < len(optimized_route) - 1:
                next_place_idx = optimized_route[i + 1]
                travel_time = self.distance_matrix.get_travel_time(
                    place.address,
                    places[next_place_idx].address,
                    travel_mode
                )
            else:
                travel_time = 0
            
            place_times.append((place_idx, visit_time, travel_time))
        
        # Distribute places across days
        daily_plans = []
        current_day = 1
        current_day_places = []
        current_day_time = 0
        
        for place_idx, visit_time, travel_time in place_times:
            total_time_needed = visit_time + travel_time
            
            # Check if this place fits in current day
            if current_day_time + total_time_needed <= available_minutes:
                current_day_places.append(place_idx)
                current_day_time += total_time_needed
            else:
                # Create plan for current day
                if current_day_places:
                    daily_plans.append(self._create_daily_plan(
                        places, current_day_places, start_time, current_day
                    ))
                
                # Start new day
                current_day += 1
                if current_day > num_days:
                    break  # Stop if we exceed requested days
                
                current_day_places = [place_idx]
                current_day_time = total_time_needed
        
        # Add final day if there are remaining places
        if current_day_places and current_day <= num_days:
            daily_plans.append(self._create_daily_plan(
                places, current_day_places, start_time, current_day
            ))
        
        return daily_plans
    
    def _create_daily_plan(
        self, 
        places: List[Place], 
        day_places: List[int], 
        start_time: str, 
        day: int
    ) -> DailyPlan:
        """Create a detailed daily plan with itinerary and map"""
        
        itinerary = []
        current_time = start_time
        total_duration = 0
        
        for i, place_idx in enumerate(day_places):
            place = places[place_idx]
            
            # Calculate arrival time
            if i > 0:
                # Add travel time from previous place
                prev_place_idx = day_places[i - 1]
                travel_time = self.distance_matrix.get_travel_time(
                    places[prev_place_idx].address,
                    place.address,
                    "driving"  # Use driving for consistency
                )
                current_time = TimeUtils.add_minutes_to_time(current_time, travel_time)
                total_duration += travel_time
            
            arrival_time = current_time
            
            # Calculate departure time
            departure_time = TimeUtils.add_minutes_to_time(current_time, place.visit_duration)
            total_duration += place.visit_duration
            
            # Create itinerary item
            itinerary_item = ItineraryItem(
                place_name=place.place_name,
                address=place.address,
                arrival_time=arrival_time,
                departure_time=departure_time,
                visit_order=i + 1,
                visit_duration=place.visit_duration
            )
            
            itinerary.append(itinerary_item)
            current_time = departure_time
        
        # Generate interactive map
        place_addresses = [places[idx].address for idx in day_places]
        # Use the actual optimized route order for the map
        map_order = list(range(len(day_places)))  # This represents the order within this day's plan
        print(f"DEBUG: Day {day} - day_places indices: {day_places}")
        print(f"DEBUG: Day {day} - place names: {[places[idx].place_name for idx in day_places]}")
        print(f"DEBUG: Day {day} - map_order: {map_order}")
        map_html = self.map_generator.generate_interactive_map(place_addresses, map_order, day)
        
        # Create legend
        legend = []
        for i, place_idx in enumerate(day_places):
            label = chr(65 + i)  # A, B, C, etc.
            place = places[place_idx]
            legend.append(f"{label} - {i + 1}st place: {place.place_name}")
        
        return DailyPlan(
            day=day,
            itinerary=itinerary,
            total_duration=total_duration,
            map_html=map_html,
            legend=legend
        ) 