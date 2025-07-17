import time as time_module
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
from config import Config

class TimeUtils:
    @staticmethod
    def time_to_minutes(time_str: str) -> int:
        """Convert HH:MM format to minutes since midnight"""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    @staticmethod
    def minutes_to_time(minutes: int) -> str:
        """Convert minutes since midnight to HH:MM format"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    @staticmethod
    def add_minutes_to_time(time_str: str, minutes: int) -> str:
        """Add minutes to a time string and return new time string"""
        total_minutes = TimeUtils.time_to_minutes(time_str) + minutes
        return TimeUtils.minutes_to_time(total_minutes)
    
    @staticmethod
    def is_time_between(time_str: str, start_str: str, end_str: str) -> bool:
        """Check if time is between start and end times"""
        time_minutes = TimeUtils.time_to_minutes(time_str)
        start_minutes = TimeUtils.time_to_minutes(start_str)
        end_minutes = TimeUtils.time_to_minutes(end_str)
        return start_minutes <= time_minutes <= end_minutes

class DistanceMatrix:
    def __init__(self):
        self.cache: Dict[Tuple[str, str, str], Tuple[int, int, float]] = {}
        self.api_key = Config.GOOGLE_API_KEY
    
    def get_travel_time_and_distance(self, origin: str, destination: str, mode: str) -> Tuple[int, int]:
        """Get travel time (minutes) and distance (meters) between two locations"""
        cache_key = (origin, destination, mode)
        if cache_key in self.cache:
            cached_time, cached_distance, cached_timestamp = self.cache[cache_key]
            if time_module.time() - cached_timestamp < Config.CACHE_DURATION:
                return cached_time, cached_distance
        if not self.api_key:
            raise ValueError("Google Maps API key is required. Please add GOOGLE_API_KEY to your .env file")
        try:
            params = {
                'origins': origin,
                'destinations': destination,
                'mode': mode,
                'key': self.api_key
            }
            response = requests.get(Config.GOOGLE_DISTANCE_MATRIX_URL, params=params)
            response.raise_for_status()
            data = response.json()
            if data['status'] == 'OK' and data['rows'][0]['elements'][0]['status'] == 'OK':
                duration = data['rows'][0]['elements'][0]['duration']['value'] // 60  # minutes
                distance = data['rows'][0]['elements'][0]['distance']['value']  # meters
                self.cache[cache_key] = (duration, distance, time_module.time())
                return duration, distance
            else:
                error_msg = f"Google API error: {data.get('status', 'Unknown error')}"
                if 'error_message' in data:
                    error_msg += f" - {data['error_message']}"
                raise ValueError(error_msg)
        except Exception as e:
            raise ValueError(f"Error fetching travel time/distance from Google API: {e}")

    def get_travel_time(self, origin: str, destination: str, mode: str) -> int:
        """Get travel time between two locations in minutes (for backward compatibility)"""
        time, _ = self.get_travel_time_and_distance(origin, destination, mode)
        return time
    
    def get_full_matrix(self, addresses: List[str], mode: str) -> List[List[int]]:
        """Fetch the full NxN travel time matrix using Google Distance Matrix API in one call."""
        n = len(addresses)
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        if not self.api_key:
            raise ValueError("Google Maps API key is required. Please add GOOGLE_API_KEY to your .env file")
        params = {
            'origins': '|'.join(addresses),
            'destinations': '|'.join(addresses),
            'mode': mode,
            'key': self.api_key
        }
        response = requests.get(Config.GOOGLE_DISTANCE_MATRIX_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data['status'] != 'OK':
            raise ValueError(f"Google API error: {data.get('status', 'Unknown error')}")
        for i, row in enumerate(data['rows']):
            for j, element in enumerate(row['elements']):
                if element['status'] == 'OK':
                    matrix[i][j] = element['duration']['value'] // 60  # minutes
                else:
                    matrix[i][j] = float('inf')
        return matrix

class MapGenerator:
    def __init__(self):
        self.api_key = Config.GOOGLE_API_KEY
    
    def generate_static_map(self, places: List[str], path_order: List[int], day: int) -> str:
        """Generate Google Static Map URL for a day's itinerary"""
        if not self.api_key:
            raise ValueError("Google Maps API key is required. Please add GOOGLE_API_KEY to your .env file")
        
        # Create markers for each place
        markers = []
        for i, place_idx in enumerate(path_order):
            label = chr(65 + i)  # A, B, C, etc.
            markers.append(f"label:{label}|{places[place_idx]}")
        
        # Create path for the route
        path_addresses = [places[idx] for idx in path_order]
        path = f"color:0xff0000ff|weight:5|{'|'.join(path_addresses)}"
        
        # Build URL
        params = {
            'size': f"{Config.MAP_WIDTH}x{Config.MAP_HEIGHT}",
            'zoom': Config.MAP_ZOOM,
            'markers': '&markers='.join(markers),
            'path': path,
            'key': self.api_key
        }
        
        url = f"{Config.GOOGLE_STATIC_MAPS_URL}?size={params['size']}&zoom={params['zoom']}&markers={'&markers='.join(markers)}&path={path}&key={self.api_key}"
        
        return url
    
    def generate_interactive_map(self, places: List[str], path_order: List[int], day: int) -> str:
        """Generate interactive Google Maps with route visualization"""
        if not self.api_key:
            return ""
        
        if not places:
            return ""
        
        # Create a unique ID for this map
        map_id = f"map-day-{day}"
        
        # Get places in the correct order
        ordered_places = [places[idx] for idx in path_order]
        
        # Generate the map HTML with JavaScript
        map_html = f"""
        <div id="{map_id}" style="width: 100%; height: 400px; border-radius: 8px; margin: 1rem 0; position: relative;"></div>
        <script>
            function initMap{day}() {{
                const map = new google.maps.Map(document.getElementById('{map_id}'), {{
                    zoom: 10,
                    center: {{ lat: 40.7128, lng: -74.0060 }}, // Default to NYC
                    mapTypeId: google.maps.MapTypeId.ROADMAP
                }});
                
                const bounds = new google.maps.LatLngBounds();
                const markers = [];
                const positions = [];
                const directionsService = new google.maps.DirectionsService();
                
                // Add markers for each place
                const places = {ordered_places};
                const labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'];
                
                let markersLoaded = 0;
                
                places.forEach((place, index) => {{
                    const geocoder = new google.maps.Geocoder();
                    geocoder.geocode({{ address: place }}, (results, status) => {{
                        if (status === 'OK') {{
                            const position = results[0].geometry.location;
                            
                            // Create marker
                            const marker = new google.maps.Marker({{
                                position: position,
                                map: map,
                                label: {{
                                    text: labels[index],
                                    color: 'white',
                                    fontWeight: 'bold'
                                }},
                                icon: {{
                                    path: google.maps.SymbolPath.CIRCLE,
                                    scale: 12,
                                    fillColor: '#E53E3E',
                                    fillOpacity: 1,
                                    strokeColor: '#FFFFFF',
                                    strokeWeight: 2
                                }},
                                title: place
                            }});
                            
                            // Add info window
                            const infoWindow = new google.maps.InfoWindow({{
                                content: `<div style="padding: 10px; max-width: 200px;">
                                    <h4 style="margin: 0 0 5px 0; color: #333;">${{index + 1}}. ${{place}}</h4>
                                    <p style="margin: 0; color: #666; font-size: 14px;">Visit Order: ${{index + 1}}st</p>
                                    <p style="margin: 0; color: #666; font-size: 14px;">Label: ${{labels[index]}}</p>
                                </div>`
                            }});
                            
                            marker.addListener('click', () => {{
                                infoWindow.open(map, marker);
                            }});
                            
                            markers.push(marker);
                            positions[index] = position;
                            bounds.extend(position);
                        }}
                        
                        markersLoaded++;
                        
                        // If all markers are loaded, fit bounds and draw route
                        if (markersLoaded === places.length) {{
                            console.log('All markers loaded, total:', markersLoaded);
                            
                            // Fit bounds with padding
                            if (!bounds.isEmpty()) {{
                                map.fitBounds(bounds);
                                // Add padding after bounds are set
                                google.maps.event.addListenerOnce(map, 'bounds_changed', () => {{
                                    const currentBounds = map.getBounds();
                                    if (currentBounds) {{
                                        const ne = currentBounds.getNorthEast();
                                        const sw = currentBounds.getSouthWest();
                                        const latSpan = ne.lat() - sw.lat();
                                        const lngSpan = ne.lng() - sw.lng();
                                        
                                        const newNe = new google.maps.LatLng(ne.lat() + latSpan * 0.15, ne.lng() + lngSpan * 0.15);
                                        const newSw = new google.maps.LatLng(sw.lat() - latSpan * 0.15, sw.lng() - lngSpan * 0.15);
                                        const paddedBounds = new google.maps.LatLngBounds(newSw, newNe);
                                        map.fitBounds(paddedBounds);
                                    }}
                                }});
                            }}
                            
                            // Draw the route
                            console.log('Calling drawRoute function');
                            drawRoute();
                        }}
                    }});
                }});
                
                function drawRoute() {{
                    if (places.length < 2) return;
                    
                    console.log('Drawing route for places:', places);
                    console.log('Positions array:', positions);
                    
                    // Draw a simple polyline connecting the markers in order
                    if (positions.length >= 2) {{
                        const path = positions.map(pos => pos);
                        
                        const polyline = new google.maps.Polyline({{
                            path: path,
                            geodesic: true,
                            strokeColor: '#FF0000',
                            strokeOpacity: 1.0,
                            strokeWeight: 4,
                            icons: [{{
                                icon: {{
                                    path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW
                                }},
                                offset: '50%',
                                repeat: '100px'
                            }}]
                        }});
                        
                        polyline.setMap(map);
                        console.log('Simple polyline drawn successfully');
                        
                        // Add route info panel
                        const routeInfo = document.createElement('div');
                        routeInfo.style.cssText = `
                            position: absolute;
                            top: 10px;
                            right: 10px;
                            background: white;
                            padding: 10px;
                            border-radius: 8px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                            font-size: 12px;
                            z-index: 1000;
                            max-width: 200px;
                        `;
                        routeInfo.innerHTML = `
                            <div style="font-weight: bold; margin-bottom: 5px;">Algorithm's Optimal Route</div>
                            <div style="margin-bottom: 8px; font-size: 11px; color: #666;">Red line shows visit order</div>
                            <div>Stops: ${{places.length}}</div>
                            <div style="margin-top: 8px; font-size: 11px; color: #666;">Numbers show visit order</div>
                        `;
                        
                        document.getElementById('{map_id}').appendChild(routeInfo);
                        
                        // Add numbered markers for visit order
                        positions.forEach((position, index) => {{
                            new google.maps.Marker({{
                                position: position,
                                map: map,
                                label: {{
                                    text: (index + 1).toString(),
                                    color: 'white',
                                    fontWeight: 'bold',
                                    fontSize: '12px'
                                }},
                                icon: {{
                                    path: google.maps.SymbolPath.CIRCLE,
                                    scale: 8,
                                    fillColor: '#FF0000',
                                    fillOpacity: 0.8,
                                    strokeColor: '#FFFFFF',
                                    strokeWeight: 2
                                }},
                                zIndex: 1000
                            }});
                        }});
                    }} else {{
                        console.error('Not enough positions to draw route');
                    }}
                }}
            }}
            
            // Initialize map when Google Maps API is loaded
            if (typeof google !== 'undefined' && google.maps) {{
                initMap{day}();
            }} else {{
                window.initMap{day} = initMap{day};
            }}
        </script>
        """
        
        return map_html

class RouteOptimizer:
    def __init__(self, distance_matrix: DistanceMatrix):
        self.distance_matrix = distance_matrix
    
    def optimize_route(self, places: List, algorithm: str, travel_mode: str) -> List[int]:
        """Optimize route using specified algorithm"""
        if algorithm == "greedy":
            return self._greedy_algorithm(places, travel_mode)
        elif algorithm == "ortools":
            return self._ortools_algorithm(places, travel_mode)
        elif algorithm == "brute":
            return self._brute_force_algorithm(places, travel_mode)
        elif algorithm == "dynamic_programming":
            return self._dynamic_programming_algorithm(places, travel_mode)
        elif algorithm == "optimal":
            return self._optimal_algorithm(places, travel_mode)
        elif algorithm == "genetic_annealing":
            return self._genetic_annealing_algorithm(places, travel_mode)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _greedy_algorithm(self, places: List, travel_mode: str) -> List[int]:
        n = len(places)
        if n == 0:
            return []
        
        print(f"Greedy: Optimizing route for {n} places")
        
        # Try all possible starting points to find the globally optimal route
        best_route = None
        best_total_time = float('inf')
        
        for start_point in range(n):
            unvisited = set(range(n))
            route = [start_point]  # Start with current starting point
            unvisited.remove(start_point)
            
            while unvisited:
                current = route[-1]
                best_next = None
                best_time = float('inf')
                
                for next_place in unvisited:
                    travel_time = self.distance_matrix.get_travel_time(
                        places[current].address, 
                        places[next_place].address, 
                        travel_mode
                    )
                    
                    if travel_time < best_time:
                        best_next = next_place
                        best_time = travel_time
                
                if best_next is not None:
                    route.append(best_next)
                    unvisited.remove(best_next)
                else:
                    break
            
            # Calculate total time for this route
            total_time = 0
            for i in range(len(route) - 1):
                total_time += self.distance_matrix.get_travel_time(
                    places[route[i]].address,
                    places[route[i + 1]].address,
                    travel_mode
                )
            
            # Update best route if this one is better
            if total_time < best_total_time:
                best_total_time = total_time
                best_route = route.copy()
        
        print(f"Greedy: Best route found: {best_route} with total time: {best_total_time}")
        return best_route
    
    def _ortools_algorithm(self, places: List, travel_mode: str) -> List[int]:
        try:
            from ortools.sat.python import cp_model
            
            n = len(places)
            if n == 0:
                return []
            
            # Create distance matrix
            distances = []
            for i in range(n):
                row = []
                for j in range(n):
                    if i == j:
                        row.append(0)
                    else:
                        row.append(self.distance_matrix.get_travel_time(
                            places[i].address, 
                            places[j].address, 
                            travel_mode
                        ))
                distances.append(row)
            
            # Try all starting points to find the globally optimal route
            best_route = None
            best_cost = float('inf')
            
            for start_city in range(n):
                model = cp_model.CpModel()
                
                x = {}
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            x[i, j] = model.NewBoolVar(f'x_{i}_{j}')
                
                # Each city must have exactly one incoming and one outgoing edge
                for i in range(n):
                    model.Add(sum(x[i, j] for j in range(n) if i != j) == 1)
                    model.Add(sum(x[j, i] for j in range(n) if i != j) == 1)
                
                # Subtour elimination constraints (Miller-Tucker-Zemlin formulation)
                u = {}
                for i in range(n):
                    if i != start_city:
                        u[i] = model.NewIntVar(1, n - 1, f'u_{i}')
                
                for i in range(n):
                    for j in range(n):
                        if i != j and i != start_city and j != start_city:
                            model.Add(u[i] - u[j] + n * x[i, j] <= n - 1)
                
                # Objective: minimize total distance
                objective_terms = []
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            objective_terms.append(distances[i][j] * x[i, j])
                
                model.Minimize(sum(objective_terms))
                
                solver = cp_model.CpSolver()
                status = solver.Solve(model)
                
                if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                    # Reconstruct route starting from start_city
                    route = [start_city]
                    current = start_city
                    visited = {start_city}
                    
                    while len(visited) < n:
                        for j in range(n):
                            if j != current and j not in visited and solver.Value(x[current, j]) == 1:
                                route.append(j)
                                visited.add(j)
                                current = j
                                break
                        else:
                            break
                    
                    # Calculate total cost for this route
                    total_cost = 0
                    for i in range(len(route) - 1):
                        total_cost += distances[route[i]][route[i + 1]]
                    
                    # Update best route if this one is better
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_route = route.copy()
            
            if best_route:
                return best_route
            else:
                return self._greedy_algorithm(places, travel_mode)
                
        except ImportError:
            print("OR-Tools not available, falling back to greedy algorithm")
            return self._greedy_algorithm(places, travel_mode)
    
    def _brute_force_algorithm(self, places: List, travel_mode: str) -> List[int]:
        from itertools import permutations
        
        n = len(places)
        if n > Config.MAX_BRUTE_FORCE_PLACES:
            raise ValueError(f"Brute force can only handle up to {Config.MAX_BRUTE_FORCE_PLACES} places")
        
        if n <= 1:
            return list(range(n))
        
        # Create distance matrix
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    row.append(self.distance_matrix.get_travel_time(
                        places[i].address, 
                        places[j].address, 
                        travel_mode
                    ))
            distances.append(row)
        
        best_route = None
        best_distance = float('inf')
        
        # Try all possible permutations (all starting points)
        for perm in permutations(range(n)):
            total_distance = 0
            
            # Calculate total distance for this permutation
            for i in range(len(perm) - 1):
                total_distance += distances[perm[i]][perm[i + 1]]
            
            if total_distance < best_distance:
                best_distance = total_distance
                best_route = list(perm)
        
        return best_route or list(range(n))
    
    def _dynamic_programming_algorithm(self, places: List, travel_mode: str) -> List[int]:
        """Dynamic programming solution for TSP using Held-Karp algorithm"""
        n = len(places)
        if n <= 1:
            return list(range(n))
        
        if n > 12:  # Dynamic programming becomes too memory intensive
            print(f"Dynamic programming limited to 12 places, falling back to greedy for {n} places")
            return self._greedy_algorithm(places, travel_mode)
        
        print(f"Dynamic Programming: Optimizing route for {n} places")
        
        # Create distance matrix
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    travel_time = self.distance_matrix.get_travel_time(
                        places[i].address, 
                        places[j].address, 
                        travel_mode
                    )
                    row.append(travel_time)
            distances.append(row)
        
        print(f"Distance matrix: {distances}")
        
        # Try all starting points to find the globally optimal route
        best_route = None
        best_cost = float('inf')
        
        for start_city in range(n):
            # Initialize DP table: dp[mask][pos] = minimum cost to visit all cities in mask ending at pos
            dp = {}
            
            def solve(mask, pos):
                if mask == (1 << n) - 1:  # All cities visited
                    return distances[pos][start_city]  # Return to start city
            
                state = (mask, pos)
                if state in dp:
                    return dp[state]
                
                min_cost = float('inf')
                
                # Try visiting each unvisited city
                for next_city in range(n):
                    if mask & (1 << next_city) == 0:  # City not visited
                        new_mask = mask | (1 << next_city)
                        cost = distances[pos][next_city] + solve(new_mask, next_city)
                        min_cost = min(min_cost, cost)
                
                dp[state] = min_cost
                return min_cost
            
            # Find the optimal route by backtracking
            def reconstruct_route():
                route = [start_city]
                mask = 1 << start_city  # Start city visited
                pos = start_city
                
                while len(route) < n:
                    min_cost = float('inf')
                    best_next = None
                    
                    for next_city in range(n):
                        if mask & (1 << next_city) == 0:  # City not visited
                            new_mask = mask | (1 << next_city)
                            cost = distances[pos][next_city] + solve(new_mask, next_city)
                            if cost < min_cost:
                                min_cost = cost
                                best_next = next_city
                    
                    if best_next is not None:
                        route.append(best_next)
                        mask |= (1 << best_next)
                        pos = best_next
                    else:
                        break
                
                return route
            
            # Solve and reconstruct for this starting point
            solve(1 << start_city, start_city)  # Start with only start_city visited, at position start_city
            route = reconstruct_route()
            cost = solve(1 << start_city, start_city)
            
            # Update best route if this one is better
            if cost < best_cost:
                best_cost = cost
                best_route = route.copy()
        
        print(f"Dynamic Programming: Optimal route found: {best_route} with cost: {best_cost}")
        return best_route
    
    def _optimal_algorithm(self, places: List, travel_mode: str) -> List[int]:
        """Find the truly optimal route by trying all starting points"""
        from itertools import permutations
        
        n = len(places)
        if n <= 1:
            return list(range(n))
        
        print(f"Optimal: Finding best route for {n} places (trying all starting points)")
        
        # Create distance matrix
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    travel_time = self.distance_matrix.get_travel_time(
                        places[i].address, 
                        places[j].address, 
                        travel_mode
                    )
                    row.append(travel_time)
            distances.append(row)
        
        print(f"Optimal: Distance matrix: {distances}")
        
        best_route = None
        best_time = float('inf')
        
        # Try all possible permutations (all starting points)
        for perm in permutations(range(n)):
            total_time = 0
            
            # Calculate total travel time for this permutation
            for i in range(len(perm) - 1):
                total_time += distances[perm[i]][perm[i + 1]]
            
            if total_time < best_time:
                best_time = total_time
                best_route = list(perm)
        
        print(f"Optimal: Best route found: {best_route} with {best_time} minutes")
        return best_route
    
    def _genetic_algorithm(self, places: List, travel_mode: str) -> List[int]:
        """Genetic algorithm for TSP optimization"""
        import random
        
        n = len(places)
        if n <= 1:
            return list(range(n))
        
        # Create distance matrix
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    row.append(self.distance_matrix.get_travel_time(
                        places[i].address, 
                        places[j].address, 
                        travel_mode
                    ))
            distances.append(row)
        
        def calculate_fitness(route):
            total_distance = 0
            for i in range(len(route) - 1):
                total_distance += distances[route[i]][route[i + 1]]
            return 1 / (total_distance + 1)  # Higher fitness for shorter routes
        
        def create_individual():
            # Create a random route starting from 0
            route = [0] + random.sample(range(1, n), n - 1)
            return route
        
        def crossover(parent1, parent2):
            # Order crossover
            start, end = sorted(random.sample(range(1, n), 2))
            child = [-1] * n
            child[0] = 0
            
            # Copy segment from parent1
            for i in range(start, end + 1):
                child[i] = parent1[i]
            
            # Fill remaining positions from parent2
            j = 1
            for i in range(1, n):
                if child[i] == -1:
                    while parent2[j] in child:
                        j += 1
                    child[i] = parent2[j]
                    j += 1
            
            return child
        
        def mutate(route):
            # Swap mutation
            if random.random() < 0.1:  # 10% mutation rate
                i, j = random.sample(range(1, n), 2)
                route[i], route[j] = route[j], route[i]
            return route
        
        # Initialize population
        population_size = 50
        population = [create_individual() for _ in range(population_size)]
        
        # Evolution
        generations = 100
        for generation in range(generations):
            # Calculate fitness
            fitness_scores = [(calculate_fitness(route), route) for route in population]
            fitness_scores.sort(reverse=True)
            
            # Keep best 20%
            new_population = [route for _, route in fitness_scores[:population_size // 5]]
            
            # Generate new individuals
            while len(new_population) < population_size:
                parent1 = random.choice(fitness_scores[:population_size // 2])[1]
                parent2 = random.choice(fitness_scores[:population_size // 2])[1]
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Return best route
        best_route = max(population, key=calculate_fitness)
        return best_route
    
    def _ant_colony_algorithm(self, places: List, travel_mode: str) -> List[int]:
        """Ant Colony Optimization for TSP"""
        import random
        
        n = len(places)
        if n <= 1:
            return list(range(n))
        
        # Create distance matrix
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    row.append(self.distance_matrix.get_travel_time(
                        places[i].address, 
                        places[j].address, 
                        travel_mode
                    ))
            distances.append(row)
        
        # ACO parameters
        n_ants = 20
        n_iterations = 50
        evaporation_rate = 0.1
        alpha = 1.0  # pheromone importance
        beta = 2.0   # distance importance
        
        # Initialize pheromone matrix
        pheromone = [[1.0 for _ in range(n)] for _ in range(n)]
        
        def calculate_tour_length(route):
            total = 0
            for i in range(len(route) - 1):
                total += distances[route[i]][route[i + 1]]
            return total
        
        best_route = None
        best_length = float('inf')
        
        for iteration in range(n_iterations):
            # Generate tours for all ants
            all_tours = []
            
            for ant in range(n_ants):
                # Start from place 0
                unvisited = set(range(1, n))
                current = 0
                tour = [current]
                
                while unvisited:
                    # Calculate probabilities for next city
                    probabilities = []
                    for next_city in unvisited:
                        if distances[current][next_city] > 0:
                            pheromone_value = pheromone[current][next_city]
                            distance_value = 1 / distances[current][next_city]
                            prob = (pheromone_value ** alpha) * (distance_value ** beta)
                            probabilities.append((prob, next_city))
                        else:
                            probabilities.append((0, next_city))
                    
                    if not probabilities:
                        break
                    
                    # Select next city based on probabilities
                    total_prob = sum(prob for prob, _ in probabilities)
                    if total_prob > 0:
                        r = random.random() * total_prob
                        cum_prob = 0
                        for prob, city in probabilities:
                            cum_prob += prob
                            if cum_prob >= r:
                                current = city
                                break
                    else:
                        current = random.choice(list(unvisited))
                    
                    tour.append(current)
                    unvisited.remove(current)
                
                all_tours.append(tour)
            
            # Update pheromones
            # Evaporate
            for i in range(n):
                for j in range(n):
                    pheromone[i][j] *= (1 - evaporation_rate)
            
            # Deposit pheromones
            for tour in all_tours:
                tour_length = calculate_tour_length(tour)
                if tour_length < best_length:
                    best_length = tour_length
                    best_route = tour.copy()
                
                for i in range(len(tour) - 1):
                    pheromone[tour[i]][tour[i + 1]] += 1 / tour_length
        
        return best_route or list(range(n))
    
    def _simulated_annealing_algorithm(self, places: List, travel_mode: str) -> List[int]:
        """Simulated Annealing for TSP optimization"""
        import random
        import math
        
        n = len(places)
        if n <= 1:
            return list(range(n))
        
        # Create distance matrix
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    row.append(self.distance_matrix.get_travel_time(
                        places[i].address, 
                        places[j].address, 
                        travel_mode
                    ))
            distances.append(row)
        
        def calculate_total_distance(route):
            total = 0
            for i in range(len(route) - 1):
                total += distances[route[i]][route[i + 1]]
            return total
        
        def get_neighbor(route):
            # 2-opt swap
            new_route = route.copy()
            i, j = sorted(random.sample(range(1, n), 2))
            new_route[i:j] = reversed(new_route[i:j])
            return new_route
        
        # Initial solution
        current_route = [0] + list(range(1, n))
        current_distance = calculate_total_distance(current_route)
        
        best_route = current_route.copy()
        best_distance = current_distance
        
        # Annealing parameters
        temperature = 1000
        cooling_rate = 0.95
        min_temperature = 1
        
        while temperature > min_temperature:
            for _ in range(100):
                # Generate neighbor
                neighbor_route = get_neighbor(current_route)
                neighbor_distance = calculate_total_distance(neighbor_route)
                
                # Calculate delta
                delta = neighbor_distance - current_distance
                
                # Accept or reject
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_route = neighbor_route
                    current_distance = neighbor_distance
                    
                    # Update best if necessary
                    if current_distance < best_distance:
                        best_route = current_route.copy()
                        best_distance = current_distance
            
            temperature *= cooling_rate
        
        return best_route
    
    def _preference_based_algorithm(self, places: List, travel_mode: str) -> List[int]:
        """Route optimization based on user preferences"""
        n = len(places)
        if n <= 1:
            return list(range(n))
        
        # Create preference-weighted distance matrix
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    travel_time = self.distance_matrix.get_travel_time(
                        places[i].address, 
                        places[j].address, 
                        travel_mode
                    )
                    # Weight by preference (higher preference = lower weight)
                    preference_weight = 1 / places[j].preference
                    weighted_distance = travel_time * preference_weight
                    row.append(weighted_distance)
            distances.append(row)
        
        # Use greedy algorithm with preference-weighted distances
        unvisited = set(range(n))
        route = [0]  # Start with first place
        unvisited.remove(0)
        
        while unvisited:
            current = route[-1]
            best_next = None
            best_score = float('inf')
            
            for next_place in unvisited:
                score = distances[current][next_place]
                if score < best_score:
                    best_next = next_place
                    best_score = score
            
            if best_next is not None:
                route.append(best_next)
                unvisited.remove(best_next)
            else:
                break
        
        return route
    
    def _time_window_algorithm(self, places: List, travel_mode: str) -> List[int]:
        """Route optimization considering time windows and opening hours"""
        n = len(places)
        if n <= 1:
            return list(range(n))
        
        # Create time-aware distance matrix
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    travel_time = self.distance_matrix.get_travel_time(
                        places[i].address, 
                        places[j].address, 
                        travel_mode
                    )
                    
                    # Penalize if destination has restrictive hours
                    open_minutes = TimeUtils.time_to_minutes(places[j].open_time)
                    close_minutes = TimeUtils.time_to_minutes(places[j].close_time)
                    window_size = close_minutes - open_minutes
                    
                    # Shorter time windows get higher penalties
                    time_penalty = 1 + (24 * 60 - window_size) / (24 * 60)
                    weighted_distance = travel_time * time_penalty
                    
                    row.append(weighted_distance)
            distances.append(row)
        
        # Use greedy algorithm with time-aware distances
        unvisited = set(range(n))
        route = [0]  # Start with first place
        unvisited.remove(0)
        
        while unvisited:
            current = route[-1]
            best_next = None
            best_score = float('inf')
            
            for next_place in unvisited:
                score = distances[current][next_place]
                if score < best_score:
                    best_next = next_place
                    best_score = score
            
            if best_next is not None:
                route.append(best_next)
                unvisited.remove(best_next)
            else:
                break
        
        return route
    
    def _preferred_timing_algorithm(self, places: List, travel_mode: str) -> List[int]:
        """Route optimization considering preferred timing with importance weighting"""
        n = len(places)
        if n <= 1:
            return list(range(n))
        
        # Create timing-aware distance matrix
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    travel_time = self.distance_matrix.get_travel_time(
                        places[i].address, 
                        places[j].address, 
                        travel_mode
                    )
                    
                    # Calculate timing penalty based on preferred times
                    timing_penalty = 1.0
                    if places[j].preferred_start_time and places[j].preferred_end_time:
                        # Calculate how much the timing matters
                        importance_factor = places[j].timing_importance / 5.0  # Normalize to 0-1
                        
                        # For now, we'll use a simple penalty based on importance
                        # In a full implementation, this would consider actual arrival times
                        timing_penalty = 1.0 + (importance_factor * 0.5)
                    
                    weighted_distance = travel_time * timing_penalty
                    row.append(weighted_distance)
            distances.append(row)
        
        # Use greedy algorithm with timing-aware distances
        unvisited = set(range(n))
        route = [0]  # Start with first place
        unvisited.remove(0)
        
        while unvisited:
            current = route[-1]
            best_next = None
            best_score = float('inf')
            
            for next_place in unvisited:
                score = distances[current][next_place]
                if score < best_score:
                    best_next = next_place
                    best_score = score
            
            if best_next is not None:
                route.append(best_next)
                unvisited.remove(best_next)
            else:
                break
        
        return route
    
    @staticmethod
    def greedy_best_start(distances: List[List[int]]) -> Tuple[List[int], int]:
        n = len(distances)
        best_route = None
        best_time = float('inf')
        for start in range(n):
            unvisited = set(range(n))
            route = [start]
            unvisited.remove(start)
            total_time = 0
            while unvisited:
                current = route[-1]
                next_city = min(unvisited, key=lambda j: distances[current][j])
                total_time += distances[current][next_city]
                route.append(next_city)
                unvisited.remove(next_city)
            if total_time < best_time:
                best_time = total_time
                best_route = route
        return best_route, best_time

    @staticmethod
    def dp_best_start(distances: List[List[int]]) -> Tuple[List[int], int]:
        from functools import lru_cache
        n = len(distances)
        best_route = None
        best_time = float('inf')
        for start in range(n):
            @lru_cache(maxsize=None)
            def visit(mask, pos):
                if mask == (1 << n) - 1:
                    return 0
                min_cost = float('inf')
                for nxt in range(n):
                    if not (mask & (1 << nxt)):
                        cost = distances[pos][nxt] + visit(mask | (1 << nxt), nxt)
                        if cost < min_cost:
                            min_cost = cost
                return min_cost
            # Reconstruct route
            def build_route():
                mask = 1 << start
                pos = start
                route = [start]
                for _ in range(n - 1):
                    nxt = min((j for j in range(n) if not (mask & (1 << j))), key=lambda j: distances[pos][j] + visit(mask | (1 << j), j))
                    route.append(nxt)
                    mask |= (1 << nxt)
                    pos = nxt
                return route
            total_time = visit(1 << start, start)
            route = build_route()
            if total_time < best_time:
                best_time = total_time
                best_route = route
        return best_route, best_time

    @staticmethod
    def ortools_best_start(distances: List[List[int]]) -> Tuple[List[int], int]:
        try:
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2
            n = len(distances)
            manager = pywrapcp.RoutingIndexManager(n, 1, list(range(n)))
            routing = pywrapcp.RoutingModel(manager)
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return distances[from_node][to_node]
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            search_parameters.time_limit.seconds = 10
            assignment = routing.SolveWithParameters(search_parameters)
            if assignment:
                index = routing.Start(0)
                route = []
                total_time = 0
                while not routing.IsEnd(index):
                    route.append(manager.IndexToNode(index))
                    next_index = assignment.Value(routing.NextVar(index))
                    total_time += distances[manager.IndexToNode(index)][manager.IndexToNode(next_index)]
                    index = next_index
                return route, total_time
            else:
                return list(range(n)), float('inf')
        except ImportError:
            print("OR-Tools not available.")
            return list(range(len(distances))), float('inf')

    @staticmethod
    def brute_force_best_start(distances: List[List[int]]) -> Tuple[List[int], int]:
        from itertools import permutations
        n = len(distances)
        best_route = None
        best_time = float('inf')
        for perm in permutations(range(n)):
            total_time = sum(distances[perm[i]][perm[i+1]] for i in range(n-1))
            if total_time < best_time:
                best_time = total_time
                best_route = list(perm)
        return best_route, best_time

    def _genetic_annealing_algorithm(self, places: List, travel_mode: str) -> List[int]:
        """Genetic Annealing: Combines genetic algorithm with simulated annealing for hybrid optimization"""
        import random
        import math
        
        n = len(places)
        if n <= 1:
            return list(range(n))
        
        print(f"Genetic Annealing: Optimizing route for {n} places")
        
        # Create distance matrix
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    row.append(self.distance_matrix.get_travel_time(
                        places[i].address, 
                        places[j].address, 
                        travel_mode
                    ))
            distances.append(row)
        
        def calculate_fitness(route):
            """Calculate fitness (inverse of total distance)"""
            total_distance = 0
            for i in range(len(route) - 1):
                total_distance += distances[route[i]][route[i + 1]]
            return 1.0 / (total_distance + 1)  # Avoid division by zero
        
        def create_individual():
            """Create a random route starting from 0"""
            route = [0] + random.sample(range(1, n), n - 1)
            return route
        
        def crossover(parent1, parent2):
            """Order crossover (OX)"""
            if n <= 2:
                return parent1.copy()
            
            start, end = sorted(random.sample(range(1, n), 2))
            child = [-1] * n
            child[0] = 0  # Always start from 0
            
            # Copy segment from parent1
            for i in range(start, end + 1):
                child[i] = parent1[i]
            
            # Fill remaining positions from parent2
            j = 1
            for i in range(1, n):
                if child[i] == -1:
                    while parent2[j] in child:
                        j += 1
                    child[i] = parent2[j]
                    j += 1
            
            return child
        
        def mutate(route, temperature):
            """Temperature-dependent mutation"""
            mutated_route = route.copy()
            
            # Higher temperature = higher mutation rate
            mutation_rate = 0.1 + (temperature / 1000.0) * 0.3  # 10% to 40%
            
            if random.random() < mutation_rate:
                # 2-opt swap mutation
                i, j = sorted(random.sample(range(1, n), 2))
                mutated_route[i:j] = reversed(mutated_route[i:j])
            
            return mutated_route
        
        def simulated_annealing_step(route, temperature):
            """Single simulated annealing step"""
            neighbor = mutate(route, temperature)
            
            current_fitness = calculate_fitness(route)
            neighbor_fitness = calculate_fitness(neighbor)
            
            # Calculate acceptance probability
            delta_fitness = neighbor_fitness - current_fitness
            if delta_fitness > 0:
                return neighbor  # Always accept better solutions
            else:
                # Accept worse solutions with probability based on temperature
                acceptance_prob = math.exp(delta_fitness / temperature)
                if random.random() < acceptance_prob:
                    return neighbor
                else:
                    return route
        
        # Genetic Annealing parameters
        population_size = 30
        generations = 50
        initial_temperature = 1000
        cooling_rate = 0.95
        min_temperature = 1
        
        # Initialize population
        population = [create_individual() for _ in range(population_size)]
        
        # Track best solution
        best_route = None
        best_fitness = 0
        
        # Evolution with annealing
        temperature = initial_temperature
        
        for generation in range(generations):
            # Calculate fitness for all individuals
            fitness_scores = [(calculate_fitness(route), route) for route in population]
            fitness_scores.sort(reverse=True)  # Sort by fitness (descending)
            
            # Update best solution
            if fitness_scores[0][0] > best_fitness:
                best_fitness = fitness_scores[0][0]
                best_route = fitness_scores[0][1].copy()
            
            # Create new population
            new_population = []
            
            # Elitism: Keep top 20% of individuals
            elite_count = max(1, population_size // 5)
            new_population.extend([route for _, route in fitness_scores[:elite_count]])
            
            # Generate rest of population through crossover and annealing
            while len(new_population) < population_size:
                # Tournament selection
                tournament_size = 3
                parent1 = min(random.sample(population, tournament_size), 
                            key=lambda x: calculate_fitness(x))
                parent2 = min(random.sample(population, tournament_size), 
                            key=lambda x: calculate_fitness(x))
                
                # Crossover
                child = crossover(parent1, parent2)
                
                # Apply simulated annealing to child
                child = simulated_annealing_step(child, temperature)
                
                new_population.append(child)
            
            population = new_population
            
            # Cool down temperature
            temperature *= cooling_rate
            temperature = max(temperature, min_temperature)
            
            # Print progress every 10 generations
            if generation % 10 == 0:
                current_best = max(calculate_fitness(route) for route in population)
                print(f"Generation {generation}: Best fitness = {current_best:.6f}, Temperature = {temperature:.2f}")
        
        print(f"Genetic Annealing: Best route found: {best_route} with fitness: {best_fitness:.6f}")
        return best_route or list(range(n))