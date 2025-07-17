from utils import DistanceMatrix, RouteOptimizer
from models import Place
from itertools import permutations

def test_all_algorithms():
    # Create the places from your example
    places = [
        Place(
            place_name="Statue of Liberty",
            address="New York, NY 10004, USA",
            open_time="09:00",
            close_time="17:00",
            visit_duration=60,
            preference=3
        ),
        Place(
            place_name="Coney Island",
            address="Coney Island, Brooklyn, NY, USA",
            open_time="06:00",
            close_time="22:00",
            visit_duration=60,
            preference=3
        ),
        Place(
            place_name="Bronx Zoo",
            address="2300 Southern Blvd, Bronx, NY 10460, USA",
            open_time="09:00",
            close_time="17:00",
            visit_duration=90,
            preference=3
        ),
        Place(
            place_name="Times Square",
            address="Manhattan, NY 10036, USA",
            open_time="00:00",
            close_time="23:59",
            visit_duration=60,
            preference=3
        )
    ]
    
    distance_matrix = DistanceMatrix()
    
    print("=" * 80)
    print("FINDING TRULY OPTIMAL ROUTE (ANY STARTING POINT)")
    print("=" * 80)
    print(f"Places: {[p.place_name for p in places]}")
    print()
    
    # Test all possible permutations (true optimal solution)
    print("Testing ALL POSSIBLE PERMUTATIONS...")
    print()
    
    best_route = None
    best_time = float('inf')
    all_routes = []
    
    # Try all possible starting points and permutations
    for perm in permutations(range(len(places))):
        total_time = 0
        route_details = []
        
        for i in range(len(perm) - 1):
            origin = places[perm[i]].address
            destination = places[perm[i + 1]].address
            travel_time = distance_matrix.get_travel_time(origin, destination, "driving")
            total_time += travel_time
            route_details.append(f"{places[perm[i]].place_name} → {places[perm[i + 1]].place_name}: {travel_time} min")
        
        route_names = [places[i].place_name for i in perm]
        all_routes.append({
            'route': perm,
            'total_time': total_time,
            'route_details': route_details,
            'route_names': route_names
        })
        
        if total_time < best_time:
            best_time = total_time
            best_route = perm
    
    # Sort all routes by time
    all_routes.sort(key=lambda x: x['total_time'])
    
    print("TOP 10 OPTIMAL ROUTES:")
    print("=" * 80)
    
    for i, route_info in enumerate(all_routes[:10], 1):
        print(f"{i}. {route_info['total_time']} minutes")
        print(f"   Route: {' → '.join(route_info['route_names'])}")
        for detail in route_info['route_details']:
            print(f"   {detail}")
        print()
    
    print("=" * 80)
    print(f"🏆 ABSOLUTE OPTIMAL ROUTE: {best_time} minutes")
    print(f"Route: {' → '.join([places[i].place_name for i in best_route])}")
    print("=" * 80)
    
    # Now test our algorithms with the constraint of starting from first place
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON (CONSTRAINED TO START FROM FIRST PLACE)")
    print("=" * 80)
    
    route_optimizer = RouteOptimizer(distance_matrix)
    algorithms = ["greedy", "ortools", "brute", "dynamic_programming"]
    
    results = []
    
    for algorithm in algorithms:
        try:
            print(f"Testing {algorithm.upper()} algorithm...")
            route = route_optimizer.optimize_route(places, algorithm, "driving")
            
            # Calculate total travel time for this route
            total_time = 0
            route_details = []
            
            for i in range(len(route) - 1):
                origin = places[route[i]].address
                destination = places[route[i + 1]].address
                travel_time = distance_matrix.get_travel_time(origin, destination, "driving")
                total_time += travel_time
                route_details.append(f"{places[route[i]].place_name} → {places[route[i + 1]].place_name}: {travel_time} min")
            
            results.append({
                'algorithm': algorithm,
                'route': route,
                'total_time': total_time,
                'route_details': route_details,
                'route_names': [places[i].place_name for i in route]
            })
            
            print(f"  Route: {' → '.join([places[i].place_name for i in route])}")
            print(f"  Total travel time: {total_time} minutes")
            print()
            
        except Exception as e:
            print(f"  Error with {algorithm}: {e}")
            print()
    
    # Find the best constrained route
    if results:
        best_constrained = min(results, key=lambda x: x['total_time'])
        
        print("=" * 80)
        print("CONSTRAINED ALGORITHM RESULTS")
        print("=" * 80)
        
        # Sort by total time
        results.sort(key=lambda x: x['total_time'])
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['algorithm'].upper()}: {result['total_time']} minutes")
            print(f"   Route: {' → '.join(result['route_names'])}")
            for detail in result['route_details']:
                print(f"   {detail}")
            print()
        
        print("=" * 80)
        print(f"BEST CONSTRAINED: {best_constrained['algorithm'].upper()} with {best_constrained['total_time']} minutes")
        print(f"Route: {' → '.join(best_constrained['route_names'])}")
        print()
        print(f"OPTIMAL vs CONSTRAINED: {best_time} vs {best_constrained['total_time']} minutes")
        print(f"Difference: {best_constrained['total_time'] - best_time} minutes")
        print("=" * 80)

if __name__ == "__main__":
    test_all_algorithms() 