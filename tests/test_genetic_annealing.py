from utils import DistanceMatrix, RouteOptimizer
from models import Place

def test_genetic_annealing():
    # Create test places
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
            place_name="Times Square",
            address="Manhattan, NY 10036, USA",
            open_time="00:00",
            close_time="23:59",
            visit_duration=60,
            preference=3
        ),
        Place(
            place_name="Central Park",
            address="Central Park, New York, NY, USA",
            open_time="06:00",
            close_time="22:00",
            visit_duration=90,
            preference=3
        ),
        Place(
            place_name="Brooklyn Bridge",
            address="Brooklyn Bridge, New York, NY, USA",
            open_time="00:00",
            close_time="23:59",
            visit_duration=45,
            preference=3
        ),
        Place(
            place_name="Empire State Building",
            address="20 W 34th St, New York, NY 10001, USA",
            open_time="08:00",
            close_time="23:00",
            visit_duration=75,
            preference=3
        )
    ]
    
    distance_matrix = DistanceMatrix()
    route_optimizer = RouteOptimizer(distance_matrix)
    
    print("=" * 80)
    print("TESTING GENETIC ANNEALING ALGORITHM")
    print("=" * 80)
    print(f"Places: {[p.place_name for p in places]}")
    print()
    
    # Test Genetic Annealing algorithm
    print("Testing Genetic Annealing Algorithm...")
    try:
        genetic_annealing_route = route_optimizer.optimize_route(places, 'genetic_annealing', 'driving')
        print(f"Genetic Annealing route: {genetic_annealing_route}")
        print(f"Genetic Annealing route names: {[places[i].place_name for i in genetic_annealing_route]}")
        
        # Calculate total travel time
        total_time = 0
        for i in range(len(genetic_annealing_route) - 1):
            travel_time = distance_matrix.get_travel_time(
                places[genetic_annealing_route[i]].address,
                places[genetic_annealing_route[i + 1]].address,
                'driving'
            )
            total_time += travel_time
            print(f"  {places[genetic_annealing_route[i]].place_name} → {places[genetic_annealing_route[i + 1]].place_name}: {travel_time} min")
        
        print(f"Genetic Annealing total travel time: {total_time} minutes")
        print()
        
    except Exception as e:
        print(f"Error with Genetic Annealing: {e}")
        print()
    
    # Compare with existing algorithms
    print("=" * 80)
    print("COMPARISON WITH EXISTING ALGORITHMS")
    print("=" * 80)
    
    algorithms = ['greedy', 'genetic_annealing']
    results = {}
    
    for algorithm in algorithms:
        try:
            route = route_optimizer.optimize_route(places, algorithm, 'driving')
            total_time = 0
            for i in range(len(route) - 1):
                total_time += distance_matrix.get_travel_time(
                    places[route[i]].address,
                    places[route[i + 1]].address,
                    'driving'
                )
            results[algorithm] = {
                'route': route,
                'total_time': total_time,
                'route_names': [places[i].place_name for i in route]
            }
        except Exception as e:
            results[algorithm] = {'error': str(e)}
    
    # Sort by total time
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if 'error' not in v],
        key=lambda x: x[1]['total_time']
    )
    
    print("Algorithm Performance Comparison:")
    for i, (algorithm, result) in enumerate(sorted_results, 1):
        print(f"{i}. {algorithm.upper()}: {result['total_time']} minutes")
        print(f"   Route: {' → '.join(result['route_names'])}")
        print()
    
    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_genetic_annealing() 