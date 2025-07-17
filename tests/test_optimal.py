from utils import DistanceMatrix, RouteOptimizer
from models import Place

def test_optimal_algorithm():
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
    route_optimizer = RouteOptimizer(distance_matrix)
    
    print("=" * 60)
    print("TESTING NEW OPTIMAL ALGORITHM")
    print("=" * 60)
    
    # Test the optimal algorithm
    route = route_optimizer.optimize_route(places, "optimal", "driving")
    
    # Calculate total travel time
    total_time = 0
    route_details = []
    
    for i in range(len(route) - 1):
        origin = places[route[i]].address
        destination = places[route[i + 1]].address
        travel_time = distance_matrix.get_travel_time(origin, destination, "driving")
        total_time += travel_time
        route_details.append(f"{places[route[i]].place_name} → {places[route[i + 1]].place_name}: {travel_time} min")
    
    print(f"\nOptimal Route: {' → '.join([places[i].place_name for i in route])}")
    print(f"Total Travel Time: {total_time} minutes")
    print("\nRoute Details:")
    for detail in route_details:
        print(f"  {detail}")
    
    print("\n" + "=" * 60)
    print("EXPECTED RESULT: Bronx Zoo → Times Square → Statue of Liberty → Coney Island")
    print("EXPECTED TIME: 73 minutes")
    print("=" * 60)

if __name__ == "__main__":
    test_optimal_algorithm() 