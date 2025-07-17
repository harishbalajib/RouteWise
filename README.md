# Smart US Travel Planner

A travel planning application built with FastAPI and Python, designed to optimize travel routes using advanced algorithms. The application considers user preferences, time windows, and preferred visit times to generate efficient itineraries.

## Features

### Core Functionality
- **Route Optimization**: Multiple algorithms including greedy, OR-Tools, genetic, ant colony, and more
- **Time Window Constraints**: Respects opening/closing hours of destinations
- **Daily Itinerary Splitting**: Automatically splits routes across multiple days
- **Interactive Route Maps**: Google Static Maps with labeled markers and legends
- **Travel Time Calculation**: Real-time travel times using Google Distance Matrix API

### Enhanced Features
- **Preferred Timing**: Specify preferred visit times for each location
- **Automatic Opening Hours**: Fetches and populates opening hours from Google Places API
- **Place Details**: Comprehensive information including ratings, business status, and photos
- **Address Autocomplete**: Google Places Autocomplete for accurate address input
- **Caching**: Reduces API calls and improves performance

## Supported Algorithms

- **Greedy**: Fast nearest-neighbor approach
- **OR-Tools**: Optimal solution using Google's CP-SAT solver
- **Brute Force**: Exhaustive search (limited to 7 places)
- **Genetic**: Evolutionary algorithm
- **Ant Colony**: Swarm intelligence
- **Simulated Annealing**: Probabilistic optimization
- **Preference-Based**: Considers user preference ratings
- **Time Window**: Optimizes for opening hours
- **Preferred Timing**: Prioritizes preferred visit times

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RouteWise
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Google Maps API key
```

### Google Maps API Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable these APIs:
   - Distance Matrix API
   - Places API
   - Static Maps API
4. Create an API key and add it to your `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```
5. Ensure billing is enabled for your project

### Usage

1. Start the server:
```bash
python main.py
```
2. Open your browser at `http://localhost:8000`
3. Fill in the travel planner form:
   - Set daily schedule, number of days, travel mode, algorithm, and destinations

#### Adding Destinations
- **Place Name**: Name of the location
- **Address**: Use autocomplete for accuracy
- **Opening/Closing Times**: Specify hours
- **Visit Duration**: Minutes to spend at each place
- **Preference**: Importance (1-5 scale)
- **Preferred Start/End Time**: (Optional)
- **Timing Importance**: (1-5, optional)

## API Endpoints

- `GET /` : Main interface
- `POST /plan` : Submit travel planning request
- `GET /api/places/search` : Search for places
- `GET /api/places/details/{place_id}` : Get place details
- `GET /health` : Health check

## Testing

All test files are now located in the `tests` directory for better organization.

To run the test suite:
```bash
cd tests
python run_tests.py
```
See `tests/README.md` for more details and individual test usage.

## Configuration

Key options in `config.py`:
- `CACHE_DURATION`: API response cache duration
- `MAP_WIDTH`, `MAP_HEIGHT`, `MAP_ZOOM`: Map display settings
- Google API endpoints

## Architecture

- **Backend**: FastAPI, Pydantic models
- **Frontend**: Minimal HTML/CSS/JavaScript (Jinja2 templates)
- **Algorithms**: Custom and OR-Tools
- **APIs**: Google Maps Distance Matrix, Places, Static Maps
- **Caching**: In-memory for API responses

## Performance

- **Caching**: Reduces API calls
- **Algorithm Selection**: Choose based on needs
- **Parallel Processing**: Efficient API requests
- **Error Handling**: Comprehensive and user-friendly

## Troubleshooting

- **API Key Errors**: Ensure the key is in `.env`, valid, and billing is enabled
- **REQUEST_DENIED**: Check API enablement and quotas
- **Slow Responses**: Check connection and API status; use the greedy algorithm for speed
- **Place Search Issues**: Ensure Places API is enabled and queries are valid

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

Licensed under the MIT License. See the LICENSE file for details. 