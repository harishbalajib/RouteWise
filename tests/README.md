# Test Suite - Smart US Travel Planner

This directory contains the test files for the Smart US Travel Planner application.

## Test Files

- **test_genetic_annealing.py** - Tests the genetic annealing algorithm
- **test_plan_endpoint.py** - Tests the /plan API endpoint
- **test_optimal.py** - Tests optimal route calculations
- **test_algorithms.py** - Tests various route optimization algorithms
- **run_tests.py** - Test runner script

## Running Tests

To run all tests:
```bash
cd tests
python run_tests.py
```

To run an individual test file:
```bash
cd tests
python test_algorithms.py
```

## Prerequisites

- Ensure your `.env` file is configured with a valid Google API key if required by the tests.
- Install all dependencies:
```bash
pip install -r ../requirements.txt
```

## Notes

- All test files are now organized in this directory for clarity and maintainability.
- For more information on each test, see the comments at the top of each file.

## Test Categories

### 1. Mock Data Tests (`test_mock_data.py`)
- Tests fallback behavior when Google API is unavailable
- Verifies mock travel times and place data
- Ensures application works offline

### 2. Algorithm Tests (`test_all_algorithms.py`)
- Tests all 9 optimization algorithms:
  - Greedy
  - OR-Tools
  - Brute Force
  - Genetic Algorithm
  - Ant Colony Optimization
  - Simulated Annealing
  - Preference-Based
  - Time Window Aware
  - Preferred Timing
- Verifies algorithm outputs and performance

### 3. API Integration Tests (`test_api_integration.py`)
- Tests Google Distance Matrix API
- Tests Google Static Maps API
- Verifies error handling and fallbacks
- Tests caching functionality

### 4. Place Search Tests (`test_place_search.py`)
- Tests Google Places Autocomplete API
- Tests place details retrieval
- Verifies address formatting and place information

### 5. Preferred Timing Tests (`test_preferred_timing.py`)
- Tests the new preferred timing system
- Verifies timing importance weighting
- Tests form validation for timing fields
- Tests place details with opening hours

### 6. API Key Tests (`test_api_key.py`)
- Tests API key configuration
- Verifies all Google APIs are accessible
- Provides detailed error messages for troubleshooting

### 7. Debug Tests (`debug_api.py`)
- Comprehensive API debugging tool
- Tests each API individually with detailed output
- Provides troubleshooting information
- Shows API response details

## Test Output

### Successful Test
```
✓ Test completed successfully
[Test output details]
```

### Failed Test
```
✗ Test failed
Error: [Error details]
Output: [Test output]
```

## Troubleshooting

### Common Issues

1. **"No API key found"**
   - Ensure `.env` file exists in project root
   - Verify API key is correctly formatted

2. **"REQUEST_DENIED" errors**
   - Check that required APIs are enabled in Google Cloud Console
   - Verify billing is enabled
   - Check API key restrictions

3. **"Legacy API" errors**
   - Enable legacy APIs or upgrade to newer APIs
   - See `GOOGLE_API_SETUP.md` for detailed instructions

4. **Import errors**
   - Ensure you're running tests from the `tests/` directory
   - Check that all dependencies are installed

### Getting Help

1. **Check API Setup**: See `../GOOGLE_API_SETUP.md`
2. **Run Debug Tool**: `python debug_api.py`
3. **Check Logs**: Look for detailed error messages in test output

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```bash
# Run all tests and exit with error code on failure
cd tests && python run_tests.py
```

The test runner returns:
- `0` (success) if all tests pass
- `1` (failure) if any test fails 