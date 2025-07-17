import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_plan_endpoint_returns_results():
    # Prepare form data for at least two places
    form_data = {
        'start_time': '09:00',
        'end_time': '18:00',
        'num_days': '1',
        'travel_mode': 'driving',
        'place_name': ['Statue of Liberty', 'Central Park'],
        'address': ['New York, NY 10004', 'New York, NY 10024'],
        'open_time': ['09:00', '06:00'],
        'close_time': ['17:00', '22:00'],
        'visit_duration': ['60', '90'],
        'preference': ['5', '4'],
        'preferred_start_time': ['', ''],
        'preferred_end_time': ['', ''],
        'timing_importance': ['1', '1'],
    }
    response = client.post('/plan', data=form_data)
    assert response.status_code == 200
    html = response.text
    # Check that at least one algorithm name appears
    assert 'Greedy' in html or 'OR-Tools' in html or 'Dynamic Programming' in html or 'Brute Force' in html
    # Check that both place names appear
    assert 'Statue of Liberty' in html
    assert 'Central Park' in html
    # Check that distance column is present
    assert 'Total Travel Distance (km)' in html
    # Check that at least one distance value is rendered (should be a number with decimal)
    import re
    assert re.search(r'<td>\d+\.\d+</td>', html) 