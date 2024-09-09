import pytest
import requests


@pytest.fixture
def base_url():
    return 'http://localhost:3000'


def test_predict_endpoint(base_url):
    payload = {
        'Height': 68,
        'Weight': 66,
        'Chest': 36,
        'Waist': 28,
        'Hips': 36
    }

    try:
        response = requests.post(f'{base_url}/predict', json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        assert response.status_code == 200

        response_json = response.json()
        assert 'predicted_size' in response_json
        assert isinstance(response_json['predicted_size'], (int, float))

    except requests.RequestException as e:
        pytest.fail(f"Request failed: {e}")

    except ValueError as e:
        pytest.fail(f"Error parsing JSON response: {e}")
