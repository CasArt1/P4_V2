"""
API Test Client
Example script showing how to use the NVDA Trading Signal API
"""

import requests
import json
import numpy as np
import pandas as pd

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_model_info():
    """Test the model info endpoint"""
    print("\n" + "="*60)
    print("Testing Model Info Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_prediction_with_real_data():
    """Test prediction with real data from test set"""
    print("\n" + "="*60)
    print("Testing Prediction with Real Data")
    print("="*60)
    
    try:
        # Load test data
        test_df = pd.read_csv('data/NVDA_test.csv', index_col=0, parse_dates=True)
        feature_cols = [col for col in test_df.columns if col.endswith('_norm')]
        
        # Get last 10 rows as a sequence
        sequence = test_df[feature_cols].tail(10).values.tolist()
        
        # Create request
        payload = {
            "sequences": sequence
        }
        
        # Make API call
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n📊 Prediction Results:")
            print(f"  Signal: {result['signal']} ({result['signal_name']})")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Probabilities:")
            for signal, prob in result['probabilities'].items():
                print(f"    {signal}: {prob:.2%}")
            print(f"  Timestamp: {result['timestamp']}")
            print(f"  Model: {result['model_name']}")
            
            # Show actual target if available
            if 'target' in test_df.columns:
                actual_target = test_df['target'].iloc[-1]
                actual_signal_map = {-1: "SHORT", 0: "HOLD", 1: "LONG"}
                print(f"\n  Actual Signal: {actual_target} ({actual_signal_map[actual_target]})")
                print(f"  Prediction Correct: {'✅ Yes' if result['signal'] == actual_target else '❌ No'}")
        else:
            print(f"Error: {response.json()}")
        
        return response.status_code == 200
        
    except FileNotFoundError:
        print("❌ Test data not found. Make sure you've run Phase 2 (feature engineering).")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_prediction_with_dummy_data():
    """Test prediction with dummy data"""
    print("\n" + "="*60)
    print("Testing Prediction with Dummy Data")
    print("="*60)
    
    # Create dummy sequence (10 timesteps x 28 features)
    # Using random normalized values between -1 and 1
    np.random.seed(42)
    sequence = np.random.randn(10, 28).tolist()
    
    payload = {
        "sequences": sequence
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 Prediction Results:")
        print(f"  Signal: {result['signal']} ({result['signal_name']})")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Model: {result['model_name']}")
    else:
        print(f"Error: {response.json()}")
    
    return response.status_code == 200

def test_single_prediction():
    """Test single feature vector prediction"""
    print("\n" + "="*60)
    print("Testing Single Feature Prediction")
    print("="*60)
    
    # Create dummy single feature vector (28 features)
    np.random.seed(42)
    features = np.random.randn(28).tolist()
    
    payload = {
        "features": features
    }
    
    response = requests.post(f"{BASE_URL}/predict/single", json=payload)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 Prediction Results:")
        print(f"  Signal: {result['signal']} ({result['signal_name']})")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Note: This endpoint uses simplified sequence generation")
    else:
        print(f"Error: {response.json()}")
    
    return response.status_code == 200

def test_error_handling():
    """Test API error handling"""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)
    
    # Test 1: Wrong number of features
    print("\nTest 1: Wrong number of features")
    payload = {
        "sequences": [[0.5] * 20 for _ in range(10)]  # Only 20 features instead of 28
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code} (expected 400)")
    print(f"Error Message: {response.json().get('detail', 'No error message')}")
    
    # Test 2: Wrong number of timesteps
    print("\nTest 2: Wrong number of timesteps")
    payload = {
        "sequences": [[0.5] * 28 for _ in range(5)]  # Only 5 timesteps instead of 10
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code} (expected 400)")
    print(f"Error Message: {response.json().get('detail', 'No error message')}")
    
    print("\n✅ Error handling tests complete")

def run_all_tests():
    """Run all API tests"""
    print("="*60)
    print("NVDA TRADING SIGNAL API - TEST SUITE")
    print("="*60)
    print("\nMake sure the API is running on http://localhost:8000")
    print("Start it with: python api/main.py")
    
    input("\nPress Enter to start tests...")
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Prediction with Real Data", test_prediction_with_real_data),
        ("Prediction with Dummy Data", test_prediction_with_dummy_data),
        ("Single Feature Prediction", test_single_prediction),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Connection Error: Cannot connect to {BASE_URL}")
            print("Make sure the API server is running!")
            break
        except Exception as e:
            print(f"\n❌ Test failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Test error handling separately
    try:
        test_error_handling()
    except:
        pass
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()