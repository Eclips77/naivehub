"""
NaiveHub Client Test Suite
=========================
Comprehensive testing for the Streamlit client interface.
"""

import unittest
import requests
import json
import time
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestNaiveHubClient(unittest.TestCase):
    """Test suite for NaiveHub client functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.storage_url = "http://localhost:8002"
        cls.trainer_url = "http://localhost:8001"
        cls.predictor_url = "http://localhost:8000"
        cls.test_dataset_id = f"test_dataset_{datetime.now().strftime('%H%M%S')}"
        cls.test_model_name = f"test_model_{datetime.now().strftime('%H%M%S')}"
    
    def test_01_server_connectivity(self):
        """Test server connectivity and health endpoints."""
        print("\n🔍 Testing server connectivity...")
        
        servers = {
            "Storage": self.storage_url,
            "Training": self.trainer_url,
            "Prediction": self.predictor_url
        }
        
        for name, url in servers.items():
            with self.subTest(server=name):
                try:
                    response = requests.get(f"{url}/health", timeout=10)
                    self.assertEqual(response.status_code, 200, f"{name} server is not responding")
                    health_data = response.json()
                    self.assertIn('status', health_data, f"{name} health response missing status")
                    print(f"  ✅ {name} server: Online")
                except Exception as e:
                    self.fail(f"Failed to connect to {name} server: {e}")
    
    def test_02_data_loading(self):
        """Test data loading from files."""
        print("\n📊 Testing data loading...")
        
        # Get available files
        try:
            response = requests.get(f"{self.storage_url}/files")
            self.assertEqual(response.status_code, 200)
            files_data = response.json()
            
            if files_data['count'] > 0:
                test_file = files_data['available_files'][0]['file_name']
                print(f"  Using test file: {test_file}")
                
                # Load data
                load_data = {
                    "file_name": test_file,
                    "dataset_id": self.test_dataset_id
                }
                response = requests.post(f"{self.storage_url}/data/load", json=load_data)
                self.assertEqual(response.status_code, 200)
                result = response.json()
                
                self.assertIn('dataset_id', result)
                self.assertIn('shape', result)
                self.assertIn('columns', result)
                print(f"  ✅ Data loaded: {result['shape']}")
                
                # Store columns for later tests
                self.test_columns = result['columns']
                self.test_target_column = result['columns'][-1]  # Use last column as target
            else:
                self.skipTest("No test files available")
                
        except Exception as e:
            self.fail(f"Data loading test failed: {e}")
    
    def test_03_data_preparation(self):
        """Test data preparation for training."""
        print("\n🔧 Testing data preparation...")
        
        try:
            prep_data = {
                "dataset_id": self.test_dataset_id,
                "target_column": self.test_target_column,
                "train_size": 0.7
            }
            response = requests.post(f"{self.storage_url}/data/prepare", json=prep_data)
            self.assertEqual(response.status_code, 200)
            result = response.json()
            
            self.assertIn('dataset_id', result)
            self.assertIn('train_shape', result)
            self.assertIn('test_shape', result)
            self.assertIn('target_column', result)
            
            # Update dataset ID to prepared version
            self.test_dataset_id = result['dataset_id']
            print(f"  ✅ Data prepared: Train {result['train_shape']}, Test {result['test_shape']}")
            
        except Exception as e:
            self.fail(f"Data preparation test failed: {e}")
    
    def test_04_model_training(self):
        """Test model training."""
        print("\n🎓 Testing model training...")
        
        try:
            train_data = {
                "dataset_id": self.test_dataset_id,
                "target_column": self.test_target_column,
                "model_name": self.test_model_name,
                "metadata": {
                    "test": True,
                    "created_at": datetime.now().isoformat(),
                    "description": "Test model for client validation"
                }
            }
            response = requests.post(f"{self.trainer_url}/train", json=train_data)
            self.assertEqual(response.status_code, 200)
            result = response.json()
            
            self.assertIn('model_name', result)
            self.assertIn('classes', result)
            self.assertIn('features', result)
            self.assertIn('training_time_seconds', result)
            
            # Store for later tests
            self.test_features = result['features']
            self.test_classes = result['classes']
            print(f"  ✅ Model trained: {len(result['classes'])} classes, {len(result['features'])} features")
            
        except Exception as e:
            self.fail(f"Model training test failed: {e}")
    
    def test_05_model_evaluation(self):
        """Test model evaluation."""
        print("\n📊 Testing model evaluation...")
        
        try:
            eval_data = {"model_name": self.test_model_name}
            response = requests.post(f"{self.trainer_url}/evaluate", json=eval_data)
            self.assertEqual(response.status_code, 200)
            result = response.json()
            
            self.assertIn('accuracy', result)
            self.assertIn('accuracy_percentage', result)
            self.assertIn('model_name', result)
            
            accuracy = result['accuracy']
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)
            print(f"  ✅ Model evaluated: {result['accuracy_percentage']} accuracy")
            
        except Exception as e:
            self.fail(f"Model evaluation test failed: {e}")
    
    def test_06_model_loading_to_predictor(self):
        """Test loading model to prediction server."""
        print("\n📥 Testing model loading to predictor...")
        
        try:
            load_data = {"model_name": self.test_model_name}
            response = requests.post(f"{self.predictor_url}/load_model", json=load_data)
            self.assertEqual(response.status_code, 200)
            result = response.json()
            
            self.assertIn('message', result)
            self.assertIn('model_name', result)
            print(f"  ✅ Model loaded to prediction server: {result['model_name']}")
            
        except Exception as e:
            self.fail(f"Model loading test failed: {e}")
    
    def test_07_predictions(self):
        """Test making predictions."""
        print("\n🔮 Testing predictions...")
        
        try:
            # Create a test record with sample values
            test_record = {}
            for feature in self.test_features:
                if feature != self.test_target_column:
                    # Use placeholder values for testing
                    test_record[feature] = "test_value"
            
            pred_data = {
                "model_name": self.test_model_name,
                "record": test_record
            }
            response = requests.post(f"{self.predictor_url}/predict", json=pred_data)
            self.assertEqual(response.status_code, 200)
            result = response.json()
            
            self.assertIn('prediction', result)
            self.assertIn('model_name', result)
            self.assertIn('confidence', result)
            
            prediction = result['prediction']
            self.assertIn(prediction, self.test_classes)
            print(f"  ✅ Prediction made: {result['prediction']} (confidence: {result.get('confidence', 'N/A')})")
            
        except Exception as e:
            self.fail(f"Prediction test failed: {e}")
    
    def test_08_cache_management(self):
        """Test cache status and management."""
        print("\n🗂️ Testing cache management...")
        
        try:
            # Get cache status
            response = requests.get(f"{self.predictor_url}/cache")
            self.assertEqual(response.status_code, 200)
            cache_status = response.json()
            
            self.assertIn('cache_size', cache_status)
            self.assertIn('max_cache_size', cache_status)
            self.assertIn('cached_models', cache_status)
            
            # Verify our test model is in cache
            self.assertIn(self.test_model_name, cache_status['cached_models'])
            print(f"  ✅ Cache status: {cache_status['cache_size']}/{cache_status['max_cache_size']} models")
            
        except Exception as e:
            self.fail(f"Cache management test failed: {e}")
    
    def test_09_data_retrieval(self):
        """Test data and model retrieval endpoints."""
        print("\n📋 Testing data retrieval...")
        
        try:
            # Get datasets
            response = requests.get(f"{self.storage_url}/data")
            self.assertEqual(response.status_code, 200)
            datasets = response.json()
            
            self.assertIn('available_datasets', datasets)
            self.assertIn('count', datasets)
            print(f"  ✅ Datasets retrieved: {datasets['count']} available")
            
            # Get models
            response = requests.get(f"{self.storage_url}/models")
            self.assertEqual(response.status_code, 200)
            models = response.json()
            
            self.assertIn('available_models', models)
            self.assertIn('count', models)
            print(f"  ✅ Models retrieved: {models['count']} available")
            
        except Exception as e:
            self.fail(f"Data retrieval test failed: {e}")
    
    def test_10_cleanup(self):
        """Test cleanup operations."""
        print("\n🧹 Testing cleanup...")
        
        try:
            # Unload model from predictor
            response = requests.post(f"{self.predictor_url}/unload_model?model_name={self.test_model_name}")
            self.assertEqual(response.status_code, 200)
            result = response.json()
            
            self.assertIn('message', result)
            print(f"  ✅ Model unloaded from cache: {result['message']}")
            
            # Verify cache is updated
            response = requests.get(f"{self.predictor_url}/cache")
            cache_status = response.json()
            self.assertNotIn(self.test_model_name, cache_status['cached_models'])
            print(f"  ✅ Cache updated: {cache_status['cache_size']} models remaining")
            
        except Exception as e:
            self.fail(f"Cleanup test failed: {e}")

class TestClientIntegration(unittest.TestCase):
    """Integration tests for client workflow."""
    
    def setUp(self):
        """Set up for integration tests."""
        self.storage_url = "http://localhost:8002"
        self.trainer_url = "http://localhost:8001"
        self.predictor_url = "http://localhost:8000"
    
    def test_complete_workflow(self):
        """Test complete ML workflow through client."""
        print("\n🔄 Testing complete workflow...")
        
        workflow_id = datetime.now().strftime('%H%M%S')
        
        try:
            # Step 1: Load data
            response = requests.get(f"{self.storage_url}/files")
            files_data = response.json()
            
            if files_data['count'] == 0:
                self.skipTest("No test files available for workflow test")
            
            test_file = files_data['available_files'][0]['file_name']
            dataset_id = f"workflow_test_{workflow_id}"
            
            load_data = {
                "file_name": test_file,
                "dataset_id": dataset_id
            }
            response = requests.post(f"{self.storage_url}/data/load", json=load_data)
            self.assertEqual(response.status_code, 200)
            load_result = response.json()
            
            # Step 2: Prepare data
            target_column = load_result['columns'][-1]
            prep_data = {
                "dataset_id": dataset_id,
                "target_column": target_column,
                "train_size": 0.8
            }
            response = requests.post(f"{self.storage_url}/data/prepare", json=prep_data)
            self.assertEqual(response.status_code, 200)
            prep_result = response.json()
            
            # Step 3: Train model
            model_name = f"workflow_model_{workflow_id}"
            train_data = {
                "dataset_id": prep_result['dataset_id'],
                "target_column": target_column,
                "model_name": model_name,
                "metadata": {"workflow_test": True}
            }
            response = requests.post(f"{self.trainer_url}/train", json=train_data)
            self.assertEqual(response.status_code, 200)
            train_result = response.json()
            
            # Step 4: Evaluate model
            eval_data = {"model_name": model_name}
            response = requests.post(f"{self.trainer_url}/evaluate", json=eval_data)
            self.assertEqual(response.status_code, 200)
            eval_result = response.json()
            
            # Step 5: Load to predictor and predict
            load_data = {"model_name": model_name}
            response = requests.post(f"{self.predictor_url}/load_model", json=load_data)
            self.assertEqual(response.status_code, 200)
            
            # Create test record
            test_record = {}
            for feature in train_result['features']:
                if feature != target_column:
                    test_record[feature] = "test_value"
            
            pred_data = {
                "model_name": model_name,
                "record": test_record
            }
            response = requests.post(f"{self.predictor_url}/predict", json=pred_data)
            self.assertEqual(response.status_code, 200)
            pred_result = response.json()
            
            print(f"  ✅ Complete workflow successful:")
            print(f"    📊 Data: {load_result['shape']}")
            print(f"    🎓 Model: {len(train_result['classes'])} classes")
            print(f"    📈 Accuracy: {eval_result['accuracy_percentage']}")
            print(f"    🔮 Prediction: {pred_result['prediction']}")
            
        except Exception as e:
            self.fail(f"Complete workflow test failed: {e}")

def run_performance_tests():
    """Run performance and stress tests."""
    print("\n⚡ Running Performance Tests...")
    
    storage_url = "http://localhost:8002"
    predictor_url = "http://localhost:8000"
    
    # Test multiple simultaneous requests
    start_time = time.time()
    
    try:
        # Health check performance
        health_times = []
        for i in range(10):
            start = time.time()
            response = requests.get(f"{storage_url}/health")
            end = time.time()
            health_times.append(end - start)
        
        avg_health_time = sum(health_times) / len(health_times)
        print(f"  📊 Health check average: {avg_health_time:.3f}s")
        
        # Cache status performance
        cache_times = []
        for i in range(10):
            start = time.time()
            response = requests.get(f"{predictor_url}/cache")
            end = time.time()
            cache_times.append(end - start)
        
        avg_cache_time = sum(cache_times) / len(cache_times)
        print(f"  🗂️ Cache status average: {avg_cache_time:.3f}s")
        
        total_time = time.time() - start_time
        print(f"  ⏱️ Total performance test time: {total_time:.2f}s")
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")

def main():
    """Run all tests."""
    print("🧪 NaiveHub Client Test Suite")
    print("=" * 50)
    
    # Check if servers are running
    print("🔍 Checking server availability...")
    servers = {
        "Storage": "http://localhost:8002",
        "Training": "http://localhost:8001", 
        "Prediction": "http://localhost:8000"
    }
    
    all_online = True
    for name, url in servers.items():
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"  ✅ {name} server: Online")
            else:
                print(f"  ❌ {name} server: Error (Status: {response.status_code})")
                all_online = False
        except Exception as e:
            print(f"  ❌ {name} server: Offline ({e})")
            all_online = False
    
    if not all_online:
        print("\n❌ Some servers are not available. Please start all servers and try again.")
        print("Run: docker-compose up -d")
        return False
    
    print("\n✅ All servers are online. Starting tests...\n")
    
    # Run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNaiveHubClient))
    suite.addTests(loader.loadTestsFromTestCase(TestClientIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance tests
    run_performance_tests()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 All tests passed! Your NaiveHub client is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
    
    return success

if __name__ == "__main__":
    main()
