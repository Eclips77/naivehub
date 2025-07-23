#!/usr/bin/env python3
"""
NaiveHub Integration Test Suite
==============================

This script performs comprehensive testing of the entire NaiveHub system:
- Standalone library components
- Docker containers
- API endpoints
- Data flow between services
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import subprocess
from typing import Dict, Any, List

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class NaiveHubTester:
    def __init__(self):
        self.trainer_url = "http://localhost:8001"
        self.classifier_url = "http://localhost:8000"
        self.test_model_name = "test_model"
        self.test_data_file = "test_data.csv"
        self.results = []
        
    def print_header(self, title: str):
        """Print formatted test section header"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}{title:^60}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    
    def print_test(self, test_name: str, status: str, details: str = ""):
        """Print test result with colors"""
        if status == "PASS":
            color = Colors.GREEN
            symbol = "✅"
        elif status == "FAIL":
            color = Colors.RED
            symbol = "❌"
        elif status == "SKIP":
            color = Colors.YELLOW
            symbol = "⏭️"
        else:
            color = Colors.WHITE
            symbol = "ℹ️"
            
        print(f"{symbol} {color}{test_name}: {status}{Colors.END}")
        if details:
            print(f"   {Colors.WHITE}{details}{Colors.END}")
        
        self.results.append({
            "test": test_name,
            "status": status,
            "details": details
        })
    
    def create_test_data(self) -> bool:
        """Create sample test data"""
        try:
            test_data = {
                'weather': ['sunny', 'rainy', 'cloudy', 'sunny', 'rainy', 'cloudy', 
                           'sunny', 'rainy', 'cloudy', 'sunny', 'rainy', 'cloudy',
                           'sunny', 'rainy', 'cloudy', 'sunny'],
                'temperature': ['hot', 'cold', 'mild', 'mild', 'cold', 'mild',
                               'hot', 'cold', 'mild', 'hot', 'mild', 'cold',
                               'hot', 'cold', 'mild', 'mild'],
                'humidity': ['high', 'low', 'medium', 'low', 'high', 'medium',
                            'low', 'high', 'low', 'medium', 'high', 'low',
                            'high', 'low', 'medium', 'low'],
                'play_tennis': ['no', 'no', 'yes', 'yes', 'no', 'yes',
                               'yes', 'no', 'yes', 'yes', 'yes', 'no',
                               'no', 'no', 'yes', 'yes']
            }
            
            df = pd.DataFrame(test_data)
            df.to_csv(self.test_data_file, index=False)
            return True
        except Exception as e:
            print(f"Error creating test data: {e}")
            return False
    
    def test_standalone_components(self):
        """Test standalone library components"""
        self.print_header("TESTING STANDALONE COMPONENTS")
        
        try:
            # Test DataManager
            from managers.data_manager import DataManager
            data_manager = DataManager(self.test_data_file)
            train_df, test_df = data_manager.prepare_data()
            
            if len(train_df) > 0 and len(test_df) > 0:
                self.print_test("DataManager", "PASS", f"Train: {len(train_df)}, Test: {len(test_df)}")
            else:
                self.print_test("DataManager", "FAIL", "Empty datasets returned")
                return False
                
        except Exception as e:
            self.print_test("DataManager", "FAIL", str(e))
            return False
        
        try:
            # Test NaiveBayesTrainer
            from services.trainer import NaiveBayesTrainer
            trainer = NaiveBayesTrainer()
            model = trainer.fit(train_df, "play_tennis")
            
            if "classes" in model and "priors" in model and "likelihoods" in model:
                self.print_test("NaiveBayesTrainer", "PASS", f"Classes: {model['classes']}")
            else:
                self.print_test("NaiveBayesTrainer", "FAIL", "Invalid model structure")
                return False
                
        except Exception as e:
            self.print_test("NaiveBayesTrainer", "FAIL", str(e))
            return False
        
        try:
            # Test NaiveBayesPredictor
            from services.classifier import NaiveBayesPredictor
            predictor = NaiveBayesPredictor(model)
            
            test_sample = {"weather": "sunny", "temperature": "hot", "humidity": "low"}
            prediction = predictor.predict(test_sample)
            
            if prediction in model["classes"]:
                self.print_test("NaiveBayesPredictor", "PASS", f"Prediction: {prediction}")
            else:
                self.print_test("NaiveBayesPredictor", "FAIL", f"Invalid prediction: {prediction}")
                return False
                
        except Exception as e:
            self.print_test("NaiveBayesPredictor", "FAIL", str(e))
            return False
        
        try:
            # Test NaiveBayesEvaluator
            from services.evaluator import NaiveBayesEvaluator
            evaluator = NaiveBayesEvaluator(predictor)
            
            X_test = test_df.drop(columns=["play_tennis"])
            y_test = test_df["play_tennis"]
            results = evaluator.evaluate(X_test, y_test)
            
            if "accuracy" in results and 0 <= results["accuracy"] <= 1:
                self.print_test("NaiveBayesEvaluator", "PASS", f"Accuracy: {results['accuracy']:.3f}")
            else:
                self.print_test("NaiveBayesEvaluator", "FAIL", "Invalid evaluation results")
                return False
                
        except Exception as e:
            self.print_test("NaiveBayesEvaluator", "FAIL", str(e))
            return False
        
        return True
    
    def check_docker_status(self) -> bool:
        """Check if Docker containers are running"""
        self.print_header("CHECKING DOCKER CONTAINERS")
        
        try:
            result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
            if result.returncode != 0:
                self.print_test("Docker", "FAIL", "Docker not available")
                return False
                
            output = result.stdout
            trainer_running = "naivehub-trainer" in output
            classifier_running = "naivehub-classifier" in output
            
            if trainer_running:
                self.print_test("Training Container", "PASS", "Container is running")
            else:
                self.print_test("Training Container", "FAIL", "Container not found")
                
            if classifier_running:
                self.print_test("Classification Container", "PASS", "Container is running")
            else:
                self.print_test("Classification Container", "FAIL", "Container not found")
                
            return trainer_running and classifier_running
            
        except Exception as e:
            self.print_test("Docker Check", "FAIL", str(e))
            return False
    
    def wait_for_services(self, timeout: int = 30) -> bool:
        """Wait for services to be ready"""
        self.print_header("WAITING FOR SERVICES")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check trainer health
                trainer_resp = requests.get(f"{self.trainer_url}/health", timeout=5)
                trainer_ready = trainer_resp.status_code == 200
                
                # Check classifier health
                classifier_resp = requests.get(f"{self.classifier_url}/health", timeout=5)
                classifier_ready = classifier_resp.status_code == 200
                
                if trainer_ready and classifier_ready:
                    self.print_test("Service Readiness", "PASS", f"Both services ready in {time.time() - start_time:.1f}s")
                    return True
                    
                time.sleep(2)
                
            except Exception:
                time.sleep(2)
                continue
        
        self.print_test("Service Readiness", "FAIL", f"Services not ready after {timeout}s")
        return False
    
    def test_api_endpoints(self) -> bool:
        """Test all API endpoints"""
        self.print_header("TESTING API ENDPOINTS")
        
        # Copy test data to trainer container
        try:
            subprocess.run([
                "docker", "cp", self.test_data_file, "naivehub-trainer:/app/"
            ], check=True, capture_output=True)
            self.print_test("Copy Test Data", "PASS", "Data copied to trainer container")
        except Exception as e:
            self.print_test("Copy Test Data", "FAIL", str(e))
            return False
        
        # Test training endpoint
        try:
            train_data = {
                "file_path": self.test_data_file,
                "target_column": "play_tennis",
                "model_name": self.test_model_name
            }
            
            response = requests.post(f"{self.trainer_url}/train", json=train_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                self.print_test("Training API", "PASS", f"Model trained: {result['message']}")
            else:
                self.print_test("Training API", "FAIL", f"Status: {response.status_code}, {response.text}")
                return False
                
        except Exception as e:
            self.print_test("Training API", "FAIL", str(e))
            return False
        
        # Test model list endpoint
        try:
            response = requests.get(f"{self.trainer_url}/models", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if self.test_model_name in result.get("available_models", []):
                    self.print_test("Model List API", "PASS", f"Found models: {result['available_models']}")
                else:
                    self.print_test("Model List API", "FAIL", f"Test model not in list: {result}")
                    return False
            else:
                self.print_test("Model List API", "FAIL", f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.print_test("Model List API", "FAIL", str(e))
            return False
        
        # Test model loading
        try:
            load_data = {"model_name": self.test_model_name}
            response = requests.post(f"{self.classifier_url}/load_model", json=load_data, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                self.print_test("Model Load API", "PASS", result["message"])
            else:
                self.print_test("Model Load API", "FAIL", f"Status: {response.status_code}, {response.text}")
                return False
                
        except Exception as e:
            self.print_test("Model Load API", "FAIL", str(e))
            return False
        
        # Test prediction endpoint
        try:
            predict_data = {
                "model_name": self.test_model_name,
                "record": {"weather": "sunny", "temperature": "hot", "humidity": "low"}
            }
            
            response = requests.post(f"{self.classifier_url}/predict", json=predict_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                self.print_test("Prediction API", "PASS", f"Prediction: {result['prediction']}")
            else:
                self.print_test("Prediction API", "FAIL", f"Status: {response.status_code}, {response.text}")
                return False
                
        except Exception as e:
            self.print_test("Prediction API", "FAIL", str(e))
            return False
        
        # Test evaluation endpoint
        try:
            eval_data = {
                "model_name": self.test_model_name,
                "test_data": [
                    {"weather": "sunny", "temperature": "hot", "humidity": "low", "play_tennis": "yes"},
                    {"weather": "rainy", "temperature": "cold", "humidity": "high", "play_tennis": "no"},
                    {"weather": "cloudy", "temperature": "mild", "humidity": "medium", "play_tennis": "yes"}
                ]
            }
            
            response = requests.post(f"{self.classifier_url}/evaluate", json=eval_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                self.print_test("Evaluation API", "PASS", f"Accuracy: {result['accuracy_percentage']}")
            else:
                self.print_test("Evaluation API", "FAIL", f"Status: {response.status_code}, {response.text}")
                return False
                
        except Exception as e:
            self.print_test("Evaluation API", "FAIL", str(e))
            return False
        
        return True
    
    def test_volume_persistence(self) -> bool:
        """Test model volume persistence"""
        self.print_header("TESTING VOLUME PERSISTENCE")
        
        try:
            # Check if model file exists in volume
            result = subprocess.run([
                "docker", "exec", "naivehub-classifier", "ls", "-la", "/app/models/"
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and f"{self.test_model_name}.json" in result.stdout:
                self.print_test("Model File Persistence", "PASS", "Model saved to volume")
            else:
                self.print_test("Model File Persistence", "FAIL", "Model file not found in volume")
                return False
                
        except Exception as e:
            self.print_test("Model File Persistence", "FAIL", str(e))
            return False
        
        try:
            # Check volume info
            result = subprocess.run([
                "docker", "volume", "inspect", "pythonproject7_models-volume"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.print_test("Volume Inspection", "PASS", "Volume exists and accessible")
            else:
                self.print_test("Volume Inspection", "FAIL", "Volume not found")
                return False
                
        except Exception as e:
            self.print_test("Volume Inspection", "FAIL", str(e))
            return False
        
        return True
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("TEST SUMMARY")
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.results if r["status"] == "FAIL"])
        skipped_tests = len([r for r in self.results if r["status"] == "SKIP"])
        
        print(f"\n{Colors.BOLD}Total Tests: {total_tests}{Colors.END}")
        print(f"{Colors.GREEN}✅ Passed: {passed_tests}{Colors.END}")
        print(f"{Colors.RED}❌ Failed: {failed_tests}{Colors.END}")
        print(f"{Colors.YELLOW}⏭️ Skipped: {skipped_tests}{Colors.END}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"\n{Colors.BOLD}Success Rate: {success_rate:.1f}%{Colors.END}")
        
        if failed_tests > 0:
            print(f"\n{Colors.RED}{Colors.BOLD}FAILED TESTS:{Colors.END}")
            for result in self.results:
                if result["status"] == "FAIL":
                    print(f"{Colors.RED}❌ {result['test']}: {result['details']}{Colors.END}")
        
        return failed_tests == 0
    
    def cleanup(self):
        """Cleanup test files"""
        try:
            if os.path.exists(self.test_data_file):
                os.remove(self.test_data_file)
        except Exception:
            pass
    
    def run_all_tests(self):
        """Run complete test suite"""
        print(f"{Colors.PURPLE}{Colors.BOLD}")
        print("🧪 NaiveHub Integration Test Suite")
        print("==================================")
        print(f"{Colors.END}")
        
        # Create test data
        if not self.create_test_data():
            print(f"{Colors.RED}Failed to create test data. Exiting.{Colors.END}")
            return False
        
        # Test standalone components
        standalone_ok = self.test_standalone_components()
        
        # Check Docker containers
        docker_ok = self.check_docker_status()
        
        if docker_ok:
            # Wait for services
            services_ok = self.wait_for_services()
            
            if services_ok:
                # Test APIs
                api_ok = self.test_api_endpoints()
                
                # Test volume persistence
                volume_ok = self.test_volume_persistence()
            else:
                self.print_test("API Tests", "SKIP", "Services not ready")
                self.print_test("Volume Tests", "SKIP", "Services not ready")
        else:
            self.print_test("Service Check", "SKIP", "Docker containers not running")
            self.print_test("API Tests", "SKIP", "Docker containers not running")
            self.print_test("Volume Tests", "SKIP", "Docker containers not running")
        
        # Print summary
        success = self.print_summary()
        
        # Cleanup
        self.cleanup()
        
        return success

def main():
    """Main test runner"""
    tester = NaiveHubTester()
    
    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.END}")
        tester.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.END}")
        tester.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
