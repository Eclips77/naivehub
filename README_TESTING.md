# NaiveHub Testing Guide
## 🧪 Comprehensive Test Suite

This document explains how to test the entire NaiveHub system to ensure everything works correctly.

## 📋 Test Overview

The test suite (`test_naivehub.py`) performs comprehensive testing of:

### 🔧 Standalone Components
- **DataManager**: Data loading, cleaning, and splitting
- **NaiveBayesTrainer**: Model training functionality
- **NaiveBayesPredictor**: Prediction capabilities
- **NaiveBayesEvaluator**: Model evaluation metrics

### 🐳 Docker Infrastructure
- Container status verification
- Service health checks
- Volume persistence testing

### 🌐 API Endpoints
- Training service APIs (port 8001)
- Classification service APIs (port 8000)
- Cross-service communication
- Data flow validation

## 🚀 Quick Start

### Option 1: Quick Test Script (Recommended)

**Windows:**
```bash
quick-test.bat
```

**Linux/Mac:**
```bash
chmod +x quick-test.sh
./quick-test.sh
```

### Option 2: Manual Testing

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the test suite:**
   ```bash
   python test_naivehub.py
   ```

## 📊 Test Scenarios

### Scenario 1: Standalone Testing Only
- No Docker required
- Tests library components only
- Perfect for development validation

### Scenario 2: Full Integration Testing
- Requires Docker containers running
- Tests entire microservices architecture
- Validates API endpoints and data flow

## 🐳 Docker Testing Prerequisites

Before running full integration tests, ensure Docker containers are running:

```bash
# Start the entire system
docker-compose up -d

# Or use the startup script
docker-start.bat    # Windows
./docker-start.sh   # Linux/Mac
```

## 📈 Test Results Interpretation

### ✅ Success Indicators
- **Green checkmarks**: Tests passed successfully
- **High success rate**: >90% indicates healthy system
- **All APIs responding**: Full integration working

### ❌ Failure Indicators
- **Red X marks**: Component or integration failures
- **Service timeouts**: Docker/networking issues
- **API errors**: Server configuration problems

### Example Successful Output:
```
✅ DataManager: PASS - Train: 12, Test: 4
✅ NaiveBayesTrainer: PASS - Classes: ['no', 'yes']
✅ NaiveBayesPredictor: PASS - Prediction: yes
✅ Training Container: PASS - Container is running
✅ Service Readiness: PASS - Both services ready in 3.2s
✅ Training API: PASS - Model trained successfully
✅ Prediction API: PASS - Prediction: yes
✅ Model File Persistence: PASS - Model saved to volume

Success Rate: 100.0%
```

## 🔧 Troubleshooting

### Common Issues

**1. Python Dependencies Missing**
```bash
# Solution
pip install -r requirements.txt
```

**2. Docker Containers Not Running**
```bash
# Solution
docker-compose up -d
# Wait 30 seconds for services to start
```

**3. Port Conflicts**
```bash
# Check if ports are in use
netstat -an | findstr ":8000"  # Windows
netstat -an | grep ":8000"     # Linux/Mac

# Solution: Stop conflicting services or change ports
```

**4. Permission Issues (Linux/Mac)**
```bash
# Solution
chmod +x quick-test.sh
chmod +x docker-start.sh
```

**5. Docker Volume Issues**
```bash
# Reset volumes if needed
docker-compose down -v
docker-compose up -d
```

## 📝 Test Data

The test suite automatically creates sample data:
- Weather conditions (sunny, rainy, cloudy)
- Temperature levels (hot, cold, mild)
- Humidity levels (high, low, medium)
- Tennis playing decisions (yes, no)

This synthetic dataset ensures consistent testing results.

## 🔄 Continuous Testing

### Development Workflow
1. Make code changes
2. Run `quick-test.bat` / `quick-test.sh`
3. Verify all tests pass
4. Commit changes

### CI/CD Integration
The test script can be integrated into CI/CD pipelines:
```yaml
# Example GitHub Actions step
- name: Run NaiveHub Tests
  run: python test_naivehub.py
```

## 📞 Test Support

If tests fail consistently:

1. **Check logs**: `docker-compose logs`
2. **Verify system requirements**: Python 3.11+, Docker
3. **Reset environment**: `docker-compose down -v && docker-compose up -d`
4. **Review error messages**: The test suite provides detailed failure information

## 🎯 Test Coverage

Current test coverage includes:
- ✅ Data processing pipeline
- ✅ Machine learning algorithms
- ✅ API endpoints
- ✅ Docker containerization
- ✅ Volume persistence
- ✅ Service communication
- ✅ Health monitoring

The test suite provides confidence that all system components work together correctly.
