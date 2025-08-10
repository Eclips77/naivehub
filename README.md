# NaiveHub - Naive Bayes Classification System

A comprehensive Naive Bayes classification system built with Python, featuring both standalone library usage and REST API services with Docker containerization.

## 🚀 Features

- **Complete Naive Bayes Implementation**: Custom implementation with Laplace smoothing
- **Data Pipeline**: Automated data loading, cleaning, and splitting
- **Microservices Architecture**: Separate training and classification servers with clear separation of concerns
- **Docker Containerization**: Easy deployment with Docker Compose
- **REST API**: FastAPI-based web services for remote model training and prediction
- **Comprehensive Evaluation**: Built-in model evaluation with detailed metrics on training server
- **Easy Integration**: Well-structured managers for different workflows
- **Comprehensive Testing**: Complete test suite with standalone and integration testing

## 📁 Project Structure

```
├── main.py                     # Main demonstration script
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker services configuration
├── Dockerfile.trainer          # Training server Docker image
├── Dockerfile.classifier       # Classification server Docker image
├── .gitignore                  # Git ignore patterns
├── test_naivehub.py           # Comprehensive test suite
├── USAGE_GUIDE.md             # Detailed usage guide
├── POSTMAN_GUIDE_V2.md        # API testing with Postman
├── managers/                   # High-level workflow managers
│   ├── classifier_manager.py   # Classification workflow management
│   ├── data_manager.py         # Data preparation pipeline
│   └── trainer_manager.py      # Model training workflow
├── services/                   # Core algorithm implementations
│   ├── classifier.py           # Naive Bayes prediction logic
│   ├── evaluator.py           # Model evaluation metrics
│   └── trainer.py             # Naive Bayes training algorithm
├── servers/                    # REST API servers
│   ├── classifier_server.py   # Classification API service (simplified)
│   └── trainer_server.py      # Training API service (with evaluation)
├── utils/                      # Utility modules
│   ├── data_cleaner.py        # Data cleaning operations
│   ├── data_loader.py         # Data loading from various formats
│   ├── data_splitter.py       # Train/test splitting
│   └── model_loader.py        # Model serialization/loading
└── Data/                      # Data directory (empty by default)
```

## 🛠️ Installation

### Option 1: Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <(https://github.com/Eclips77/naivehub.git)>
   cd NaiveHub
   ```

2. **Start with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

   This will start both services:
   - Training Server: http://localhost:8001
   - Classification Server: http://localhost:8000

3. **Check service health**:
   ```bash
   curl http://localhost:8001/health
   curl http://localhost:8000/health
   ```

### Option 2: Local Development

1. **Clone the repository**:
   ```bash
   git clone <(https://github.com/Eclips77/naivehub.git)>
   cd NaiveHub
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start servers manually** (see Server Setup section below)

## 🎯 Quick Start

### Standalone Usage

Run the complete demonstration:

```bash
python main.py
```

This will:
- Create sample data
- Train a Naive Bayes model
- Make predictions
- Evaluate performance
- Show API usage examples

### Library Usage

```python
from managers.data_manager import DataManager
from managers.trainer_manager import NaiveBayesTrainingManager
from managers.classifier_manager import NaiveBayesClassificationManager

# Prepare data
data_manager = DataManager("your_data.csv")
train_df, test_df = data_manager.prepare_data()

# Train model
trainer = NaiveBayesTrainingManager(train_df, "target_column", "model.json")
model = trainer.train_and_save()

# Make predictions
classifier = NaiveBayesClassificationManager(model_path="model.json")
prediction = classifier.predict_single({"feature1": "value1", "feature2": "value2"})
```

## 🌐 Microservices Architecture

NaiveHub uses a simplified two-server architecture with clear separation of concerns:

### 🎓 Training Server (Port 8001)
- **Primary Purpose**: Complete training workflow management
- **Responsibilities**: 
  - Data processing, cleaning, and splitting
  - Model training with Naive Bayes algorithm
  - Model evaluation using stored test data
  - Model storage and serving to classification server
- **Storage**: Keeps trained models and test data in memory for evaluation
- **Key Feature**: Handles evaluation requests using stored test data for trained models

### 🔮 Classification Server (Port 8000)  
- **Primary Purpose**: Model serving and prediction only
- **Responsibilities**:
  - Loading models from training server
  - Saving models locally as JSON files
  - Making predictions on new data
  - Model management (loading/unloading)
- **Storage**: Downloads models from training server and caches locally
- **Simplified Design**: No evaluation functionality - focused purely on predictions

### 🔄 Communication Flow
1. **Training Server** processes data, trains models, and stores test data for evaluation
2. **Classification Server** requests trained models from training server
3. **Training Server** sends model data to classification server
4. **Classification Server** saves models as JSON files and loads them for predictions
5. **Clients** send prediction requests to Classification Server
6. **Clients** send evaluation requests to Training Server (using stored test data)

## 🐳 Docker Setup

### Quick Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and start (after code changes)
docker-compose build --no-cache
docker-compose up -d
```

### Individual Container Management

```bash
# Build specific service
docker-compose build trainer
docker-compose build classifier

# Restart specific service
docker-compose restart trainer
docker-compose restart classifier
```

## 🛠️ Server Setup (Local Development)

### Option 1: Using Startup Scripts

**Windows:**
```bash
start_servers.bat
```

**Linux/Mac:**
```bash
chmod +x start_servers.sh
./start_servers.sh
```

### Option 2: Manual Startup

**Terminal 1 - Training Server:**
```bash
uvicorn servers.trainer_server:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 2 - Classification Server:**
```bash
uvicorn servers.classifier_server:app --host 0.0.0.0 --port 8000 --reload
```

## 📋 Complete Workflow Example

### 1. Train a Model
```bash
curl -X POST "http://localhost:8001/train" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "sample_data.csv",
    "target_column": "play_tennis",
    "model_name": "tennis_model"
  }'
```

### 2. Evaluate the Model (NEW - on Training Server)
```bash
curl -X POST "http://localhost:8001/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "tennis_model"
  }'
```

### 3. Load Model into Classification Server
```bash
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "tennis_model"
  }'
```

### 4. Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "tennis_model",
    "record": {
      "weather": "sunny",
      "temperature": "hot",
      "humidity": "low"
    }
  }'
```

## 🌐 API Endpoints

### Training Server (http://localhost:8001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train` | Train a new model with data processing |
| POST | `/evaluate` | **NEW** - Evaluate model using stored test data |
| GET | `/models` | List all trained models in memory |
| GET | `/model/{model_name}` | Get specific model data |
| GET | `/health` | Health check with training capabilities |

### Classification Server (http://localhost:8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/load_model` | Load model from training server |
| POST | `/predict` | Make predictions using loaded models |
| GET | `/models` | List models available on training server |
| GET | `/health` | Health check with loaded models info |

**Note**: Evaluation functionality has been moved from Classification Server to Training Server for better separation of concerns.

## 📊 Example Usage

### Training a Model

```python
import pandas as pd
from managers.data_manager import DataManager
from managers.trainer_manager import NaiveBayesTrainingManager

# Load and prepare data
data_manager = DataManager("tennis_data.csv")
train_df, test_df = data_manager.prepare_data()

# Train model
trainer = NaiveBayesTrainingManager(
    df=train_df,
    label_column="play_tennis",
    output_path="tennis_model.json"
)
model = trainer.train_and_save()
```

### Making Predictions

```python
from managers.classifier_manager import NaiveBayesClassificationManager

# Load trained model
classifier = NaiveBayesClassificationManager(model_path="tennis_model.json")

# Single prediction
prediction = classifier.predict_single({
    "weather": "sunny",
    "temperature": "hot",
    "humidity": "low"
})

# Batch predictions
predictions = classifier.predict_batch(test_features_df)

# Model evaluation
results = classifier.evaluate_model(X_test, y_test)
print(f"Accuracy: {results['accuracy']:.3f}")
```

## 🔧 Configuration

### Environment Variables

- `TRAINER_URL`: URL of the training server (default: http://localhost:8001)

### Docker Configuration

The `docker-compose.yml` file configures:
- **Network**: Custom bridge network for service communication
- **Volumes**: Persistent storage for models and data
- **Health Checks**: Automatic service health monitoring
- **Port Mapping**: External access to services

### Data Format

The system expects CSV files with:
- Categorical features (string values)
- A target column with class labels
- No missing values (automatically cleaned)

## 📖 Documentation

- **`USAGE_GUIDE.md`**: Comprehensive usage examples and workflows
- **`POSTMAN_GUIDE_V2.md`**: API testing guide with Postman collections
- **Inline Documentation**: Detailed docstrings in all modules

## 📈 Performance & Features

- **Laplace Smoothing**: Handles unseen feature values gracefully
- **Log Probabilities**: Prevents numerical underflow in calculations
- **Automatic Data Cleaning**: Removes missing values and duplicates
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score
- **Microservices Ready**: Scalable API architecture with Docker support
- **Clear Separation of Concerns**: Training server handles training/evaluation, classification server handles predictions
- **Persistent Storage**: Docker volumes ensure data persistence
- **Health Monitoring**: Built-in health checks for all services
- **Comprehensive Testing**: Full test coverage with automated validation

## 🏗️ Architecture Benefits

### Simplified Design
- **Training Server**: Handles complete training workflow including evaluation
- **Classification Server**: Focused solely on model serving and predictions
- **Clear Responsibilities**: No overlap between training and prediction concerns

### Scalability
- **Independent Scaling**: Scale training and prediction services separately
- **Docker Support**: Easy horizontal scaling with container orchestration
- **Stateless Predictions**: Classification server can be replicated easily

### Maintainability
- **Modular Design**: Easy to modify individual components
- **Comprehensive Testing**: Automated validation of all functionality
- **Documentation**: Complete guides for usage and API testing

## 🧪 Testing

### Comprehensive Test Suite

Run the complete test suite that validates the entire system:

```bash
python test_naivehub.py
```

This test suite includes:
- ✅ **Standalone Component Testing**: Data managers, trainers, classifiers
- ✅ **Docker Integration Testing**: Container health and communication
- ✅ **API Endpoint Testing**: All server endpoints and workflows
- ✅ **End-to-End Testing**: Complete training → evaluation → prediction workflow

### Manual Testing

Run the demonstration to verify installation:

```bash
python main.py
```

### Docker Testing

Test the containerized services:

```bash
# Start services
docker-compose up -d

# Run test suite
python test_naivehub.py

# Check logs
docker-compose logs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues, please open an issue on the repository or contact the development team.

---

**NaiveHub** - Simple, powerful, and scalable Naive Bayes classification! 🎯
