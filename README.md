# NaiveHub - Naive Bayes Classification System

A comprehensive Naive Bayes classification system built with Python, featuring both standalone library usage and REST API services.

## 🚀 Features

- **Complete Naive Bayes Implementation**: Custom implementation with Laplace smoothing
- **Data Pipeline**: Automated data loading, cleaning, and splitting
- **Microservices Architecture**: Separate training and classification servers
- **REST API**: FastAPI-based web services for remote model training and prediction
- **Comprehensive Evaluation**: Built-in model evaluation with detailed metrics
- **Easy Integration**: Well-structured managers for different workflows

## 📁 Project Structure

```
├── main.py                     # Main demonstration script
├── requirements.txt            # Python dependencies
├── managers/                   # High-level workflow managers
│   ├── classifier_manager.py   # Classification workflow management
│   ├── data_manager.py         # Data preparation pipeline
│   └── trainer_manager.py      # Model training workflow
├── services/                   # Core algorithm implementations
│   ├── classifier.py           # Naive Bayes prediction logic
│   ├── evaluator.py           # Model evaluation metrics
│   └── trainer.py             # Naive Bayes training algorithm
├── servers/                    # REST API servers
│   ├── classifier_server.py   # Classification API service
│   └── trainer_server.py      # Training API service
├── utils/                      # Utility modules
│   ├── data_cleaner.py        # Data cleaning operations
│   ├── data_loader.py         # Data loading from various formats
│   ├── data_splitter.py       # Train/test splitting
│   └── model_loader.py        # Model serialization/loading
└── Data/                      # Data directory (empty by default)
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd NaiveHub
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

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

NaiveHub uses a two-server architecture:

### 🎓 Training Server (Port 8001)
- **Purpose**: Data processing, cleaning, splitting, and model training
- **Storage**: Keeps trained models in memory
- **Workflow**: Receives training requests → processes data → trains models → stores in memory

### 🔮 Classification Server (Port 8000)  
- **Purpose**: Model serving and predictions
- **Storage**: Downloads models from training server and saves as JSON files locally
- **Workflow**: Requests models from training server → saves locally → loads for predictions

### 🔄 Communication Flow
1. **Training Server** processes data and trains models
2. **Classification Server** requests trained models
3. **Training Server** sends model data  
4. **Classification Server** saves models as JSON files and loads for predictions
5. **Clients** send prediction requests to Classification Server

## 🛠️ Quick Server Setup

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

### 2. Load Model into Classification Server
```bash
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "tennis_model"
  }'
```

### 3. Make Predictions
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
| GET | `/models` | List all trained models in memory |
| GET | `/model/{model_name}` | Get specific model data |
| GET | `/health` | Health check |

### Classification Server (http://localhost:8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/load_model` | Load model from training server |
| POST | `/load_local_model` | Load model from local JSON file |
| POST | `/predict` | Make predictions |
| GET | `/models/available` | List models available on training server |
| GET | `/models/loaded` | List models loaded in memory |
| GET | `/models/local` | List models saved locally as JSON |
| GET | `/health` | Health check |

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

### Data Format

The system expects CSV files with:
- Categorical features (string values)
- A target column with class labels
- No missing values (automatically cleaned)

## 📈 Performance & Features

- **Laplace Smoothing**: Handles unseen feature values
- **Log Probabilities**: Prevents numerical underflow
- **Automatic Data Cleaning**: Removes missing values and duplicates
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score
- **Microservices Ready**: Scalable API architecture

## 🧪 Testing

Run the demonstration to verify installation:

```bash
python main.py
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
