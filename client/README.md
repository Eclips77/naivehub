# NaiveHub Streamlit Client

## 🤖 Professional Machine Learning Platform

This is a comprehensive Streamlit client for the NaiveHub 3-tier machine learning architecture, providing a user-friendly interface for data management, model training, and predictions.

## 🏗️ Architecture

The client interfaces with three microservices:
- **🗄️ Storage Server (Port 8002)**: Data and model management
- **🎓 Training Server (Port 8001)**: Model training and evaluation  
- **🔮 Prediction Server (Port 8000)**: High-performance predictions with caching

## 🚀 Quick Start

### Prerequisites

1. Ensure all three servers are running:
```bash
docker-compose up -d
```

2. Install client dependencies:
```bash
pip install -r requirements.txt
```

### Running the Client

```bash
streamlit run naivehub_client.py
```

The client will be available at `http://localhost:8501`

## ✨ Features

### 🖥️ Server Status Dashboard
- Real-time health monitoring of all three servers
- Resource usage metrics
- Connection status validation

### 📊 Data Management
- **File Upload**: Load CSV files from the local Data directory
- **URL Import**: Download and import data from web URLs
- **Dataset Explorer**: Browse and manage loaded datasets
- **Data Preparation**: Clean and split data for training

### 🎓 Model Training
- Interactive training interface
- Model metadata and versioning
- Automatic evaluation with visualization
- Training history and model comparison

### 🔮 Predictions
- Single record predictions
- Batch prediction from CSV files
- Model cache management
- Performance monitoring

### 🧪 Comprehensive Testing
- Full system validation
- Performance benchmarking
- End-to-end workflow testing
- Automated health checks

## 🎯 Usage Guide

### 1. Server Status
Start here to verify all servers are online and healthy. Monitor resource usage and connection status.

### 2. Data Management
- Upload CSV files or import from URLs
- Explore your datasets with the built-in viewer
- Prepare data for training by selecting target columns and split ratios

### 3. Model Training
- Select prepared datasets
- Configure model parameters and metadata
- Train Naive Bayes models with automatic evaluation
- View accuracy metrics and performance charts

### 4. Predictions
- Load trained models to the prediction server
- Make single predictions with manual input
- Upload CSV files for batch predictions
- Download prediction results

### 5. System Testing
Run comprehensive tests to validate the entire system:
- Server connectivity
- Data operations
- Model training and evaluation
- Prediction pipeline
- Cache management

## 🔧 Configuration

### Server Endpoints
```python
STORAGE_URL = "http://localhost:8002"
TRAINER_URL = "http://localhost:8001"
PREDICTOR_URL = "http://localhost:8000"
```

### Customization
The client can be easily customized by modifying:
- Server URLs in the configuration section
- UI components and styling
- Additional visualization charts
- Custom validation logic

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_client.py
```

The test suite includes:
- ✅ Server connectivity tests
- ✅ Data loading and preparation
- ✅ Model training and evaluation
- ✅ Prediction operations
- ✅ Cache management
- ✅ Complete workflow integration
- ✅ Performance benchmarks

## 📋 API Integration

The client integrates with the following APIs:

### Storage Server
- `GET /health` - Health check
- `GET /files` - List available files
- `POST /data/load` - Load data from file
- `POST /data/load_from_url` - Load data from URL
- `POST /data/prepare` - Prepare data for training
- `GET /data` - List datasets
- `GET /models` - List models

### Training Server
- `GET /health` - Health check
- `POST /train` - Train model
- `POST /evaluate` - Evaluate model

### Prediction Server
- `GET /health` - Health check
- `POST /load_model` - Load model to cache
- `POST /predict` - Make prediction
- `GET /cache` - Get cache status
- `POST /unload_model` - Unload model

## 🎨 User Interface

The client features a modern, responsive design with:
- **Clean Navigation**: Sidebar with clear section organization
- **Real-time Feedback**: Progress indicators and status updates
- **Interactive Charts**: Plotly visualizations for model performance
- **Professional Styling**: Custom CSS for enhanced user experience
- **Responsive Layout**: Optimized for different screen sizes

## 🔍 Troubleshooting

### Common Issues

1. **Servers Not Responding**
   - Check if Docker containers are running: `docker ps`
   - Restart services: `docker-compose restart`

2. **Import Errors**
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Connection Timeouts**
   - Verify server endpoints in configuration
   - Check firewall settings

4. **Data Upload Issues**
   - Ensure CSV files are in the correct format
   - Check file permissions in the Data directory

## 🚀 Performance

The client is optimized for:
- **Fast Loading**: Efficient API calls and caching
- **Responsive UI**: Streamlit's reactive framework
- **Memory Management**: Intelligent model loading/unloading
- **Batch Processing**: Efficient handling of large datasets

## 🔄 Updates

To update the client:
1. Pull latest changes
2. Update dependencies: `pip install -r requirements.txt --upgrade`
3. Restart the Streamlit application

## 📞 Support

For support and issues:
1. Check the troubleshooting section
2. Run the test suite to identify problems
3. Review server logs: `docker-compose logs`
4. Verify system requirements

---

**Built with ❤️ for the NaiveHub Machine Learning Platform**
