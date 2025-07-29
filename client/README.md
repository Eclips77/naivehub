# 🧠 NaiveHub Client - Streamlit Dashboard

## Overview
A beautiful, modern Streamlit web interface for the NaiveHub Machine Learning platform. This client provides a complete GUI for data management, model training, and predictions with advanced session state management.

## ✨ Features

### 🎨 Modern UI/UX
- **Responsive Design**: Beautiful gradient-based interface that works on all screen sizes
- **Real-time Status**: Live server health monitoring
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **Intuitive Navigation**: Tab-based layout with clear sections

### 📊 Data Management
- **Multiple Data Sources**: Load from local files or URLs
- **Data Preparation**: Clean and split data for training
- **Dataset Overview**: View all datasets with detailed information
- **Sample Preview**: Inspect data before processing

### 🚀 Model Training
- **One-Click Training**: Train Naive Bayes models with automatic evaluation
- **Real-time Results**: Immediate accuracy feedback
- **Training History**: Track all training sessions
- **Model Comparison**: Visual accuracy comparisons

### 🔮 Smart Predictions
- **Interactive Prediction**: Easy feature input interface
- **Confidence Scores**: Detailed prediction confidence visualization
- **Prediction History**: Track all prediction attempts
- **Visual Feedback**: Plotly charts for confidence scores

### 🛡️ Session State Management
- **Persistent Data**: No data loss on page refreshes
- **State Preservation**: Maintains user selections and history
- **Smart Caching**: Efficient data management
- **Auto-refresh**: Intelligent data synchronization

## 🚀 Quick Start

### Prerequisites
Make sure all NaiveHub servers are running:
- **Storage Server**: `http://localhost:8002`
- **Training Server**: `http://localhost:8001`
- **Prediction Server**: `http://localhost:8003`

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the client
streamlit run naivehub_client_v2.py
```

### Usage
1. **Open your browser** to `http://localhost:8501`
2. **Check server status** in the sidebar (should show 🟢 for all servers)
3. **Load data** from the Data tab
4. **Prepare data** for training
5. **Train models** from the Training tab
6. **Make predictions** using trained models

## 📱 Interface Sections

### 📊 Data Tab
- **Load Data**: Import CSV files from local storage or URLs
- **Prepare Data**: Clean and split datasets for training
- **View Datasets**: Browse all available datasets

### 🚀 Training Tab
- **Select Dataset**: Choose from prepared datasets
- **Configure Training**: Set model name and parameters
- **Monitor Progress**: Real-time training feedback
- **View Results**: Accuracy and model details

### 🤖 Models Tab
- **Models Overview**: View all trained models
- **Accuracy Comparison**: Visual model performance comparison
- **Model Details**: Features, classes, and metadata

### 🔮 Predictions Tab
- **Select Model**: Choose from trained models
- **Input Features**: Easy-to-use feature input form
- **View Results**: Prediction with confidence scores
- **Visual Analysis**: Confidence score charts

## 🎯 Key Benefits

### For Data Scientists
- **Rapid Prototyping**: Quick model training and testing
- **Visual Analytics**: Immediate feedback with charts
- **Experiment Tracking**: Complete history of all activities
- **No Code Required**: GUI-based workflow

### For Business Users
- **User-Friendly**: No technical knowledge required
- **Real-time Insights**: Immediate prediction results
- **Visual Reports**: Easy-to-understand charts and metrics
- **Reliable**: Persistent session state prevents data loss

## 🔧 Configuration

### Server URLs
Update these in the client code if your servers run on different ports:
```python
STORAGE_URL = "http://localhost:8002"
TRAINER_URL = "http://localhost:8001"
PREDICTOR_URL = "http://localhost:8003"
```

### Styling
The interface uses custom CSS for modern styling. You can modify the styles in the `st.markdown()` section at the top of the file.

## 🛠️ Technical Details

### Session State Management
The client uses a sophisticated session state system that:
- Preserves all user data across page interactions
- Maintains training and prediction history
- Caches API responses for better performance
- Prevents data loss during Streamlit reruns

### Error Handling
- **Comprehensive Error Messages**: Clear feedback for all operations
- **Server Health Checks**: Automatic server connectivity monitoring
- **Graceful Degradation**: Interface remains functional even if servers are down
- **Timeout Management**: Prevents hanging requests

### Performance Optimization
- **Smart Caching**: Reduces unnecessary API calls
- **Lazy Loading**: Data loaded only when needed
- **Efficient Updates**: Targeted state updates
- **Responsive UI**: Fast interaction feedback

## 🔍 Troubleshooting

### Common Issues

1. **Servers Not Responding**
   - Check that all three servers are running
   - Verify port numbers in configuration
   - Check firewall settings

2. **Data Loading Fails**
   - Ensure CSV files are in the correct format
   - Check file permissions
   - Verify URL accessibility for external files

3. **Training Takes Too Long**
   - Check dataset size (large datasets take more time)
   - Monitor server logs for errors
   - Ensure sufficient system resources

4. **Predictions Not Working**
   - Verify model was trained successfully
   - Check feature names and types
   - Ensure all required features are provided

## 📈 Advanced Features

### Custom Styling
The interface includes:
- Gradient backgrounds
- Hover effects on buttons
- Custom metric cards
- Responsive design elements

### Data Visualization
- Interactive Plotly charts
- Real-time accuracy comparisons
- Confidence score visualizations
- Training history trends

### State Persistence
- Automatic session recovery
- Cross-tab data sharing
- History preservation
- Smart refresh mechanisms

## 🤝 Contributing

To extend the client:
1. Add new methods to the `APIClient` class for new endpoints
2. Create new sections in the `Dashboard` class
3. Update the session state management in `SessionState`
4. Add new visualizations using Plotly

---

**Built with ❤️ for the NaiveHub ML Platform**

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
