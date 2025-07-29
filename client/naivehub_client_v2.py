import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import time
import os

# Page configuration
st.set_page_config(
    page_title="NaiveHub - ML Training Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .error-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #d63384;
        margin: 1rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #0c63e4;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
STORAGE_URL = "http://localhost:8002"
TRAINER_URL = "http://localhost:8001"
PREDICTOR_URL = "http://localhost:8003"

class SessionState:
    """Session state management to solve Streamlit rerun issues."""
    
    @staticmethod
    def init_state():
        """Initialize all session state variables."""
        if 'datasets' not in st.session_state:
            st.session_state.datasets = []
        if 'models' not in st.session_state:
            st.session_state.models = []
        if 'selected_dataset' not in st.session_state:
            st.session_state.selected_dataset = None
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = None
        if 'training_history' not in st.session_state:
            st.session_state.training_history = []
        if 'predictions_history' not in st.session_state:
            st.session_state.predictions_history = []
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'data_cache' not in st.session_state:
            st.session_state.data_cache = {}

class APIClient:
    """API client for communicating with NaiveHub servers."""
    
    @staticmethod
    def check_server_health(url: str, server_name: str) -> bool:
        """Check if a server is healthy."""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def get_datasets() -> List[Dict]:
        """Get all available datasets."""
        try:
            response = requests.get(f"{STORAGE_URL}/data", timeout=10)
            if response.status_code == 200:
                return response.json().get("available_datasets", [])
            return []
        except Exception as e:
            st.error(f"Error fetching datasets: {str(e)}")
            return []
    
    @staticmethod
    def get_models() -> List[Dict]:
        """Get all available models."""
        try:
            response = requests.get(f"{STORAGE_URL}/models", timeout=10)
            if response.status_code == 200:
                return response.json().get("available_models", [])
            return []
        except Exception as e:
            st.error(f"Error fetching models: {str(e)}")
            return []
    
    @staticmethod
    def get_files() -> List[Dict]:
        """Get available CSV files."""
        try:
            response = requests.get(f"{STORAGE_URL}/files", timeout=10)
            if response.status_code == 200:
                return response.json().get("available_files", [])
            return []
        except Exception as e:
            st.error(f"Error fetching files: {str(e)}")
            return []
    
    @staticmethod
    def load_data_from_file(file_name: str, dataset_id: Optional[str] = None) -> Dict:
        """Load data from file."""
        try:
            payload = {"file_name": file_name}
            if dataset_id:
                payload["dataset_id"] = dataset_id
            
            response = requests.post(f"{STORAGE_URL}/data/load", json=payload, timeout=30)
            return response.json()
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    @staticmethod
    def load_data_from_url(url: str, dataset_id: Optional[str] = None) -> Dict:
        """Load data from URL."""
        try:
            payload = {"url": url}
            if dataset_id:
                payload["dataset_id"] = dataset_id
            
            response = requests.post(f"{STORAGE_URL}/data/load_from_url", json=payload, timeout=30)
            return response.json()
        except Exception as e:
            raise Exception(f"Error loading data from URL: {str(e)}")
    
    @staticmethod
    def prepare_data(dataset_id: str, target_column: str, train_size: float = 0.7) -> Dict:
        """Prepare data for training."""
        try:
            payload = {
                "dataset_id": dataset_id,
                "target_column": target_column,
                "train_size": train_size
            }
            response = requests.post(f"{STORAGE_URL}/data/prepare", json=payload, timeout=30)
            return response.json()
        except Exception as e:
            raise Exception(f"Error preparing data: {str(e)}")
    
    @staticmethod
    def train_model(dataset_id: str, target_column: str, model_name: str) -> Dict:
        """Train a model."""
        try:
            payload = {
                "dataset_id": dataset_id,
                "target_column": target_column,
                "model_name": model_name
            }
            response = requests.post(f"{TRAINER_URL}/train", json=payload, timeout=120)
            return response.json()
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    @staticmethod
    def predict(model_name: str, features: Dict) -> Dict:
        """Make prediction."""
        try:
            payload = {
                "model_name": model_name,
                "features": features
            }
            response = requests.post(f"{PREDICTOR_URL}/predict", json=payload, timeout=30)
            return response.json()
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")

class Dashboard:
    """Main dashboard class."""
    
    @staticmethod
    def render_header():
        """Render the main header."""
        st.markdown('<h1 class="main-header">🧠 NaiveHub ML Platform</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    @staticmethod
    def render_server_status():
        """Render server status indicators."""
        col1, col2, col3 = st.columns(3)
        
        servers = [
            ("Storage", STORAGE_URL, col1),
            ("Trainer", TRAINER_URL, col2),
            ("Predictor", PREDICTOR_URL, col3)
        ]
        
        for name, url, col in servers:
            with col:
                is_healthy = APIClient.check_server_health(url, name)
                status = "🟢 Online" if is_healthy else "🔴 Offline"
                st.metric(f"{name} Server", status)
    
    @staticmethod
    def render_datasets_section():
        """Render datasets management section."""
        st.header("📊 Data Management")
        
        tab1, tab2, tab3 = st.tabs(["📁 Load Data", "🔧 Prepare Data", "📋 View Datasets"])
        
        with tab1:
            Dashboard.render_data_loading()
        
        with tab2:
            Dashboard.render_data_preparation()
        
        with tab3:
            Dashboard.render_datasets_view()
    
    @staticmethod
    def render_data_loading():
        """Render data loading interface."""
        st.subheader("Load Data")
        
        load_method = st.radio("Choose loading method:", ["📂 From File", "🌐 From URL"])
        
        if load_method == "📂 From File":
            files = APIClient.get_files()
            if files:
                file_names = [f["file_name"] for f in files]
                selected_file = st.selectbox("Select CSV file:", file_names)
                custom_id = st.text_input("Custom Dataset ID (optional):")
                
                if st.button("📥 Load Data"):
                    with st.spinner("Loading data..."):
                        try:
                            result = APIClient.load_data_from_file(selected_file, custom_id or None)
                            st.success(f"✅ {result['message']}")
                            
                            # Display data info
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Rows", result['shape'][0])
                            with col2:
                                st.metric("Columns", result['shape'][1])
                            
                            # Show sample data
                            st.subheader("Sample Data")
                            df_sample = pd.DataFrame(result['sample_data'])
                            st.dataframe(df_sample, use_container_width=True)
                            
                            # Refresh datasets
                            st.session_state.datasets = APIClient.get_datasets()
                            
                        except Exception as e:
                            st.error(f"❌ {str(e)}")
            else:
                st.info("No CSV files found in the Data directory.")
        
        else:  # From URL
            url = st.text_input("Enter CSV URL:")
            custom_id = st.text_input("Custom Dataset ID (optional):")
            
            if st.button("📥 Load from URL") and url:
                with st.spinner("Downloading and loading data..."):
                    try:
                        result = APIClient.load_data_from_url(url, custom_id or None)
                        st.success(f"✅ {result['message']}")
                        
                        # Display data info
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Rows", result['shape'][0])
                        with col2:
                            st.metric("Columns", result['shape'][1])
                        
                        # Show sample data
                        st.subheader("Sample Data")
                        df_sample = pd.DataFrame(result['sample_data'])
                        st.dataframe(df_sample, use_container_width=True)
                        
                        # Refresh datasets
                        st.session_state.datasets = APIClient.get_datasets()
                        
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
    
    @staticmethod
    def render_data_preparation():
        """Render data preparation interface."""
        st.subheader("Prepare Data for Training")
        
        datasets = APIClient.get_datasets()
        raw_datasets = [d for d in datasets if d.get("type") == "raw"]
        
        if raw_datasets:
            dataset_names = [f"{d['dataset_id']} ({d['shape'][0]} rows, {d['shape'][1]} cols)" for d in raw_datasets]
            selected_idx = st.selectbox("Select dataset to prepare:", range(len(dataset_names)), 
                                      format_func=lambda x: dataset_names[x])
            
            if selected_idx is not None:
                selected_dataset = raw_datasets[selected_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    target_column = st.selectbox("Select target column:", selected_dataset['columns'])
                with col2:
                    train_size = st.slider("Training set size:", 0.5, 0.9, 0.7, 0.05)
                
                if st.button("🔧 Prepare Data"):
                    with st.spinner("Preparing data..."):
                        try:
                            result = APIClient.prepare_data(selected_dataset['dataset_id'], target_column, train_size)
                            st.success(f"✅ {result['message']}")
                            
                            # Display preparation results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Training samples", result['train_shape'][0])
                            with col2:
                                st.metric("Test samples", result['test_shape'][0])
                            
                            # Refresh datasets
                            st.session_state.datasets = APIClient.get_datasets()
                            
                        except Exception as e:
                            st.error(f"❌ {str(e)}")
        else:
            st.info("No raw datasets available. Please load data first.")
    
    @staticmethod
    def render_datasets_view():
        """Render datasets overview."""
        st.subheader("Available Datasets")
        
        datasets = APIClient.get_datasets()
        st.session_state.datasets = datasets
        
        if datasets:
            for dataset in datasets:
                with st.expander(f"📊 {dataset['dataset_id']} ({dataset['type']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    if dataset['type'] == 'raw':
                        with col1:
                            st.metric("Rows", dataset['shape'][0])
                        with col2:
                            st.metric("Columns", dataset['shape'][1])
                        with col3:
                            st.metric("Type", "Raw Data")
                        
                        st.write("**Columns:**", ", ".join(dataset['columns']))
                        st.write("**Source:**", dataset['source'])
                    
                    else:  # prepared
                        with col1:
                            st.metric("Train samples", dataset['train_shape'][0])
                        with col2:
                            st.metric("Test samples", dataset['test_shape'][0])
                        with col3:
                            st.metric("Target", dataset['target_column'])
                        
                        st.write("**Source:**", dataset['source'])
                        st.write("**Prepared at:**", dataset['prepared_at'])
        else:
            st.info("No datasets available.")
    
    @staticmethod
    def render_training_section():
        """Render model training section."""
        st.header("🚀 Model Training")
        
        # Get prepared datasets
        datasets = APIClient.get_datasets()
        prepared_datasets = [d for d in datasets if d.get("type") == "prepared"]
        
        if prepared_datasets:
            col1, col2 = st.columns(2)
            
            with col1:
                dataset_names = [f"{d['dataset_id']} (target: {d['target_column']})" for d in prepared_datasets]
                selected_idx = st.selectbox("Select prepared dataset:", range(len(dataset_names)), 
                                          format_func=lambda x: dataset_names[x])
            
            with col2:
                model_name = st.text_input("Model name:", f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            if selected_idx is not None and model_name:
                selected_dataset = prepared_datasets[selected_idx]
                
                # Display dataset info
                with st.expander("📊 Dataset Information"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Training samples", selected_dataset['train_shape'][0])
                    with col2:
                        st.metric("Test samples", selected_dataset['test_shape'][0])
                    with col3:
                        st.metric("Target column", selected_dataset['target_column'])
                
                if st.button("🚀 Start Training", type="primary"):
                    with st.spinner("Training model... This may take a while."):
                        try:
                            result = APIClient.train_model(
                                selected_dataset['dataset_id'], 
                                selected_dataset['target_column'], 
                                model_name
                            )
                            
                            st.success(f"✅ {result['message']}")
                            
                            # Display training results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", result['accuracy'])
                            with col2:
                                st.metric("Features", len(result['features']))
                            with col3:
                                st.metric("Classes", len(result['classes']))
                            
                            # Store training history
                            training_record = {
                                "timestamp": datetime.now().isoformat(),
                                "model_name": model_name,
                                "dataset_id": selected_dataset['dataset_id'],
                                "accuracy": result['accuracy'],
                                "features": result['features'],
                                "classes": result['classes']
                            }
                            st.session_state.training_history.append(training_record)
                            
                            # Refresh models
                            st.session_state.models = APIClient.get_models()
                            
                            # Show classes and features
                            with st.expander("📋 Model Details"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Classes:**")
                                    for cls in result['classes']:
                                        st.write(f"• {cls}")
                                with col2:
                                    st.write("**Features:**")
                                    for feature in result['features']:
                                        st.write(f"• {feature}")
                            
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
        else:
            st.info("No prepared datasets available. Please prepare data first.")
    
    @staticmethod
    def render_models_section():
        """Render models overview section."""
        st.header("🤖 Models Overview")
        
        models = APIClient.get_models()
        st.session_state.models = models
        
        if models:
            # Models summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Models", len(models))
            with col2:
                models_with_accuracy = [m for m in models if m.get('metadata', {}).get('accuracy') is not None]
                avg_accuracy = sum([float(m['metadata']['accuracy']) for m in models_with_accuracy]) / len(models_with_accuracy) if models_with_accuracy else 0
                st.metric("Avg Accuracy", f"{avg_accuracy:.2%}" if avg_accuracy > 0 else "N/A")
            with col3:
                latest_model = max(models, key=lambda x: x.get('saved_at', ''), default=None)
                st.metric("Latest Model", latest_model['model_name'] if latest_model else "None")
            
            # Models table
            models_df = []
            for model in models:
                accuracy = model.get('metadata', {}).get('accuracy')
                models_df.append({
                    "Model Name": model['model_name'],
                    "Accuracy": f"{float(accuracy):.2%}" if accuracy else "N/A",
                    "Classes": len(model.get('classes', [])),
                    "Features": len(model.get('features', [])),
                    "Saved At": model.get('saved_at', '')[:19] if model.get('saved_at') else 'N/A'
                })
            
            df = pd.DataFrame(models_df)
            st.dataframe(df, use_container_width=True)
            
            # Accuracy visualization
            if models_with_accuracy:
                st.subheader("📈 Model Accuracy Comparison")
                
                fig = px.bar(
                    x=[m['model_name'] for m in models_with_accuracy],
                    y=[float(m['metadata']['accuracy']) * 100 for m in models_with_accuracy],
                    title="Model Accuracy Comparison",
                    labels={'x': 'Model Name', 'y': 'Accuracy (%)'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No models available. Train some models first!")
    
    @staticmethod
    def render_prediction_section():
        """Render prediction section."""
        st.header("🔮 Make Predictions")
        
        models = APIClient.get_models()
        
        if models:
            model_names = [m['model_name'] for m in models]
            selected_model_name = st.selectbox("Select model for prediction:", model_names)
            
            if selected_model_name:
                selected_model = next(m for m in models if m['model_name'] == selected_model_name)
                
                # Display model info
                with st.expander("🤖 Model Information"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        accuracy = selected_model.get('metadata', {}).get('accuracy')
                        st.metric("Accuracy", f"{float(accuracy):.2%}" if accuracy else "N/A")
                    with col2:
                        st.metric("Classes", len(selected_model.get('classes', [])))
                    with col3:
                        st.metric("Features", len(selected_model.get('features', [])))
                    
                    st.write("**Classes:**", ", ".join(selected_model.get('classes', [])))
                
                # Feature input
                st.subheader("📝 Enter Feature Values")
                features = {}
                
                # Create input fields for each feature
                feature_list = selected_model.get('features', [])
                if feature_list:
                    cols = st.columns(min(3, len(feature_list)))
                    for i, feature in enumerate(feature_list):
                        with cols[i % len(cols)]:
                            features[feature] = st.text_input(f"{feature}:", key=f"feature_{feature}")
                    
                    if st.button("🔮 Make Prediction", type="primary"):
                        if all(features.values()):
                            with st.spinner("Making prediction..."):
                                try:
                                    result = APIClient.predict(selected_model_name, features)
                                    
                                    # Display prediction result
                                    prediction = result.get('prediction', 'Unknown')
                                    confidence = result.get('confidence', {})
                                    
                                    st.success(f"🎯 **Prediction:** {prediction}")
                                    
                                    # Show confidence scores
                                    if confidence:
                                        st.subheader("📊 Confidence Scores")
                                        conf_df = pd.DataFrame([
                                            {"Class": cls, "Confidence": f"{score:.2%}"}
                                            for cls, score in confidence.items()
                                        ])
                                        st.dataframe(conf_df, use_container_width=True)
                                        
                                        # Confidence visualization
                                        fig = px.bar(
                                            conf_df, 
                                            x='Class', 
                                            y=[float(c.rstrip('%'))/100 for c in conf_df['Confidence']],
                                            title="Prediction Confidence",
                                            labels={'y': 'Confidence'}
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Store prediction history
                                    prediction_record = {
                                        "timestamp": datetime.now().isoformat(),
                                        "model_name": selected_model_name,
                                        "features": features,
                                        "prediction": prediction,
                                        "confidence": confidence
                                    }
                                    st.session_state.predictions_history.append(prediction_record)
                                    
                                except Exception as e:
                                    st.error(f"❌ Prediction failed: {str(e)}")
                        else:
                            st.warning("⚠️ Please fill in all feature values.")
                else:
                    st.error("❌ No features found for this model.")
        else:
            st.info("No models available. Train some models first!")
    
    @staticmethod
    def render_sidebar():
        """Render sidebar with navigation and controls."""
        with st.sidebar:
            st.image("https://via.placeholder.com/200x100/667eea/white?text=NaiveHub", width=200)
            st.markdown("---")
            
            # Server status
            st.subheader("🌐 Server Status")
            servers = [
                ("Storage", STORAGE_URL),
                ("Trainer", TRAINER_URL),
                ("Predictor", PREDICTOR_URL)
            ]
            
            for name, url in servers:
                is_healthy = APIClient.check_server_health(url, name)
                status = "🟢" if is_healthy else "🔴"
                st.write(f"{status} {name}")
            
            st.markdown("---")
            
            # Quick stats
            st.subheader("📊 Quick Stats")
            datasets = len(st.session_state.get('datasets', []))
            models = len(st.session_state.get('models', []))
            
            st.metric("Datasets", datasets)
            st.metric("Models", models)
            st.metric("Predictions", len(st.session_state.get('predictions_history', [])))
            
            st.markdown("---")
            
            # Refresh button
            if st.button("🔄 Refresh Data"):
                st.session_state.datasets = APIClient.get_datasets()
                st.session_state.models = APIClient.get_models()
                st.session_state.last_refresh = datetime.now()
                st.rerun()
            
            st.caption(f"Last refreshed: {st.session_state.get('last_refresh', datetime.now()).strftime('%H:%M:%S')}")
            
            st.markdown("---")
            
            # Clear history
            if st.button("🗑️ Clear History"):
                st.session_state.training_history = []
                st.session_state.predictions_history = []
                st.success("History cleared!")

def main():
    """Main application entry point."""
    # Initialize session state
    SessionState.init_state()
    
    # Render sidebar
    Dashboard.render_sidebar()
    
    # Main content
    Dashboard.render_header()
    Dashboard.render_server_status()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🚀 Training", "🤖 Models", "🔮 Predictions"])
    
    with tab1:
        Dashboard.render_datasets_section()
    
    with tab2:
        Dashboard.render_training_section()
    
    with tab3:
        Dashboard.render_models_section()
    
    with tab4:
        Dashboard.render_prediction_section()
    
    # Training history
    if st.session_state.get('training_history'):
        with st.expander("📈 Training History"):
            history_df = pd.DataFrame(st.session_state.training_history)
            st.dataframe(history_df, use_container_width=True)
    
    # Predictions history
    if st.session_state.get('predictions_history'):
        with st.expander("🔮 Prediction History"):
            pred_history = []
            for pred in st.session_state.predictions_history:
                pred_history.append({
                    "Timestamp": pred['timestamp'][:19],
                    "Model": pred['model_name'],
                    "Prediction": pred['prediction'],
                    "Features": str(pred['features'])[:50] + "..." if len(str(pred['features'])) > 50 else str(pred['features'])
                })
            
            if pred_history:
                pred_df = pd.DataFrame(pred_history)
                st.dataframe(pred_df, use_container_width=True)

if __name__ == "__main__":
    main()
