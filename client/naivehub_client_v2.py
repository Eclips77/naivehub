"""
NaiveHub Client - Professional ML Platform
==========================================
Clear workflow with beautiful UX and intuitive design.
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuration
STORAGE_URL = "http://localhost:8002"
TRAINER_URL = "http://localhost:8001"
PREDICTOR_URL = "http://localhost:8000"

# Custom CSS for beautiful design
def load_custom_css():
    st.markdown("""
    <style>
    /* Main styling */
    .main {
        padding: 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Step cards */
    .step-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e1e5e9;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .step-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .step-number {
        background: #667eea;
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        font-weight: bold;
    }
    
    /* Success/Error styling */
    .success-card {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .error-card {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
    </style>
    """, unsafe_allow_html=True)

class NaiveHubWorkflow:
    """Main workflow class for NaiveHub operations."""
    
    def __init__(self):
        self.storage_url = STORAGE_URL
        self.trainer_url = TRAINER_URL
        self.predictor_url = PREDICTOR_URL
        
        # Initialize session state with more comprehensive state management
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1
        if 'selected_file' not in st.session_state:
            st.session_state.selected_file = None
        if 'loaded_dataset' not in st.session_state:
            st.session_state.loaded_dataset = None
        if 'prepared_dataset' not in st.session_state:
            st.session_state.prepared_dataset = None
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = None
        if 'model_loaded_to_predictor' not in st.session_state:
            st.session_state.model_loaded_to_predictor = None
        if 'last_prediction_result' not in st.session_state:
            st.session_state.last_prediction_result = None
        if 'model_evaluation_result' not in st.session_state:
            st.session_state.model_evaluation_result = None
        if 'data_preparation_complete' not in st.session_state:
            st.session_state.data_preparation_complete = False
        if 'model_training_complete' not in st.session_state:
            st.session_state.model_training_complete = False
        if 'available_files_cache' not in st.session_state:
            st.session_state.available_files_cache = None
        if 'prepared_datasets_cache' not in st.session_state:
            st.session_state.prepared_datasets_cache = None
        if 'trained_models_cache' not in st.session_state:
            st.session_state.trained_models_cache = None
    
    def check_servers(self):
        """Check if all servers are running."""
        servers = {
            "🗄️ Storage": self.storage_url,
            "🎓 Training": self.trainer_url,
            "🔮 Prediction": self.predictor_url
        }
        
        all_online = True
        cols = st.columns(3)
        
        for i, (name, url) in enumerate(servers.items()):
            with cols[i]:
                try:
                    response = requests.get(f"{url}/health", timeout=3)
                    if response.status_code == 200:
                        st.success(f"{name}\n✅ Online")
                    else:
                        st.error(f"{name}\n❌ Error")
                        all_online = False
                except:
                    st.error(f"{name}\n🔴 Offline")
                    all_online = False
        
        return all_online
    
    def get_available_files(self):
        """Get available CSV files with caching."""
        # Use cached data if available to prevent unnecessary API calls
        if st.session_state.available_files_cache is not None:
            return st.session_state.available_files_cache
            
        try:
            response = requests.get(f"{self.storage_url}/files")
            if response.status_code == 200:
                result = response.json()
                st.session_state.available_files_cache = result
                return result
            return {"available_files": [], "count": 0}
        except:
            return {"available_files": [], "count": 0}
    
    def refresh_files_cache(self):
        """Force refresh of files cache."""
        st.session_state.available_files_cache = None
    
    def load_data_file(self, file_name):
        """Load data from file."""
        try:
            dataset_id = f"dataset_{datetime.now().strftime('%H%M%S')}"
            data = {"file_name": file_name, "dataset_id": dataset_id}
            response = requests.post(f"{self.storage_url}/data/load", json=data)
            if response.status_code == 200:
                return True, response.json()
            return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def prepare_data(self, dataset_id, target_column, train_size=0.7):
        """Prepare data for training."""
        try:
            data = {
                "dataset_id": dataset_id,
                "target_column": target_column,
                "train_size": train_size
            }
            response = requests.post(f"{self.storage_url}/data/prepare", json=data)
            if response.status_code == 200:
                return True, response.json()
            return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_prepared_datasets(self):
        """Get prepared datasets ready for training with caching."""
        # Use cached data if available
        if st.session_state.prepared_datasets_cache is not None:
            return st.session_state.prepared_datasets_cache
            
        try:
            response = requests.get(f"{self.storage_url}/data")
            if response.status_code == 200:
                data = response.json()
                prepared = [d for d in data.get('available_datasets', []) if d['type'] == 'prepared']
                st.session_state.prepared_datasets_cache = prepared
                return prepared
            return []
        except:
            return []
    
    def get_all_datasets(self):
        """Get all datasets (raw and prepared)."""
        try:
            response = requests.get(f"{self.storage_url}/data")
            if response.status_code == 200:
                data = response.json()
                return data.get('available_datasets', [])
            return []
        except:
            return []
    
    def refresh_datasets_cache(self):
        """Force refresh of datasets cache."""
        st.session_state.prepared_datasets_cache = None
    
    def train_model(self, dataset_id, target_column, model_name):
        """Train a new model."""
        try:
            data = {
                "dataset_id": dataset_id,
                "target_column": target_column,
                "model_name": model_name,
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "created_by": "NaiveHub Client"
                }
            }
            response = requests.post(f"{self.trainer_url}/train", json=data)
            if response.status_code == 200:
                return True, response.json()
            return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def evaluate_model(self, model_name):
        """Evaluate a trained model."""
        try:
            data = {"model_name": model_name}
            response = requests.post(f"{self.trainer_url}/evaluate", json=data)
            if response.status_code == 200:
                return True, response.json()
            return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_trained_models(self):
        """Get all trained models with caching."""
        # Use cached data if available
        if st.session_state.trained_models_cache is not None:
            return st.session_state.trained_models_cache
            
        try:
            response = requests.get(f"{self.storage_url}/models")
            if response.status_code == 200:
                models = response.json().get('available_models', [])
                st.session_state.trained_models_cache = models
                return models
            return []
        except:
            return []
    
    def refresh_models_cache(self):
        """Force refresh of models cache."""
        st.session_state.trained_models_cache = None
    
    def upload_file_to_server(self, uploaded_file, dataset_id=None):
        """Upload file from computer to server."""
        try:
            dataset_id = dataset_id or f"dataset_{datetime.now().strftime('%H%M%S')}"
            
            # First try the dedicated upload endpoint
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                data = {"dataset_id": dataset_id}
                
                response = requests.post(f"{self.storage_url}/data/upload", files=files, data=data)
                if response.status_code == 200:
                    return True, response.json()
                elif response.status_code == 404:
                    # Upload endpoint doesn't exist, try alternative method
                    return self._upload_via_temp_save(uploaded_file, dataset_id)
                else:
                    return False, response.json()
            except requests.exceptions.ConnectionError:
                return self._upload_via_temp_save(uploaded_file, dataset_id)
                
        except Exception as e:
            return False, {"error": str(e)}
    
    def _upload_via_temp_save(self, uploaded_file, dataset_id):
        """Fallback method: save file content and send as data."""
        try:
            # Read file content
            file_content = uploaded_file.getvalue().decode('utf-8')
            
            # Send as raw CSV data
            data = {
                "dataset_id": dataset_id,
                "csv_content": file_content,
                "filename": uploaded_file.name
            }
            
            response = requests.post(f"{self.storage_url}/data/load_csv_content", json=data)
            if response.status_code == 200:
                return True, response.json()
            else:
                # If this also fails, create a temporary approach
                return False, {"error": "Upload not supported by server. Please save file to Data directory manually."}
                
        except Exception as e:
            return False, {"error": f"Upload failed: {str(e)}"}
    
    def load_data_from_url(self, url, dataset_id=None):
        """Load data from URL with better error handling."""
        try:
            dataset_id = dataset_id or f"dataset_{datetime.now().strftime('%H%M%S')}"
            data = {"url": url, "dataset_id": dataset_id}
            
            # Add timeout and better error handling
            response = requests.post(f"{self.storage_url}/data/load_from_url", json=data, timeout=30)
            if response.status_code == 200:
                return True, response.json()
            elif response.status_code == 404:
                return False, {"error": "URL loading not supported by server"}
            else:
                return False, response.json()
        except requests.exceptions.Timeout:
            return False, {"error": "Request timed out. URL may be slow or inaccessible."}
        except requests.exceptions.ConnectionError:
            return False, {"error": "Cannot connect to server"}
        except Exception as e:
            return False, {"error": str(e)}
    
    def delete_dataset(self, dataset_id):
        """Delete a dataset."""
        try:
            response = requests.delete(f"{self.storage_url}/data/{dataset_id}")
            if response.status_code == 200:
                return True, response.json()
            return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def delete_model(self, model_name):
        """Delete a trained model."""
        try:
            response = requests.delete(f"{self.storage_url}/models/{model_name}")
            if response.status_code == 200:
                return True, response.json()
            return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_system_status(self):
        """Get comprehensive system status."""
        try:
            status = {}
            
            # Get datasets info
            datasets_response = requests.get(f"{self.storage_url}/data")
            if datasets_response.status_code == 200:
                datasets_data = datasets_response.json()
                status['datasets'] = {
                    'total': datasets_data.get('count', 0),
                    'raw': len([d for d in datasets_data.get('available_datasets', []) if d['type'] == 'raw']),
                    'prepared': len([d for d in datasets_data.get('available_datasets', []) if d['type'] == 'prepared'])
                }
            
            # Get models info
            models_response = requests.get(f"{self.storage_url}/models")
            if models_response.status_code == 200:
                models_data = models_response.json()
                status['models'] = {
                    'total': models_data.get('count', 0),
                    'models': models_data.get('available_models', [])
                }
            
            # Get files info
            files_response = requests.get(f"{self.storage_url}/files")
            if files_response.status_code == 200:
                files_data = files_response.json()
                status['files'] = {
                    'total': files_data.get('count', 0),
                    'files': files_data.get('available_files', [])
                }
            
            return status
        except Exception as e:
            return {"error": str(e)}
    
    def load_model_to_predictor(self, model_name):
        """Load model to prediction server."""
        try:
            data = {"model_name": model_name}
            response = requests.post(f"{self.predictor_url}/load_model", json=data)
            if response.status_code == 200:
                return True, response.json()
            return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def make_prediction(self, model_name, record):
        """Make a prediction."""
        try:
            data = {"model_name": model_name, "record": record}
            response = requests.post(f"{self.predictor_url}/predict", json=data)
            if response.status_code == 200:
                return True, response.json()
            return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}

def show_header():
    """Show beautiful header."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 NaiveHub ML Platform</h1>
        <h3>Professional Machine Learning Workflow</h3>
        <p>📊 Data → 🎓 Training → 🔮 Predictions</p>
    </div>
    """, unsafe_allow_html=True)

def show_system_status_modal(workflow):
    """Show system status in a modal-like display."""
    if st.button("📊 System Status", key="show_status", type="secondary"):
        st.session_state.show_status_modal = True
    
    if st.session_state.get('show_status_modal', False):
        st.markdown("---")
        st.markdown("### 📊 System Status Overview")
        
        with st.spinner("Loading system status..."):
            status = workflow.get_system_status()
        
        if 'error' not in status:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>📁 Data Status</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if 'datasets' in status:
                    st.metric("Total Datasets", status['datasets']['total'])
                    st.metric("Raw Datasets", status['datasets']['raw'])
                    st.metric("Prepared Datasets", status['datasets']['prepared'])
                
                if 'files' in status:
                    st.metric("CSV Files", status['files']['total'])
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>🤖 Models Status</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if 'models' in status:
                    st.metric("Trained Models", status['models']['total'])
                    
                    if status['models']['models']:
                        st.write("**Available Models:**")
                        for model in status['models']['models']:
                            st.write(f"• {model['model_name']} ({len(model['classes'])} classes)")
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>🗂️ Storage Status</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if 'files' in status:
                    total_size = sum(f['size_bytes'] for f in status['files']['files'])
                    st.metric("Total File Size", f"{total_size/1024:.1f} KB")
                    
                    if status['files']['files']:
                        st.write("**Recent Files:**")
                        for file_info in status['files']['files'][:3]:
                            st.write(f"• {file_info['file_name']} ({file_info['size_bytes']} bytes)")
        else:
            st.error(f"Error loading status: {status['error']}")
        
        if st.button("❌ Close Status", key="close_status"):
            st.session_state.show_status_modal = False
            st.rerun()
        
        st.markdown("---")

def show_step_1_data_loading(workflow):
    """Step 1: Data Loading and Preparation."""
    st.markdown("""
    <div class="step-card">
        <div class="step-header">
            <div class="step-number">1</div>
            📊 Step 1: Data Loading & Preparation
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different data loading methods
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Server Files", "💻 Upload File", "🌐 URL Import", "🗂️ Data Management"])
    
    with tab1:
        st.subheader("📁 Load from Server Data Directory")
        st.info("ℹ️ These are CSV files already available on the server in the Data directory.")
        
        # Get available files
        files_data = workflow.get_available_files()
        
        if files_data['count'] == 0:
            st.warning("⚠️ No CSV files found in server Data directory.")
            st.write("**Options:**")
            st.write("• Upload files using the 'Upload File' tab")
            st.write("• Add CSV files to the server's Data directory")
            if st.button("🔄 Refresh Files", key="refresh_files"):
                workflow.refresh_files_cache()
                st.rerun()
        else:
            file_options = {}
            for file_info in files_data['available_files']:
                file_options[f"{file_info['file_name']} ({file_info['size_bytes']} bytes)"] = file_info['file_name']
            
            # Restore previous selection if exists
            current_selection = 0
            if st.session_state.selected_file and not st.session_state.selected_file.startswith("Upload:"):
                for i, (label, filename) in enumerate(file_options.items()):
                    if filename == st.session_state.selected_file:
                        current_selection = i
                        break
            
            selected_file_label = st.selectbox(
                "Choose a CSV file from server:",
                list(file_options.keys()),
                index=current_selection,
                key="file_selector"
            )
            selected_file = file_options[selected_file_label]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"📄 Selected file: **{selected_file}**")
            
            with col2:
                if st.button("📥 Load Data", type="primary", key="load_data_local"):
                    with st.spinner("Loading data from server..."):
                        success, result = workflow.load_data_file(selected_file)
                        
                        if success:
                            st.session_state.loaded_dataset = result
                            st.session_state.selected_file = selected_file
                            workflow.refresh_datasets_cache()  # Refresh cache
                            st.success("✅ Data loaded successfully from server!")
                            st.rerun()
                        else:
                            st.error(f"❌ Loading failed: {result.get('error', 'Unknown error')}")
    
    with tab2:
        st.subheader("💻 Upload File from Your Computer")
        st.info("ℹ️ Upload a CSV file from your computer to the server and load it immediately.")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file from your computer:",
            type=['csv'],
            key="file_uploader",
            help="Select a CSV file from anywhere on your computer"
        )
        
        if uploaded_file is not None:
            # Show file info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"📁 **File Selected:** {uploaded_file.name}")
                st.write(f"**Size:** {uploaded_file.size} bytes")
                
                # Preview the file
                if st.checkbox("🔍 Preview file content", key="preview_upload"):
                    try:
                        # Read and display first few rows
                        uploaded_file.seek(0)  # Reset file pointer
                        preview_df = pd.read_csv(uploaded_file, nrows=5)
                        st.write("**Preview (first 5 rows):**")
                        st.dataframe(preview_df)
                        uploaded_file.seek(0)  # Reset again for actual upload
                    except Exception as e:
                        st.error(f"❌ Error reading file: {e}")
            
            with col2:
                dataset_id = st.text_input(
                    "Dataset ID (optional):",
                    placeholder="my_uploaded_data",
                    key="upload_dataset_id"
                )
                
                if st.button("🚀 Upload & Load", type="primary", key="upload_and_load"):
                    with st.spinner("Uploading file to server..."):
                        success, result = workflow.upload_file_to_server(uploaded_file, dataset_id)
                        
                        if success:
                            st.session_state.loaded_dataset = result
                            st.session_state.selected_file = f"Upload: {uploaded_file.name}"
                            workflow.refresh_datasets_cache()  # Refresh cache
                            workflow.refresh_files_cache()  # Refresh files cache too
                            st.success("✅ File uploaded and data loaded successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
    
    with tab3:
        st.subheader("🌐 Import from URL")
        st.info("ℹ️ Load data directly from a public URL (like GitHub raw files).")
        
        url = st.text_input(
            "CSV File URL:",
            placeholder="https://raw.githubusercontent.com/user/repo/main/data.csv",
            key="url_input",
            help="Enter a direct link to a CSV file"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            dataset_id = st.text_input(
                "Dataset ID (optional):",
                placeholder="my_url_dataset",
                key="url_dataset_id"
            )
        
        with col2:
            st.write("") # Spacer
            st.write("") # Spacer
            if st.button("🌐 Import from URL", type="primary", key="load_data_url"):
                if url:
                    with st.spinner("Downloading and processing from URL..."):
                        success, result = workflow.load_data_from_url(url, dataset_id)
                        
                        if success:
                            st.session_state.loaded_dataset = result
                            st.session_state.selected_file = f"URL: {url}"
                            workflow.refresh_datasets_cache()  # Refresh cache
                            st.success("✅ Data imported successfully from URL!")
                            st.rerun()
                        else:
                            st.error(f"❌ Import failed: {result.get('error', 'Unknown error')}")
                            st.write("**Possible issues:**")
                            st.write("• URL is not accessible or requires authentication")
                            st.write("• File is not in CSV format")
                            st.write("• Network connection issues")
                else:
                    st.warning("⚠️ Please enter a valid URL")
        
        # URL Examples
        with st.expander("💡 URL Examples & Tips"):
            st.write("**✅ Good examples:**")
            st.code("https://raw.githubusercontent.com/username/repo/main/data.csv")
            st.code("https://example.com/public/dataset.csv")
            st.code("https://drive.google.com/uc?export=download&id=FILE_ID")
            
            st.write("**❌ Won't work:**")
            st.write("• URLs that require login")
            st.write("• Google Drive/Dropbox share links (without proper export format)")
            st.write("• Password-protected files")
            
            st.write("**💡 Tips:**")
            st.write("• For GitHub: use 'Raw' button to get direct CSV link")
            st.write("• Test the URL in your browser first")
            st.write("• Make sure the file downloads directly (not shows a webpage)")
    
    with tab4:
        st.subheader("🗂️ Dataset Management")
        
        # Get all datasets
        all_datasets = workflow.get_all_datasets()
        
        if all_datasets:
            st.write(f"**Total datasets: {len(all_datasets)}**")
            
            # Show datasets with delete option
            for i, dataset in enumerate(all_datasets):
                with st.expander(f"📊 {dataset['dataset_id']} ({dataset['type']})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        if dataset['type'] == 'raw':
                            st.write(f"**Shape:** {dataset['shape']}")
                            st.write(f"**Columns:** {', '.join(dataset['columns'])}")
                            st.write(f"**Source:** {dataset['source']}")
                            st.write(f"**Loaded:** {dataset['loaded_at']}")
                        elif dataset['type'] == 'prepared':
                            st.write(f"**Train Shape:** {dataset['train_shape']}")
                            st.write(f"**Test Shape:** {dataset['test_shape']}")
                            st.write(f"**Target:** {dataset['target_column']}")
                            st.write(f"**Source:** {dataset['source']}")
                            st.write(f"**Prepared:** {dataset['prepared_at']}")
                    
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_dataset_{i}", help="Delete this dataset"):
                            # Confirmation in a separate container to avoid nesting
                            st.session_state[f'confirm_delete_{i}'] = True
                
                # Handle deletion confirmation outside the expander
                if st.session_state.get(f'confirm_delete_{i}', False):
                    st.warning(f"⚠️ Delete dataset '{dataset['dataset_id']}'?")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("✅ Yes, Delete", key=f"yes_delete_{i}"):
                            success, result = workflow.delete_dataset(dataset['dataset_id'])
                            if success:
                                st.success("✅ Dataset deleted!")
                                workflow.refresh_datasets_cache()
                                # Clear confirmation state
                                st.session_state[f'confirm_delete_{i}'] = False
                                st.rerun()
                            else:
                                st.error(f"❌ Delete failed: {result.get('error', 'Unknown error')}")
                    with col_no:
                        if st.button("❌ Cancel", key=f"cancel_delete_{i}"):
                            st.session_state[f'confirm_delete_{i}'] = False
                            st.rerun()
        else:
            st.info("📭 No datasets available yet.")
            st.write("**Get started:**")
            st.write("• Upload a CSV file from your computer")
            st.write("• Load data from a URL")
            st.write("• Use files from the server's Data directory")
    
    # Show loaded data info if available (appears in all tabs)
    if st.session_state.loaded_dataset:
        st.markdown("---")
        st.markdown("### 📊 Currently Loaded Dataset")
        result = st.session_state.loaded_dataset
        
        st.markdown("""
        <div class="success-card">
            <h4>✅ Data Loaded Successfully!</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Show data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Rows", result['shape'][0])
        with col2:
            st.metric("📋 Columns", result['shape'][1])
        with col3:
            st.metric("🗂️ Dataset ID", result['dataset_id'][:12] + "...")
        with col4:
            st.metric("📁 Source", st.session_state.selected_file[:15] + "..." if len(st.session_state.selected_file) > 15 else st.session_state.selected_file)
        
        # Show columns
        st.subheader("📋 Data Columns")
        cols = st.columns(min(len(result['columns']), 4))
        for i, col in enumerate(result['columns']):
            with cols[i % len(cols)]:
                st.write(f"**{col}**")
        
        # Show data preview
        if 'sample_data' in result and result['sample_data']:
            with st.expander("👀 Preview Data (first 5 rows)"):
                preview_df = pd.DataFrame(result['sample_data'][:5])
                st.dataframe(preview_df, use_container_width=True)
        
        # Data preparation section
        if not st.session_state.data_preparation_complete:
            st.subheader("🔧 Prepare Data for Training")
            
            col1, col2 = st.columns(2)
            with col1:
                target_column = st.selectbox(
                    "🎯 Select Target Column:",
                    result['columns'],
                    key="target_selector",
                    help="Choose the column you want to predict"
                )
            
            with col2:
                train_size = st.slider(
                    "📊 Training Set Size:",
                    0.5, 0.9, 0.7,
                    key="train_size_slider",
                    help="Percentage of data to use for training (rest for testing)"
                )
            
            if st.button("🚀 Prepare Data", type="primary", key="prepare_data"):
                with st.spinner("Preparing data for machine learning..."):
                    prep_success, prep_result = workflow.prepare_data(
                        result['dataset_id'], 
                        target_column, 
                        train_size
                    )
                    
                    if prep_success:
                        st.session_state.prepared_dataset = prep_result
                        st.session_state.data_preparation_complete = True
                        workflow.refresh_datasets_cache()  # Refresh cache
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ Preparation failed: {prep_result.get('error', 'Unknown error')}")
        else:
            # Show preparation completed
            st.markdown("""
            <div class="success-card">
                <h4>✅ Data Prepared Successfully!</h4>
                <p>Your data is ready for machine learning training!</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("➡️ Continue to Training", type="primary", key="continue_training"):
                st.session_state.current_step = 2
                st.rerun()

def show_step_2_model_training(workflow):
    """Step 2: Model Training."""
    st.markdown("""
    <div class="step-card">
        <div class="step-header">
            <div class="step-number">2</div>
            🎓 Step 2: Model Training
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for training and model management
    tab1, tab2 = st.tabs(["🚀 Train Model", "🗂️ Model Management"])
    
    with tab1:
        # Get prepared datasets
        prepared_datasets = workflow.get_prepared_datasets()
        
        if not prepared_datasets:
            st.warning("⚠️ No prepared datasets found. Please complete Step 1 first.")
            if st.button("⬅️ Back to Step 1"):
                st.session_state.current_step = 1
                st.rerun()
            return
        
        # Show prepared datasets
        st.subheader("📊 Available Prepared Datasets")
        
        dataset_options = {}
        for dataset in prepared_datasets:
            label = f"{dataset['dataset_id']} (Target: {dataset['target_column']})"
            dataset_options[label] = dataset
        
        # Remember previous selection
        current_selection = 0
        if hasattr(st.session_state, 'selected_dataset_for_training'):
            for i, label in enumerate(dataset_options.keys()):
                if label == st.session_state.selected_dataset_for_training:
                    current_selection = i
                    break
        
        selected_dataset_label = st.selectbox(
            "Choose a prepared dataset:",
            list(dataset_options.keys()),
            index=current_selection,
            key="dataset_selector"
        )
        selected_dataset = dataset_options[selected_dataset_label]
        st.session_state.selected_dataset_for_training = selected_dataset_label
        
        # Show dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏋️ Train Size", f"{selected_dataset['train_shape'][0]} rows")
        with col2:
            st.metric("🧪 Test Size", f"{selected_dataset['test_shape'][0]} rows")
        with col3:
            st.metric("🎯 Target", selected_dataset['target_column'])
        with col4:
            st.metric("📅 Prepared", selected_dataset['prepared_at'][:10])
        
        # Model configuration
        st.subheader("🤖 Model Configuration")
        
        # Remember model name if training was completed
        default_model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if hasattr(st.session_state, 'current_model_name'):
            default_model_name = st.session_state.current_model_name
        
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input(
                "Model Name:",
                value=default_model_name,
                key="model_name_input"
            )
            st.session_state.current_model_name = model_name
        
        with col2:
            description = st.text_area(
                "Description (optional):",
                placeholder="Describe your model...",
                key="model_description"
            )
        
        # Show training results if completed
        if st.session_state.model_training_complete and st.session_state.trained_model:
            result = st.session_state.trained_model
            
            st.markdown("""
            <div class="success-card">
                <h4>✅ Model Trained Successfully!</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Show training results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Classes", len(result['classes']))
            with col2:
                st.metric("📊 Features", len(result['features']))
            with col3:
                st.metric("⏱️ Training Time", f"{result.get('training_time_seconds', 0):.2f}s")
            
            # Show evaluation if available
            if st.session_state.model_evaluation_result:
                eval_result = st.session_state.model_evaluation_result
                accuracy = eval_result['accuracy'] * 100
                
                # Beautiful accuracy visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=accuracy,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Model Accuracy (%)"},
                    delta={'reference': 80},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Success message
                if accuracy >= 80:
                    st.success(f"🎉 Excellent! Model achieved {accuracy:.1f}% accuracy!")
                elif accuracy >= 60:
                    st.info(f"👍 Good! Model achieved {accuracy:.1f}% accuracy!")
                else:
                    st.warning(f"⚠️ Model achieved {accuracy:.1f}% accuracy. Consider more data or different features.")
            
            if st.button("➡️ Continue to Predictions", type="primary", key="continue_predictions"):
                st.session_state.current_step = 3
                st.rerun()
                
        else:
            # Training button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Train Model", type="primary", key="train_model", use_container_width=True):
                    if model_name:
                        with st.spinner("Training model... This may take a moment."):
                            # Progress bar animation
                            progress_bar = st.progress(0)
                            for i in range(100):
                                progress_bar.progress(i + 1)
                                time.sleep(0.01)
                            
                            success, result = workflow.train_model(
                                selected_dataset['dataset_id'],
                                selected_dataset['target_column'],
                                model_name
                            )
                            
                            if success:
                                st.session_state.trained_model = result
                                st.session_state.model_training_complete = True
                                workflow.refresh_models_cache()  # Refresh cache
                                
                                # Auto-evaluation
                                eval_success, eval_result = workflow.evaluate_model(model_name)
                                if eval_success:
                                    st.session_state.model_evaluation_result = eval_result
                                
                                st.rerun()  # Refresh to show results
                            else:
                                st.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.warning("⚠️ Please enter a model name.")
    
    with tab2:
        st.subheader("🗂️ Trained Models Management")
        
        # Get all trained models
        trained_models = workflow.get_trained_models()
        
        if trained_models:
            for i, model in enumerate(trained_models):
                with st.expander(f"🤖 {model['model_name']} ({len(model['classes'])} classes)"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Classes:** {', '.join(model['classes'])}")
                        st.write(f"**Features:** {len(model['features'])}")
                        st.write(f"**Feature List:** {', '.join(model['features'][:5])}{'...' if len(model['features']) > 5 else ''}")
                        if 'saved_at' in model:
                            st.write(f"**Created:** {model['saved_at']}")
                        
                        # Evaluate button
                        if st.button(f"📊 Evaluate Model", key=f"eval_model_{i}"):
                            with st.spinner("Evaluating model..."):
                                success, result = workflow.evaluate_model(model['model_name'])
                                if success:
                                    accuracy = result['accuracy'] * 100
                                    st.success(f"📊 **Accuracy: {accuracy:.1f}%**")
                                    
                                    # Show detailed results
                                    if 'classification_report' in result:
                                        st.text("Classification Report:")
                                        st.code(result['classification_report'])
                                else:
                                    st.error(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")
                    
                    with col2:
                        st.write("") # Spacer
                        if st.button("🗑️ Delete", key=f"delete_model_{i}", help="Delete this model"):
                            if st.button(f"⚠️ Confirm Delete", key=f"confirm_delete_model_{i}"):
                                success, result = workflow.delete_model(model['model_name'])
                                if success:
                                    st.success("✅ Model deleted!")
                                    workflow.refresh_models_cache()
                                    st.rerun()
                                else:
                                    st.error(f"❌ Delete failed: {result.get('error', 'Unknown error')}")
        else:
            st.info("No trained models available yet. Train your first model in the 'Train Model' tab!")

def show_step_3_predictions(workflow):
    """Step 3: Model Predictions."""
    st.markdown("""
    <div class="step-card">
        <div class="step-header">
            <div class="step-number">3</div>
            🔮 Step 3: Model Predictions
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get trained models
    trained_models = workflow.get_trained_models()
    
    if not trained_models:
        st.warning("⚠️ No trained models found. Please complete Step 2 first.")
        if st.button("⬅️ Back to Training"):
            st.session_state.current_step = 2
            st.rerun()
        return
    
    # Model selection
    st.subheader("🤖 Select Trained Model")
    
    model_options = {}
    for model in trained_models:
        label = f"{model['model_name']} ({len(model['classes'])} classes)"
        model_options[label] = model
    
    # Remember previous selection
    current_selection = 0
    if hasattr(st.session_state, 'selected_model_for_prediction'):
        for i, label in enumerate(model_options.keys()):
            if label == st.session_state.selected_model_for_prediction:
                current_selection = i
                break
    
    selected_model_label = st.selectbox(
        "Choose a trained model:",
        list(model_options.keys()),
        index=current_selection,
        key="prediction_model_selector"
    )
    selected_model = model_options[selected_model_label]
    st.session_state.selected_model_for_prediction = selected_model_label
    
    # Show model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Classes", len(selected_model['classes']))
    with col2:
        st.metric("📊 Features", len(selected_model['features']))
    with col3:
        st.metric("📅 Created", selected_model.get('saved_at', 'Unknown')[:10])
    
    # Load model to predictor
    st.subheader("📥 Load Model to Prediction Server")
    
    # Check if model is already loaded
    model_is_loaded = (st.session_state.model_loaded_to_predictor == selected_model['model_name'])
    
    if model_is_loaded:
        st.markdown("""
        <div class="success-card">
            <h4>✅ Model is loaded and ready for predictions!</h4>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("🚀 Load Model", type="primary", key="load_model"):
            with st.spinner("Loading model to prediction server..."):
                success, result = workflow.load_model_to_predictor(selected_model['model_name'])
                
                if success:
                    st.session_state.model_loaded_to_predictor = selected_model['model_name']
                    st.rerun()
                else:
                    st.error(f"❌ Failed to load model: {result.get('error', 'Unknown error')}")
    
    # Prediction interface (only show if model is loaded)
    if model_is_loaded:
        st.subheader("🎯 Make Predictions")
        
        # Show features
        features = selected_model['features']
        st.write("**Input Features:**")
        
        # Use session state to remember input values
        if 'prediction_inputs' not in st.session_state:
            st.session_state.prediction_inputs = {}
        
        prediction_record = {}
        cols = st.columns(min(2, len(features)))
        
        for i, feature in enumerate(features):
            with cols[i % len(cols)]:
                # Remember previous input
                previous_value = st.session_state.prediction_inputs.get(feature, "")
                prediction_record[feature] = st.text_input(
                    f"🔍 {feature}:",
                    value=previous_value,
                    key=f"pred_input_{feature}"
                )
                st.session_state.prediction_inputs[feature] = prediction_record[feature]
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Make Prediction", type="primary", key="make_prediction", use_container_width=True):
                if all(prediction_record.values()):
                    with st.spinner("Making prediction..."):
                        pred_success, pred_result = workflow.make_prediction(
                            selected_model['model_name'],
                            prediction_record
                        )
                        
                        if pred_success:
                            st.session_state.last_prediction_result = {
                                'prediction': pred_result['prediction'],
                                'input': prediction_record.copy(),
                                'model': selected_model['model_name']
                            }
                            st.rerun()
                        else:
                            st.error(f"❌ Prediction failed: {pred_result.get('error', 'Unknown error')}")
                else:
                    st.warning("⚠️ Please fill in all feature values.")
        
        # Show last prediction result if available
        if st.session_state.last_prediction_result:
            last_result = st.session_state.last_prediction_result
            prediction = last_result['prediction']
            
            # Beautiful result display
            st.markdown(f"""
            <div style="
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                margin: 2rem 0;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            ">
                <h2>🎯 Prediction Result</h2>
                <h1 style="font-size: 3rem; margin: 1rem 0;">{prediction}</h1>
                <p>Model: {last_result['model']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show input summary
            st.subheader("📋 Input Summary")
            input_df = pd.DataFrame([last_result['input']])
            st.dataframe(input_df, use_container_width=True)
        
        # Batch prediction
        st.subheader("📊 Batch Predictions")
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch predictions:",
            type=['csv'],
            key="batch_upload"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.write("**Preview:**")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("📊 Run Batch Predictions", key="batch_predict"):
                    with st.spinner("Processing batch predictions..."):
                        predictions = []
                        progress_bar = st.progress(0)
                        
                        for idx, (_, row) in enumerate(df.iterrows()):
                            record = row.to_dict()
                            success, result = workflow.make_prediction(
                                selected_model['model_name'],
                                record
                            )
                            predictions.append(result['prediction'] if success else "Error")
                            progress_bar.progress((idx + 1) / len(df))
                        
                        df['Prediction'] = predictions
                        
                        st.success("✅ Batch predictions completed!")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

def show_navigation_buttons():
    """Show navigation buttons."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🏠 Start Over", key="start_over"):
            # Reset session state selectively
            keys_to_reset = [
                'current_step', 'selected_file', 'loaded_dataset', 'prepared_dataset', 
                'trained_model', 'model_loaded_to_predictor', 'last_prediction_result',
                'model_evaluation_result', 'data_preparation_complete', 'model_training_complete',
                'available_files_cache', 'prepared_datasets_cache', 'trained_models_cache',
                'selected_dataset_for_training', 'selected_model_for_prediction',
                'current_model_name', 'prediction_inputs'
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_step = 1
            st.rerun()
    
    with col2:
        if st.session_state.current_step > 1:
            if st.button("⬅️ Previous Step", key="prev_step"):
                st.session_state.current_step -= 1
                st.rerun()
    
    with col3:
        if st.button("🔄 Refresh Cache", key="refresh_cache"):
            # Clear all caches to force refresh
            cache_keys = ['available_files_cache', 'prepared_datasets_cache', 'trained_models_cache']
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with col4:
        if st.session_state.current_step < 3:
            if st.button("➡️ Next Step", key="next_step"):
                st.session_state.current_step += 1
                st.rerun()
    
    with col5:
        if st.button("� Status Check", key="status_check"):
            # Just rerun to refresh status without clearing anything
            st.rerun()

def main():
    """Main application."""
    st.set_page_config(
        page_title="NaiveHub ML Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Show header
    show_header()
    
    # Initialize workflow
    workflow = NaiveHubWorkflow()
    
    # Check servers
    st.subheader("🔍 Server Status")
    
    # Add system status button
    col1, col2 = st.columns([3, 1])
    with col1:
        if not workflow.check_servers():
            st.error("❌ Some servers are offline. Please start all services with: `docker-compose up -d`")
            return
        st.success("✅ All servers are online and ready!")
    
    with col2:
        show_system_status_modal(workflow)
    
    # Show current step indicator
    steps = ["📊 Data Loading", "🎓 Model Training", "🔮 Predictions"]
    current = st.session_state.current_step
    
    progress_cols = st.columns(3)
    for i, step in enumerate(steps):
        with progress_cols[i]:
            if i + 1 == current:
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                    <b>{step}</b><br>
                    <small>Current Step</small>
                </div>
                """, unsafe_allow_html=True)
            elif i + 1 < current:
                st.markdown(f"""
                <div style="background: #28a745; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                    <b>✅ {step}</b><br>
                    <small>Completed</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #f8f9fa; color: #6c757d; padding: 1rem; border-radius: 10px; text-align: center;">
                    <b>{step}</b><br>
                    <small>Pending</small>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show current step
    if current == 1:
        show_step_1_data_loading(workflow)
    elif current == 2:
        show_step_2_model_training(workflow)
    elif current == 3:
        show_step_3_predictions(workflow)
    
    # Navigation
    st.markdown("---")
    show_navigation_buttons()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        <p><b>NaiveHub ML Platform v2.0</b> | Professional 3-Tier Architecture</p>
        <p>🗄️ Storage Server | 🎓 Training Server | 🔮 Prediction Server</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
