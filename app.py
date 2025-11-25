import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="DataLytics Pro | Advanced Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        color: yellow;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    
    .main-header p {
        color: rgba(115,125,155,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.95);
        padding: 12px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        color: black;
        font-weight: 600;
        font-size: 1rem;
        padding: 0 2rem;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.4);
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.25);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* File Uploader */
    .uploadedFile {
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 2rem;
        background: rgba(255,255,255,0.95);
    }
    
    /* Dataframe Styles */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Success/Warning/Error Messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 12px;
        padding: 1rem;
        font-weight: 500;
    }
    
    /* Expander Styles */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 10px;
        font-weight: 600;
        padding: 1rem;
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Metric Styles */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: blue;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: rgb(0, 191, 255);
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 2rem 0 1rem 0;
    }
    
    .section-header h2 {
        margin: 0;
        color: #1e3c72;
        font-weight: 700;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: blue;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.3);
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.5);
    }
    
    /* Checkbox and Radio */
    .stCheckbox, .stRadio {
        background: rgba(155,125,175,0.9);
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'encoded_data' not in st.session_state:
    st.session_state.encoded_data = None

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>📊 DataLytics Pro</h1>
    <p>Enterprise-Grade Data Processing & Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    st.markdown("---")
    
    if st.session_state.data is not None:
        st.success("✅ Dataset Loaded")
        df_shape = st.session_state.data.shape
        st.metric("Rows", df_shape[0])
        st.metric("Columns", df_shape[1])
    else:
        st.info("📁 No dataset loaded")
    
    st.markdown("---")
    st.markdown("### 📈 Processing Status")
    
    status_original = "✅" if st.session_state.data is not None else "⏳"
    status_cleaned = "✅" if st.session_state.cleaned_data is not None else "⏳"
    status_encoded = "✅" if st.session_state.encoded_data is not None else "⏳"
    
    st.markdown(f"{status_original} Original Data")
    st.markdown(f"{status_cleaned} Cleaned Data")
    st.markdown(f"{status_encoded} Encoded Data")
    
    st.markdown("---")
    st.markdown("### 🛠️ Quick Actions")
    if st.button("🔄 Reset All Data"):
        st.session_state.data = None
        st.session_state.cleaned_data = None
        st.session_state.encoded_data = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("[📖 Documentation](#)")
    st.markdown("[🎥 Video Tutorials](#)")
    st.markdown("[💬 Support](#)")

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Upload", 
    "🧹 Clean & Visualize", 
    "💡 Insights", 
    "✂️ Transform",
    "🔢 Encode & Export"
])

# ==================== TAB 1: DATA UPLOAD ====================
with tab1:
    st.markdown('<div class="section-header"><h2>📁 Data Upload Center</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Drag and drop your file here or click to browse",
            type=['csv', 'txt', 'json', 'tsv'],
            help="Supported formats: CSV, TXT, JSON, TSV"
        )
    
    with col2:
        st.markdown("### 📋 Supported Formats")
        st.markdown("✓ CSV Files")
        st.markdown("✓ TSV Files")
        st.markdown("✓ JSON Files")
        st.markdown("✓ Text Files")
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.tsv'):
                df = pd.read_csv(uploaded_file, sep='\t')
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            st.session_state.data = df
            st.success(f"✅ Successfully loaded: **{uploaded_file.name}**")
            
            # Metrics Row
            st.markdown("### 📊 Dataset Overview")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Rows", f"{df.shape[0]:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Columns", df.shape[1])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Duplicates", df.duplicated().sum())
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Missing", f"{df.isnull().sum().sum():,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col5:
                memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Size", f"{memory_mb:.2f} MB")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Data Preview
            st.markdown("### 🔍 Data Preview")
            st.dataframe(df.head(15), use_container_width=True, height=400)
            
            # Column Details
            st.markdown("### 📋 Column Details")
            col_info = pd.DataFrame({
                'Column': df.columns.tolist(),
                'Data Type': df.dtypes.astype(str).tolist(),
                'Non-Null Count': df.count().tolist(),
                'Null Count': df.isnull().sum().tolist(),
                'Unique Values': df.nunique().tolist(),
                'Sample Value': [str(df[col].iloc[0])[:50] for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True, height=300)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.info("👆 Please upload a dataset to begin analysis")

# ==================== TAB 2: CLEANING & VISUALIZATION ====================
with tab2:
    if st.session_state.data is not None:
        st.markdown('<div class="section-header"><h2>🧹 Data Cleaning Studio</h2></div>', unsafe_allow_html=True)
        df = st.session_state.data.copy()
        
        # Cleaning Options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            remove_duplicates = st.checkbox("🔄 Remove Duplicates", value=True)
        with col2:
            handle_missing = st.checkbox("🔧 Handle Missing Values", value=True)
        with col3:
            clean_text = st.checkbox("📝 Clean Text Data", value=True)
        with col4:
            remove_outliers = st.checkbox("📊 Remove Outliers", value=False)
        
        if st.button("🚀 Start Cleaning Process", type="primary"):
            cleaned_df = df.copy()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if remove_duplicates:
                status_text.text("Removing duplicates...")
                progress_bar.progress(25)
                before = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                st.success(f"✓ Removed {before - len(cleaned_df)} duplicate rows")
            
            if handle_missing:
                status_text.text("Handling missing values...")
                progress_bar.progress(50)
                for col in cleaned_df.columns:
                    if cleaned_df[col].dtype == 'object':
                        cleaned_df[col].fillna('Unknown', inplace=True)
                    else:
                        cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                st.success("✓ Missing values handled")
            
            if clean_text:
                status_text.text("Cleaning text data...")
                progress_bar.progress(75)
                for col in cleaned_df.select_dtypes(include=['object']).columns:
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
                    cleaned_df[col] = cleaned_df[col].str.replace(r'\s+', ' ', regex=True)
                st.success("✓ Text data cleaned")
            
            progress_bar.progress(100)
            status_text.text("Cleaning complete!")
            st.session_state.cleaned_data = cleaned_df
            st.balloons()
            
            # Comparison Metrics
            st.markdown("### 📊 Cleaning Results")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Rows", df.shape[0])
            with col2:
                st.metric("Cleaned Rows", cleaned_df.shape[0], delta=cleaned_df.shape[0]-df.shape[0])
            with col3:
                st.metric("Original Missing", df.isnull().sum().sum())
            with col4:
                st.metric("After Cleaning", cleaned_df.isnull().sum().sum())
        
        # Visualizations
        if st.session_state.cleaned_data is not None:
            st.markdown('<div class="section-header"><h2>📈 Data Visualizations</h2></div>', unsafe_allow_html=True)
            
            cleaned_df = st.session_state.cleaned_data
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()
            text_cols = cleaned_df.select_dtypes(include=['object']).columns.tolist()
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if len(numeric_cols) > 1:
                    st.markdown("#### 🔥 Correlation Heatmap")
                    corr_matrix = cleaned_df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, 
                                   text_auto='.2f', 
                                   aspect="auto",
                                   color_continuous_scale='RdBu_r',
                                   title="Feature Correlations")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                if len(numeric_cols) > 0:
                    st.markdown("#### 📊 Distribution Analysis")
                    selected_col = st.selectbox("Select column:", numeric_cols, key="dist")
                    fig = px.histogram(cleaned_df, x=selected_col, 
                                     marginal="box",
                                     title=f"Distribution of {selected_col}")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                if len(text_cols) > 0:
                    st.markdown("#### 📊 Category Distribution")
                    selected_col = st.selectbox("Select categorical column:", text_cols, key="cat")
                    value_counts = cleaned_df[selected_col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f"Top 10 Values in {selected_col}",
                               labels={'x': selected_col, 'y': 'Count'},
                               color=value_counts.values,
                               color_continuous_scale='viridis')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                if len(numeric_cols) > 0:
                    st.markdown("#### 📦 Outlier Detection")
                    selected_col = st.selectbox("Select column:", numeric_cols, key="box")
                    fig = px.box(cleaned_df, y=selected_col, points="outliers")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠ Please upload data in the Upload tab first")

# ==================== TAB 3: RECOMMENDATIONS ====================
with tab3:
    if st.session_state.data is not None:
        st.markdown('<div class="section-header"><h2>💡 AI-Powered Insights</h2></div>', unsafe_allow_html=True)
        
        df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
        
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Encoding Recommendations")
            
            if len(text_cols) > len(numeric_cols):
                st.info("📝 **Text-Heavy Dataset**")
                st.write("**Recommended:** TF-IDF or Bag of Words")
                st.write(f"**Target columns:** {', '.join(text_cols[:3])}")
            elif len(numeric_cols) > len(text_cols):
                st.info("🔢 **Numeric-Heavy Dataset**")
                st.write("**Recommended:** Min-Max or Standard Scaling")
                st.write(f"**Target columns:** {', '.join(numeric_cols[:3])}")
            else:
                st.info("🔀 **Mixed Data Types**")
                st.write("**Recommended:** Hybrid Approach")
            
            st.markdown("### 📋 Column-Wise Analysis")
            for col in text_cols[:5]:
                unique_vals = df[col].nunique()
                with st.expander(f"**{col}** ({unique_vals} unique)"):
                    if unique_vals < 10:
                        st.success("✅ **One-Hot Encoding** (Low cardinality)")
                    elif unique_vals < 50:
                        st.warning("⚠ **Label Encoding** (Medium cardinality)")
                    else:
                        st.error("⚡ **Advanced Encoding** (High cardinality)")
                    
                    sample_vals = df[col].value_counts().head(3)
                    st.write("**Top values:**", ", ".join([f"{k} ({v})" for k, v in sample_vals.items()]))
        
        with col2:
            st.markdown("### 📊 Data Quality Score")
            
            # Calculate quality score
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            
            completeness = ((total_cells - missing_cells) / total_cells) * 100
            uniqueness = ((df.shape[0] - duplicate_rows) / df.shape[0]) * 100
            overall_score = (completeness + uniqueness) / 2
            
            # Quality gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=overall_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Quality"},
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
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Completeness", f"{completeness:.1f}%")
            st.metric("Uniqueness", f"{uniqueness:.1f}%")
            
            st.markdown("### 🎨 Data Composition")
            fig = go.Figure(data=[go.Pie(
                labels=['Text', 'Numeric'],
                values=[len(text_cols), len(numeric_cols)],
                hole=0.5,
                marker=dict(colors=['#667eea', '#764ba2'])
            )])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠ Please upload data first")

# ==================== TAB 4: DELETE ROWS/COLUMNS ====================
with tab4:
    if st.session_state.data is not None:
        st.markdown('<div class="section-header"><h2>✂️ Data Transformation</h2></div>', unsafe_allow_html=True)
        
        df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🗑️ Column Management")
            cols_to_delete = st.multiselect("Select columns to remove:", df.columns.tolist())
            
            if cols_to_delete:
                st.warning(f"⚠ Removing {len(cols_to_delete)} column(s)")
            
            if st.button("Delete Columns", type="primary"):
                if cols_to_delete:
                    df = df.drop(columns=cols_to_delete)
                    if st.session_state.cleaned_data is not None:
                        st.session_state.cleaned_data = df
                    else:
                        st.session_state.data = df
                    st.success(f"✅ Deleted {len(cols_to_delete)} columns")
                    st.rerun()
        
        with col2:
            st.markdown("### 🗑️ Row Management")
            delete_method = st.radio("Deletion method:", ["By Index", "By Condition"])
            
            if delete_method == "By Index":
                row_indices = st.text_input("Enter indices (comma-separated):", placeholder="0,1,5,10")
                
                if st.button("Delete Rows", type="primary"):
                    if row_indices:
                        try:
                            indices = [int(i.strip()) for i in row_indices.split(',')]
                            df = df.drop(indices)
                            if st.session_state.cleaned_data is not None:
                                st.session_state.cleaned_data = df
                            else:
                                st.session_state.data = df
                            st.success(f"✅ Deleted {len(indices)} rows")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            else:
                condition_col = st.selectbox("Select column:", df.columns.tolist())
                condition_val = st.text_input("Value to remove:")
                
                if condition_val:
                    matching = df[df[condition_col].astype(str) == condition_val].shape[0]
                    st.info(f"📊 {matching} rows match")
                
                if st.button("Delete Matching Rows", type="primary"):
                    if condition_val:
                        df = df[df[condition_col].astype(str) != condition_val]
                        if st.session_state.cleaned_data is not None:
                            st.session_state.cleaned_data = df
                        else:
                            st.session_state.data = df
                        st.success("✅ Rows deleted")
                        st.rerun()
        
        st.markdown("### 🔍 Current Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        st.warning("⚠ Please upload data first")

# ==================== TAB 5: ENCODING & EXPORT ====================
with tab5:
    if st.session_state.cleaned_data is not None or st.session_state.data is not None:
        st.markdown('<div class="section-header"><h2>🔢 Encoding & Export Center</h2></div>', unsafe_allow_html=True)
        
        working_df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
        
        # Encoding Configuration
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            encoding_method = st.selectbox(
                "Encoding Technique:",
                ["None", "Label Encoding", "One-Hot Encoding", "Frequency Encoding",
                 "Bag of Words", "TF-IDF", "Min-Max Scaling", "Standard Scaling",
                 "Mapping Encoding", "Lambda Encoding"]
            )
        
        with col2:
            encoding_mode = st.radio("Mode:", ["Single", "Multiple"])
        
        with col3:
            if encoding_method not in ["None", "One-Hot Encoding"]:
                replace_column = st.checkbox("Replace original", value=True)
            else:
                replace_column = False
        
        if encoding_method != "None":
            if encoding_method in ["Min-Max Scaling", "Standard Scaling"]:
                available_cols = working_df.select_dtypes(include=['number']).columns.tolist()
            else:
                available_cols = working_df.columns.tolist()
            
            if encoding_mode == "Single":
                columns_to_encode = [st.selectbox("Select column:", available_cols)]
            else:
                columns_to_encode = st.multiselect("Select columns:", available_cols)
            
            # Special encoding parameters
            mapping_dict = None
            lambda_func = None
            
            if encoding_method == "Mapping Encoding":
                st.info("📝 Format: value1:0, value2:1")
                if columns_to_encode:
                    unique_vals = working_df[columns_to_encode[0]].unique()[:10]
                    st.write(f"**Sample values:** {', '.join(map(str, unique_vals))}")
                
                mapping_input = st.text_area("Enter mapping:", placeholder="male:0, female:1")
                
                if mapping_input:
                    try:
                        mapping_dict = {}
                        for pair in mapping_input.split(','):
                            key, value = pair.split(':')
                            mapping_dict[key.strip()] = eval(value.strip())
                        st.success(f"✓ Mapping ready: {mapping_dict}")
                    except:
                        st.error("Invalid format")
            
            elif encoding_method == "Lambda Encoding":
                st.info("🔧 Example: lambda x: x.upper()")
                lambda_input = st.text_input("Lambda function:", placeholder="lambda x: x * 2")
                
                if lambda_input:
                    try:
                        lambda_func = eval(lambda_input)
                        if callable(lambda_func):
                            st.success("✓ Valid lambda")
                    except:
                        st.error("Invalid lambda")
                        lambda_func = None
            
            if st.button("🚀 Apply Encoding", type="primary"):
                if columns_to_encode:
                    encoded_df = working_df.copy()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_cols = len(columns_to_encode)
                    
                    for idx, col in enumerate(columns_to_encode):
                        try:
                            status_text.text(f"Encoding {col}...")
                            progress = int((idx + 1) / total_cols * 100)
                            progress_bar.progress(progress)
                            
                            if encoding_method == "Label Encoding":
                                le = LabelEncoder()
                                encoded_col = le.fit_transform(encoded_df[col].astype(str))
                                if replace_column:
                                    encoded_df[col] = encoded_col
                                else:
                                    encoded_df[f'{col}_encoded'] = encoded_col
                            
                            elif encoding_method == "One-Hot Encoding":
                                one_hot = pd.get_dummies(encoded_df[col], prefix=col)
                                encoded_df = pd.concat([encoded_df, one_hot], axis=1)
                            
                            elif encoding_method == "Frequency Encoding":
                                freq = encoded_df[col].value_counts(normalize=True)
                                encoded_col = encoded_df[col].map(freq)
                                if replace_column:
                                    encoded_df[col] = encoded_col
                                else:
                                    encoded_df[f'{col}_freq'] = encoded_col
                            
                            elif encoding_method == "Min-Max Scaling":
                                scaler = MinMaxScaler()
                                encoded_col = scaler.fit_transform(encoded_df[[col]])
                                if replace_column:
                                    encoded_df[col] = encoded_col
                                else:
                                    encoded_df[f'{col}_minmax'] = encoded_col
                            
                            elif encoding_method == "Standard Scaling":
                                scaler = StandardScaler()
                                encoded_col = scaler.fit_transform(encoded_df[[col]])
                                if replace_column:
                                    encoded_df[col] = encoded_col
                                else:
                                    encoded_df[f'{col}_standard'] = encoded_col
                            
                            elif encoding_method == "Bag of Words":
                                vectorizer = CountVectorizer(max_features=10)
                                bow_matrix = vectorizer.fit_transform(encoded_df[col].astype(str))
                                bow_df = pd.DataFrame(bow_matrix.toarray(), 
                                                    columns=[f'{col}_bow{i}' for i in range(bow_matrix.shape[1])])
                                if replace_column:
                                    encoded_df = encoded_df.drop(columns=[col])
                                encoded_df = pd.concat([encoded_df.reset_index(drop=True), bow_df], axis=1)
                            
                            elif encoding_method == "TF-IDF":
                                vectorizer = TfidfVectorizer(max_features=10)
                                tfidf_matrix = vectorizer.fit_transform(encoded_df[col].astype(str))
                                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                                                      columns=[f'{col}_tfidf{i}' for i in range(tfidf_matrix.shape[1])])
                                if replace_column:
                                    encoded_df = encoded_df.drop(columns=[col])
                                encoded_df = pd.concat([encoded_df.reset_index(drop=True), tfidf_df], axis=1)
                            
                            elif encoding_method == "Mapping Encoding" and mapping_dict:
                                encoded_col = encoded_df[col].map(mapping_dict)
                                if replace_column:
                                    encoded_df[col] = encoded_col
                                else:
                                    encoded_df[f'{col}_mapped'] = encoded_col
                            
                            elif encoding_method == "Lambda Encoding" and lambda_func:
                                encoded_col = encoded_df[col].apply(lambda_func)
                                if replace_column:
                                    encoded_df[col] = encoded_col
                                else:
                                    encoded_df[f'{col}_lambda'] = encoded_col
                            
                            st.success(f"✓ Encoded '{col}'")
                        except Exception as e:
                            st.error(f"Error on '{col}': {str(e)}")
                    
                    status_text.text("Encoding complete!")
                    st.session_state.encoded_data = encoded_df
                    st.balloons()
                    
                    # Summary
                    st.markdown("### 📊 Encoding Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Columns", working_df.shape[1])
                    with col2:
                        st.metric("Encoded Columns", encoded_df.shape[1])
                    with col3:
                        st.metric("New Features", encoded_df.shape[1] - working_df.shape[1])
        
        # Visualization After Encoding
        if st.session_state.encoded_data is not None:
            st.markdown('<div class="section-header"><h2>📈 Encoded Data Analysis</h2></div>', unsafe_allow_html=True)
            
            encoded_df = st.session_state.encoded_data
            numeric_cols = encoded_df.select_dtypes(include=['number']).columns.tolist()
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if len(numeric_cols) > 1:
                    st.markdown("#### 🔥 Post-Encoding Correlation")
                    corr_matrix = encoded_df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix,
                                   text_auto='.2f',
                                   aspect="auto",
                                   color_continuous_scale='RdBu_r')
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)
                
                if len(numeric_cols) > 0:
                    st.markdown("#### 📊 Feature Distribution")
                    selected_col = st.selectbox("Select feature:", numeric_cols, key="dist_enc")
                    fig = px.histogram(encoded_df, x=selected_col, marginal="box")
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                if len(numeric_cols) > 1:
                    st.markdown("#### 🎯 Scatter Analysis")
                    col_x = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
                    col_y = st.selectbox("Y-axis:", numeric_cols, key="scatter_y")
                    fig = px.scatter(encoded_df, x=col_x, y=col_y,
                                   title=f"{col_x} vs {col_y}")
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)
                
                if len(numeric_cols) > 0:
                    st.markdown("#### 📦 Outlier Analysis")
                    selected_col = st.selectbox("Select feature:", numeric_cols, key="box_enc")
                    fig = px.box(encoded_df, y=selected_col, points="outliers")
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Data Preview and Stats
            st.markdown("### 🔍 Encoded Data Preview")
            st.dataframe(encoded_df.head(15), use_container_width=True, height=400)
            
            st.markdown("### 📊 Statistical Summary")
            st.dataframe(encoded_df.describe(), use_container_width=True, height=300)
        
        # Download Section
        st.markdown('<div class="section-header"><h2>📥 Export Data</h2></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.data is not None:
                csv = st.session_state.data.to_csv(index=False)
                st.download_button(
                    "📄 Download Original",
                    data=csv,
                    file_name="original_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.session_state.cleaned_data is not None:
                csv = st.session_state.cleaned_data.to_csv(index=False)
                st.download_button(
                    "🧹 Download Cleaned",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            if st.session_state.encoded_data is not None:
                csv = st.session_state.encoded_data.to_csv(index=False)
                st.download_button(
                    "🔢 Download Encoded",
                    data=csv,
                    file_name="encoded_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    else:
        st.warning("⚠ Please upload and clean data first")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 15px; margin-top: 2rem;'>
    <h3 style='color: #1e3c72; margin: 0;'>🎨 DataLytics Pro</h3>
    <p style='color: #666; margin: 0.5rem 0 0 0;'>Enterprise-Grade Data Analytics Platform | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)