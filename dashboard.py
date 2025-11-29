#!/usr/bin/env python3
"""
🎵 Spotify Big Data Analysis Dashboard
Interactive dashboard for exploring music trends and ML insights
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from PIL import Image

# Add utils to path
sys.path.append(str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Spotify Big Data Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1DB954 0%, #191414 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1DB954;
    }
    .section-header {
        color: #1DB954;
        font-size: 2rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    """Load the processed Spotify data"""
    data_path = (
        Path(__file__).parent / "data" / "processed" / "spotify_features.parquet"
    )

    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.info("Please run the data pipeline first to generate the processed data.")
        return None

    # Load a sample for dashboard (10k rows for speed)
    df = pd.read_parquet(data_path)
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)

    return df


def load_visualization(image_name):
    """Load a visualization image"""
    viz_path = Path(__file__).parent / "visualizations" / image_name

    if viz_path.exists():
        return Image.open(viz_path)
    return None


def show_overview_section():
    """Display project overview"""
    st.markdown(
        '<h1 class="main-header">🎵 Spotify Big Data Analysis Dashboard</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    ### 📊 Project Overview
    
    This dashboard presents a comprehensive analysis of **10 million Spotify tracks** using big data techniques 
    on AWS EC2, including:
    
    - 🧹 **Data Cleaning & ETL** - Processing 10M+ rows with chunked operations
    - 🎨 **Feature Engineering** - Creating mood scores, tempo categories, and engagement metrics
    - 🤖 **Machine Learning** - Popularity prediction (XGBoost) and music recommendations (K-Means clustering)
    - 📈 **Data Visualization** - 23+ comprehensive visualizations
    - ☁️ **Cloud Infrastructure** - AWS S3 + EC2 pipeline
    """
    )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tracks", "10,000,000", help="Total tracks processed")

    with col2:
        st.metric("Unique Artists", "~50,000", help="Unique artists in dataset")

    with col3:
        st.metric("Playlist Categories", "9", help="Different music categories")

    with col4:
        st.metric("ML Models", "2", help="Trained machine learning models")


def show_dataset_overview():
    """Display dataset composition visualizations"""
    st.markdown(
        '<h2 class="section-header">📁 Dataset Overview</h2>', unsafe_allow_html=True
    )

    st.markdown(
        """
    Comprehensive overview of the dataset composition, distribution, and key statistics.
    """
    )

    # Load overview visualizations
    col1, col2 = st.columns(2)

    with col1:
        img1 = load_visualization("overview_1_dataset_composition.png")
        if img1:
            st.image(
                img1,
                caption="Dataset Composition - Track counts, artists, years, popularity",
                use_container_width=True,
            )

    with col2:
        img2 = load_visualization("overview_2_feature_summaries.png")
        if img2:
            st.image(
                img2,
                caption="Feature Summaries - Distribution of audio features",
                use_container_width=True,
            )

    col3, col4 = st.columns(2)

    with col3:
        img3 = load_visualization("overview_3_category_pie.png")
        if img3:
            st.image(
                img3,
                caption="Category Distribution - Breakdown by playlist type",
                use_container_width=True,
            )

    with col4:
        img4 = load_visualization("overview_4_key_metrics.png")
        if img4:
            st.image(
                img4,
                caption="Key Metrics Dashboard - Comprehensive statistics",
                use_container_width=True,
            )


def show_ml_models():
    """Display ML model evaluation results"""
    st.markdown(
        '<h2 class="section-header">🤖 Machine Learning Models</h2>',
        unsafe_allow_html=True,
    )

    # Model tabs
    tab1, tab2 = st.tabs(["🎯 Popularity Prediction", "🎵 Recommendation System"])

    with tab1:
        st.markdown(
            """
        ### Popularity Prediction Model (XGBoost Regression)
        
        Predicts track popularity (0-100) based on audio features and engineered metrics.
        
        **Key Metrics:**
        - **MAE**: 0.105 - Average prediction error
        - **RMSE**: 0.177 - Error magnitude
        - **Top Features**: hype_score, chill_score, acousticness, energy
        """
        )

        img = load_visualization("ml_eval_1_popularity_model.png")
        if img:
            st.image(
                img,
                caption="Popularity Model Evaluation - Comprehensive metrics",
                use_container_width=True,
            )

    with tab2:
        st.markdown(
            """
        ### Recommendation System (K-Means Clustering)
        
        Groups similar tracks into 50 clusters for music recommendations based on audio characteristics.
        
        **Key Metrics:**
        - **Clusters**: 50 distinct music groups
        - **Silhouette Score**: 0.020 - Cluster quality
        - **Key Features**: tempo, instrumentalness, energy, danceability
        """
        )

        img = load_visualization("ml_eval_2_recommendation_model.png")
        if img:
            st.image(
                img,
                caption="Recommendation Model Evaluation - Clustering analysis",
                use_container_width=True,
            )


def show_business_insights():
    """Display business intelligence visualizations"""
    st.markdown(
        '<h2 class="section-header">💡 Business Insights</h2>', unsafe_allow_html=True
    )

    st.markdown(
        """
    Deep-dive analysis into music trends, popularity drivers, and actionable business insights.
    """
    )

    # Create 3x3 grid for insights
    insights = [
        ("insight_1_playlist_dna.png", "Playlist DNA - Category characteristics"),
        (
            "insight_2_popularity_drivers.png",
            "Popularity Drivers - Key success factors",
        ),
        ("insight_3_mood_landscape.png", "Mood Landscape - Emotional distribution"),
        ("insight_4_workout_vs_sleep.png", "Workout vs Sleep - Genre comparison"),
        (
            "insight_5_popularity_sweet_spot.png",
            "Popularity Sweet Spot - Optimal ranges",
        ),
        ("insight_6_follower_impact.png", "Follower Impact - Playlist size effects"),
        ("insight_7_tempo_zones.png", "Tempo Zones - BPM distribution"),
        ("insight_8_top_artists_profiles.png", "Top Artists - Artist profiles"),
        ("insight_9_correlation_insights.png", "Correlations - Feature relationships"),
    ]

    for i in range(0, len(insights), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(insights):
                img_name, caption = insights[i + j]
                img = load_visualization(img_name)
                if img:
                    with col:
                        st.image(img, caption=caption, use_container_width=True)


def show_data_explorer(df):
    """Interactive data explorer"""
    st.markdown(
        '<h2 class="section-header">🔍 Interactive Data Explorer</h2>',
        unsafe_allow_html=True,
    )

    if df is None:
        st.warning("Data not available. Please ensure the pipeline has been run.")
        return

    st.markdown(
        f"""
    Explore a **sample of {len(df):,} tracks** from the dataset interactively.
    """
    )

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        categories = ["All"] + sorted(df["playlist_category"].unique().tolist())
        selected_category = st.selectbox("Playlist Category", categories)

    with col2:
        min_popularity = st.slider("Minimum Popularity", 0, 100, 0)

    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["popularity_score", "energy", "danceability", "tempo", "valence"],
        )

    # Filter data
    filtered_df = df.copy()
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["playlist_category"] == selected_category]
    filtered_df = filtered_df[filtered_df["popularity_score"] >= min_popularity]
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)

    st.markdown(f"**Showing {len(filtered_df):,} tracks**")

    # Display sample data
    st.dataframe(
        filtered_df[
            [
                "track_name",
                "artist_name",
                "playlist_category",
                "popularity_score",
                "energy",
                "danceability",
                "tempo",
                "valence",
            ]
        ].head(100),
        use_container_width=True,
        height=400,
    )

    # Interactive visualizations
    st.markdown("### 📊 Interactive Charts")

    chart_tab1, chart_tab2, chart_tab3 = st.tabs(
        ["Feature Distribution", "Scatter Analysis", "Category Comparison"]
    )

    with chart_tab1:
        feature = st.selectbox(
            "Select Feature",
            ["energy", "danceability", "valence", "tempo", "acousticness", "loudness"],
        )
        fig = px.histogram(
            filtered_df,
            x=feature,
            nbins=50,
            title=f"{feature.title()} Distribution",
            color_discrete_sequence=["#1DB954"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with chart_tab2:
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(
                "X-Axis",
                ["energy", "danceability", "valence", "tempo", "acousticness"],
                key="x",
            )
        with col2:
            y_axis = st.selectbox(
                "Y-Axis",
                ["popularity_score", "energy", "danceability", "valence", "tempo"],
                key="y",
            )

        fig = px.scatter(
            filtered_df.sample(min(1000, len(filtered_df))),
            x=x_axis,
            y=y_axis,
            color="playlist_category",
            title=f"{x_axis.title()} vs {y_axis.title()}",
            opacity=0.6,
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_tab3:
        feature_to_compare = st.selectbox(
            "Feature to Compare",
            ["energy", "danceability", "valence", "tempo", "popularity_score"],
            key="compare",
        )

        fig = px.box(
            df,
            x="playlist_category",
            y=feature_to_compare,
            title=f"{feature_to_compare.title()} by Category",
            color="playlist_category",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def show_raw_analysis():
    """Display raw data analysis visualizations"""
    st.markdown(
        '<h2 class="section-header">📈 Raw Data Analysis</h2>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        img1 = load_visualization("raw_data_distributions.png")
        if img1:
            st.image(img1, caption="Feature Distributions", use_container_width=True)

        img2 = load_visualization("audio_features_dashboard.png")
        if img2:
            st.image(img2, caption="Audio Features Dashboard", use_container_width=True)

    with col2:
        img3 = load_visualization("raw_data_correlations.png")
        if img3:
            st.image(img3, caption="Feature Correlations", use_container_width=True)

        img4 = load_visualization("correlation_heatmap.png")
        if img4:
            st.image(img4, caption="Correlation Heatmap", use_container_width=True)


def show_technical_details():
    """Display technical implementation details"""
    st.markdown(
        '<h2 class="section-header">⚙️ Technical Implementation</h2>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### 🏗️ Architecture
        
        **Data Pipeline:**
        - **Data Generation**: Synthetic 10M tracks locally
        - **Storage**: Amazon S3 bucket
        - **Processing**: AWS EC2 (m5.2xlarge - 32GB RAM)
        - **Orchestration**: Python master pipeline
        
        **Data Processing:**
        - **Cleaning**: Chunked processing (500K rows/chunk)
        - **Feature Engineering**: 15+ derived features
        - **ML Training**: XGBoost + K-Means on 10M samples
        - **Evaluation**: Comprehensive metrics & visualizations
        """
        )

    with col2:
        st.markdown(
            """
        ### 🛠️ Tech Stack
        
        **Languages & Libraries:**
        - Python 3.11
        - Pandas, NumPy (data processing)
        - XGBoost, Scikit-learn (ML)
        - Matplotlib, Seaborn (visualization)
        - Streamlit (dashboard)
        
        **AWS Services:**
        - S3 (data storage)
        - EC2 (compute)
        
        **Features:**
        - Memory-efficient chunked processing
        - Error handling & logging
        - Resume capabilities
        - Comprehensive validation
        """
        )

    # Performance metrics
    st.markdown("### ⚡ Performance Metrics")

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    with perf_col1:
        st.metric("Data Cleaning", "~45 min", help="For 10M rows with chunking")

    with perf_col2:
        st.metric("Feature Engineering", "~30 min", help="Creating derived features")

    with perf_col3:
        st.metric("ML Training", "~60 min", help="Both models combined")

    with perf_col4:
        st.metric("Total Pipeline", "~3 hours", help="End-to-end execution")


def show_executive_summary():
    """Display executive summary from results"""
    st.markdown(
        '<h2 class="section-header">📋 Executive Summary</h2>', unsafe_allow_html=True
    )

    summary_path = Path(__file__).parent / "visualizations" / "EXECUTIVE_SUMMARY.txt"

    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = f.read()

        st.text_area("", summary, height=400, disabled=True)
    else:
        st.info(
            "Executive summary not available. Run the insights analysis to generate it."
        )


def main():
    """Main dashboard application"""

    # Sidebar navigation
    st.sidebar.title("🎵 Navigation")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select Section",
        [
            "🏠 Overview",
            "📁 Dataset Overview",
            "🤖 ML Models",
            "💡 Business Insights",
            "🔍 Data Explorer",
            "📈 Raw Analysis",
            "⚙️ Technical Details",
            "📋 Executive Summary",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    ### About This Project
    
    Comprehensive big data analysis of 10 million Spotify tracks using AWS cloud infrastructure.
    
    **Key Features:**
    - Data processing on EC2
    - ML models (XGBoost + K-Means)
    - 23+ visualizations
    - Interactive exploration
    
    ---
    **Tech Stack:**
    Python • AWS • Pandas • XGBoost • Streamlit
    """
    )

    # Load data once
    df = None
    if page == "🔍 Data Explorer":
        with st.spinner("Loading data..."):
            df = load_data()

    # Route to selected page
    if page == "🏠 Overview":
        show_overview_section()
    elif page == "📁 Dataset Overview":
        show_dataset_overview()
    elif page == "🤖 ML Models":
        show_ml_models()
    elif page == "💡 Business Insights":
        show_business_insights()
    elif page == "🔍 Data Explorer":
        show_data_explorer(df)
    elif page == "📈 Raw Analysis":
        show_raw_analysis()
    elif page == "⚙️ Technical Details":
        show_technical_details()
    elif page == "📋 Executive Summary":
        show_executive_summary()

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 1rem;">
        🎵 Spotify Big Data Analysis Dashboard | Built with Streamlit | Data processed on AWS EC2
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
