# 🎵 Spotify Big Data Analysis Dashboard

Interactive web dashboard for exploring music trends, ML insights, and data visualizations.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r dashboard_requirements.txt
```

Or install individually:

```bash
pip install streamlit pandas plotly Pillow pyarrow
```

### 2. Run the Dashboard

From the project root directory:

```bash
streamlit run dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## 📊 Dashboard Features

### 🏠 Overview

- Project summary and key metrics
- High-level statistics (10M tracks, 50K artists, 9 categories)
- ML models summary

### 📁 Dataset Overview

- Dataset composition visualizations
- Feature distribution summaries
- Category breakdown
- Key metrics dashboard

### 🤖 ML Models

- **Popularity Prediction Model**: XGBoost regression with comprehensive evaluation

  - Actual vs Predicted plots
  - Residual analysis
  - Feature importance rankings
  - Performance metrics (MAE, RMSE, R²)

- **Recommendation System**: K-Means clustering analysis
  - Cluster size distribution
  - PCA visualization
  - Silhouette analysis
  - Feature variance analysis

### 💡 Business Insights

- 9 deep-dive analytical visualizations:
  - Playlist DNA characteristics
  - Popularity drivers
  - Mood landscape
  - Genre comparisons
  - Artist profiles
  - Correlation insights

### 🔍 Interactive Data Explorer

- Filter by category and popularity
- Sort by different features
- View sample tracks
- Interactive charts:
  - Feature distributions
  - Scatter plot analysis
  - Category comparisons

### 📈 Raw Data Analysis

- Feature distributions
- Correlation matrices
- Audio features dashboard

### ⚙️ Technical Details

- Architecture overview
- Tech stack information
- Performance metrics
- Implementation details

### 📋 Executive Summary

- Generated insights report
- Key findings and recommendations

## 🎨 Dashboard Screenshots

The dashboard features:

- **Responsive design** - Works on desktop and tablet
- **Interactive charts** - Plotly-powered visualizations
- **Data filtering** - Explore subsets of data
- **Professional styling** - Spotify-inspired color scheme (green #1DB954)
- **Easy navigation** - Sidebar with clear sections

## 🛠️ Customization

### Change Sample Size

Edit `dashboard.py`, line ~60:

```python
if len(df) > 10000:  # Change this number
    df = df.sample(n=10000, random_state=42)
```

### Add New Visualizations

1. Generate PNG files in `visualizations/` folder
2. Add to relevant section in `dashboard.py`
3. Use `load_visualization("your_image.png")`

### Modify Theme Colors

Edit the CSS in the `st.markdown()` section at the top of `dashboard.py`:

```python
st.markdown("""
<style>
    /* Your custom CSS here */
</style>
""", unsafe_allow_html=True)
```

## 📋 Requirements

### Data Files Required

The dashboard expects:

- `data/processed/spotify_features.parquet` - Processed dataset
- `visualizations/*.png` - All generated visualization files

### System Requirements

- Python 3.8+
- 4GB+ RAM (for loading 10K sample)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🐛 Troubleshooting

### Dashboard won't start

```bash
# Make sure streamlit is installed
pip install streamlit

# Run with verbose output
streamlit run dashboard.py --logger.level=debug
```

### Data not loading

- Verify `data/processed/spotify_features.parquet` exists
- Check file permissions
- Try reducing sample size in the code

### Visualizations not showing

- Verify all PNG files are in `visualizations/` folder
- Check file names match exactly
- Ensure Pillow is installed: `pip install Pillow`

### Port already in use

```bash
# Use a different port
streamlit run dashboard.py --server.port 8502
```

## 🌐 Deployment Options

### Local Network Access

```bash
streamlit run dashboard.py --server.address 0.0.0.0
```

Access from other devices on your network using your IP address.

### Cloud Deployment

**Streamlit Cloud (Free):**

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy!

**Heroku:**

1. Create `Procfile`:
   ```
   web: streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0
   ```
2. Deploy with Heroku CLI

**AWS EC2:**

1. Upload dashboard files to EC2
2. Install dependencies
3. Run with nohup: `nohup streamlit run dashboard.py &`
4. Access via public IP

## 📝 Notes

- Dashboard uses cached data for performance (`@st.cache_data`)
- First load may take a few seconds
- Visualizations are loaded on-demand per section
- Interactive charts work best with 1K-10K samples

## 🎯 Tips for Best Experience

1. **Start with Overview** to understand the project scope
2. **Explore ML Models** to see model performance
3. **Use Data Explorer** for hands-on data exploration
4. **Check Business Insights** for actionable findings
5. **Review Technical Details** for implementation context

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Documentation](https://plotly.com/python/)
- [Project GitHub Repository](#) - Add your repo link

## 🤝 Contributing

To add new features to the dashboard:

1. Create visualizations in the pipeline
2. Add loading logic to `dashboard.py`
3. Create new section or tab as needed
4. Test locally before deploying

---

**Built with ❤️ using Streamlit | Powered by 10M Spotify tracks on AWS**
