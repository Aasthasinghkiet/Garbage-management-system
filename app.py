import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from PIL import Image

# Set page config
st.set_page_config(
    page_title="GreenAI Locality Explorer",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stDownloadButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .css-1aumxhk {
        background-color: #e8f5e9;
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("greenai_locality_green_spaces1.csv")
        
        # Handle missing coordinates
        city_centroids = {
            'Bengaluru': (12.9716, 77.5946),
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.7041, 77.1025),
            'Chennai': (13.0827, 80.2707),
            'Hyderabad': (17.3850, 78.4867),
            'Kolkata': (22.5726, 88.3639),
            'Pune': (18.5204, 73.8567),
            'Noida': (28.5355, 77.3910),
            'Gurugram': (28.4595, 77.0266)
        }
        
        for idx, row in df.iterrows():
            if pd.isna(row['latitude']) and row['city'] in city_centroids:
                df.at[idx, 'latitude'], df.at[idx, 'longitude'] = city_centroids[row['city']]
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# Sidebar filters
st.sidebar.title("🌳 Filters")
selected_cities = st.sidebar.multiselect(
    "Select Cities",
    options=df['city'].unique(),
    default=df['city'].unique()
)

green_threshold = st.sidebar.slider(
    "Minimum Green Percentage",
    min_value=0,
    max_value=int(df['green_percent'].max()),
    value=0
)

suitable_only = st.sidebar.checkbox("Show only suitable for GreenAI", value=False)

# Apply filters
filtered_df = df[
    (df['city'].isin(selected_cities)) &
    (df['green_percent'] >= green_threshold)
]

if suitable_only:
    filtered_df = filtered_df[filtered_df['suitable_for_greenai'] == True]

# Main content
st.title("🌿 GreenAI Locality Explorer")
st.markdown("""
Explore green spaces across Indian cities and identify locations suitable for GreenAI projects.
""")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌎 Total Localities", len(filtered_df))
col2.metric("✅ Suitable for GreenAI", filtered_df['suitable_for_greenai'].sum())
col3.metric("📊 Avg Green %", f"{round(filtered_df['green_percent'].mean(), 2)}%")
col4.metric("🏙️ Cities Covered", len(filtered_df['city'].unique()))

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map", "📈 Analysis", "📊 City Comparison", "📋 Data"])

with tab1:
    st.subheader("Geographical Distribution")
    
    if not filtered_df.empty and 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
        # Create map
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        
        # Add markers with clustering
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in filtered_df.iterrows():
            if pd.notna(row['latitude']):
                color = 'green' if row['suitable_for_greenai'] else 'red'
                icon = 'leaf' if row['suitable_for_greenai'] else 'info-sign'
                
                popup_text = f"""
                <b>{row['locality']}</b><br>
                City: {row['city']}<br>
                Green %: {row['green_percent']}<br>
                Suitable: {'Yes' if row['suitable_for_greenai'] else 'No'}
                """
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=popup_text,
                    icon=folium.Icon(color=color, icon=icon, prefix='fa')
                ).add_to(marker_cluster)
        
        # Display map
        st_folium(m, width=1200, height=600)
    else:
        st.warning("No geographical data available for the selected filters")

with tab2:
    st.subheader("Data Analysis")
    
    # Distribution plot
    st.markdown("### Green Area Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_df[filtered_df['green_percent'] > 0]['green_percent'], bins=20, kde=True, ax=ax1)
    ax1.set_title('Distribution of Green Area Percentage')
    ax1.set_xlabel('Green Area Percentage')
    ax1.set_ylabel('Count')
    st.pyplot(fig1)
    
    # Top localities
    st.markdown("### Top Localities")
    top_n = st.slider("Number of top localities to show", 5, 20, 10, key="top_n_slider")
    
    top_localities = filtered_df[filtered_df['green_percent'] > 0].sort_values('green_percent', ascending=False).head(top_n)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.barplot(x='green_percent', y='locality', data=top_localities, hue='city', dodge=False, ax=ax2)
    ax2.set_title(f'Top {top_n} Localities by Green Percentage')
    ax2.set_xlabel('Green Area Percentage')
    ax2.set_ylabel('Locality')
    st.pyplot(fig2)

with tab3:
    st.subheader("City Comparison")
    
    if not filtered_df.empty:
        city_stats = filtered_df.groupby('city').agg({
            'green_percent': 'mean',
            'suitable_for_greenai': 'sum',
            'locality': 'count'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Average Green Percentage")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='green_percent', y='city', data=city_stats.sort_values('green_percent', ascending=False), 
                        palette='viridis', ax=ax3)
            ax3.set_xlabel('Average Green Percentage')
            ax3.set_ylabel('City')
            st.pyplot(fig3)
        
        with col2:
            st.markdown("### Suitable Localities Count")
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='suitable_for_greenai', y='city', data=city_stats.sort_values('suitable_for_greenai', ascending=False), 
                        palette='mako', ax=ax4)
            ax4.set_xlabel('Number of Suitable Localities')
            ax4.set_ylabel('City')
            st.pyplot(fig4)

with tab4:
    st.subheader("Detailed Data View")
    
    st.dataframe(
        filtered_df.sort_values('green_percent', ascending=False),
        column_config={
            "green_percent": st.column_config.NumberColumn(format="%.2f%%"),
            "latitude": st.column_config.NumberColumn(format="%.6f"),
            "longitude": st.column_config.NumberColumn(format="%.6f")
        },
        use_container_width=True,
        height=600
    )
    
    # Download button
    st.download_button(
        label="📥 Download Filtered Data",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='greenai_filtered_data.csv',
        mime='text/csv'
    )

# Footer
st.markdown("---")
st.markdown("""
**GreenAI Locality Explorer** - Analyzing urban green spaces for sustainable AI projects.  
*Data last updated: July 2023*
""")