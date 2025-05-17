import streamlit as st
import os
import subprocess

st.set_page_config(
    page_title="Clinical Trial Stats App Launcher",
    page_icon="🧪",
    layout="centered"
)

st.markdown("""
<style>
    .app-card {
        padding: 20px;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
        border-left: 5px solid #1976d2;
        transition: transform 0.2s ease-in-out;
    }
    .app-card:hover {
        transform: scale(1.01);
    }
    .header {
        text-align: center;
        margin-bottom: 40px;
    }
    .subtitle {
        font-size: 1.2em;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header'>Clinical Trial Statistics Suite</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Choose an application to launch</p>", unsafe_allow_html=True)

# Define app paths
main_app = "Clinical_trial_analysis_app_v1.py"
dashboard_app = "interactive_dashboard.py"

# Simple function to launch an app
def launch_app(app_path):
    try:
        # Just run the app and let Streamlit handle everything else
        subprocess.Popen(["streamlit", "run", app_path])
        return True
    except Exception as e:
        st.error(f"Error launching app: {str(e)}")
        return False

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='app-card'>
        <h3>📊 Statistical Models for Clinical Trials</h3>
        <p>Explore the most common models for analyzing clinical trial data.  Use the examples or upload your own data.</p>
        <ul>
            <li>Comprehensive suite of statistical models</li>
            <li>Real world examples of trial designs, with mock data</li>
            <li>Data import and export</li>
            <li>Static visualizations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Launch Clinical Trial Analysis App"):
        with st.spinner("Starting Clinical Trial Analysis App..."):
            success = launch_app(main_app)
            if success:
                st.success("App launched! A new browser tab should open automatically.")

with col2:
    st.markdown("""
    <div class='app-card'>
        <h3>🔍 Interactive Dashboard</h3>
        <p>An interactive dashboard for exploring clinical trial data with advanced visualizations.</p>
        <ul>
            <li>Interactive plots</li>
            <li>Custom visualizations</li>
            <li>Real-time data exploration</li>
            <li>Data export options</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Launch Interactive Dashboard"):
        with st.spinner("Starting Interactive Dashboard..."):
            success = launch_app(dashboard_app)
            if success:
                st.success("App launched! A new browser tab should open automatically.")

st.markdown("---")
st.markdown("""
### About the Suite
This suite of applications provides comprehensive tools for analyzing clinical trial data. The **Clinical Trial Analysis App** focuses on robust statistical modeling and analysis, while the **Interactive Dashboard** offers dynamic visualization and exploration tools.

Both applications can work with the same data formats and share underlying statistical engines.
""")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>© 2023 Clinical Trial Statistics App</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 