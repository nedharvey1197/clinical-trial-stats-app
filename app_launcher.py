import streamlit as st
import subprocess
import os
import sys

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
    .launch-btn {
        background-color: #1976d2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
        font-weight: bold;
        margin-top: 10px;
        width: 100%;
    }
    .launch-btn:hover {
        background-color: #1565c0;
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

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='app-card'>
        <h3>📊 Clinical Trial Analysis App</h3>
        <p>The main application for analyzing clinical trial data with various statistical models.</p>
        <ul>
            <li>Comprehensive statistical analysis</li>
            <li>Multiple statistical models</li>
            <li>Data import and export</li>
            <li>Static visualizations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Launch Analysis App", key="analysis"):
        st.success("Starting Clinical Trial Analysis App...")
        # Construct the path to the main application
        script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clinical_trial_analysis_app.py")
        # Stop the current app and start the main analysis app
        st.rerun()  # This will clear the current app state
        # Use sys.executable to ensure we use the same Python interpreter
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", script_path])
        st.stop()

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
    
    if st.button("Launch Dashboard", key="dashboard"):
        st.success("Starting Interactive Dashboard...")
        # Construct the path to the dashboard
        script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "interactive_dashboard.py")
        # Stop the current app and start the dashboard
        st.rerun()  # This will clear the current app state
        # Use sys.executable to ensure we use the same Python interpreter
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", script_path])
        st.stop()

st.markdown("---")
st.markdown("""
### About the Suite
This suite of applications provides comprehensive tools for analyzing clinical trial data. The **Clinical Trial Analysis App** focuses on robust statistical modeling and analysis, while the **Interactive Dashboard** offers dynamic visualization and exploration tools.

Both applications can work with the same data formats and share underlying statistical engines.
""")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>© 2023 Clinical Trial Statistics App</p>", unsafe_allow_html=True) 