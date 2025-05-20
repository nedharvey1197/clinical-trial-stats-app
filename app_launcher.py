import streamlit as st

# Set page config
st.set_page_config(
    page_title="HOME - Clinical Trial Statistics Suite",
    page_icon="🧪",
    layout="centered"
)


st.markdown("""
<style>
    .app-card {
        padding: 10px;
        padding-top:
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
        border-left: 5px solid #1976d2;
        transition: transform 0.2s ease-in-out;
        min-height: 360px;  /* Set minimum height */
        height: auto;       /* Allow content to expand if needed */
        min-width: 300px;
        width: auto;
        display: flex;
        flex-direction: column;
    }
    .app-card:hover {
        transform: scale(1.1);
    }
    .header {
        text-align: center;
        margin-bottom: 40px;
    }
    .subtitle {
        font-size: 1.5em;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton > button, .stDownloadButton > button {
            background-color: #1976d2;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5em 1.5em;
            font-size: 1.1em;
            font-weight: bold;
            margin: 0.5em 0;
            transition: background 0.2s;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background-color: #1565c0;
            color: #fff;
        }
    
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<h1 class='header'>HOME</h1>", unsafe_allow_html=True)
    st.markdown("<h4 p class='subtitle'>Clinical Trial Statistics Suite</h4>", unsafe_allow_html=True)

    # Define app paths
    main_app = "pages/01_Clinical_Trial_Analysis.py"
    dashboard_app = "pages/02_Interactive_Dashboard.py"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='app-card'>
            <h4>📊 Statistical Models for Clinical Trials</h4>
            <p>Explore the most common models for analyzing clinical trial data.  Use the examples or upload your own data.</p>
            <ul>
                <li>Comprehensive suite of statistical models</li>
                <li>Real world examples of trial designs, with mock data</li>
                <li>Data import and export</li>
                <li>Static visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Clinical Trial Analysis App", key="btn1", help="Click to launch the Clinical Trial Analysis App"):
            with st.spinner("Starting Clinical Trial Analysis App..."):
                st.switch_page(main_app)

    with col2:
        st.markdown("""
        <div class='app-card'>
            <h4>🔍 Interactive Dashboard</h4>
            <p>An interactive dashboard for exploring clinical trial data with advanced visualizations.</p>
            <ul>
                <li>Interactive plots</li>
                <li>Custom visualizations</li>
                <li>Real-time data exploration</li>
                <li>Data export options</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Interactive Dashboard", key="btn2", help="Click to launch the Interactive Dashboard"):
            with st.spinner("Starting Interactive Dashboard..."):
                st.switch_page(dashboard_app)

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