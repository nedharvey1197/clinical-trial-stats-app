import streamlit as st

# Set page config
st.set_page_config(
    page_title="HOME - Clinical Trial Statistics Suite",
    page_icon="🧪",
    layout="wide"
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
    copilot_app = "pages/03_Copilot_Trial_Confidence_Explorer.py"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='app-card'>
            <h4>📊 Clinical Trial Model Explorer</h4>
            <p>Explore and compare common statistical models for clinical trial data. Use built-in examples or upload your own data for quick, static analysis and visualization.</p>
            <ul>
                <li>Comprehensive suite of statistical models</li>
                <li>Example trial designs with mock data</li>
                <li>Data import and export</li>
                <li>Static visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Model Explorer", key="btn1", help="Click to launch the Clinical Trial Model Explorer"):
            with st.spinner("Starting Clinical Trial Model Explorer..."):
                st.switch_page(main_app)

    with col2:
        st.markdown("""
        <div class='app-card'>
            <h4>📈 Interactive Data Visualization Dashboard</h4>
            <p>Interactively explore, visualize, and filter clinical trial datasets. Create custom plots and gain insights with real-time, dynamic visualizations.</p>
            <ul>
                <li>Interactive plots</li>
                <li>Custom visualizations</li>
                <li>Real-time data exploration</li>
                <li>Data export options</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Visualization Dashboard", key="btn2", help="Click to launch the Interactive Data Visualization Dashboard"):
            with st.spinner("Starting Interactive Data Visualization Dashboard..."):
                st.switch_page(dashboard_app)

    with col3:
        st.markdown("""
        <div class='app-card'>
            <h4>🤖 Trial Design Copilot & Power Analysis</h4>
            <p>Step-by-step workflow for designing, validating, and analyzing clinical trials. Includes statistical guidance, power/sample size calculations, and robust data validation for confident, explainable results.</p>
            <ul>
                <li>Guided trial design workflow</li>
                <li>Power/sample size analysis</li>
                <li>Statistical guidance and validation</li>
                <li>Explainable, auditable results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Trial Design Copilot", key="btn3", help="Click to launch the Trial Design Copilot & Power Analysis"):
            with st.spinner("Starting Trial Design Copilot & Power Analysis..."):
                st.switch_page(copilot_app)

    st.markdown("---")
    st.markdown("""
    ### About the Suite
    This suite of applications provides comprehensive tools for analyzing clinical trial data. The **Clinical Trial Analysis App** focuses on robust statistical modeling and analysis, while the **Interactive Dashboard** offers dynamic visualization and exploration tools.

    Both applications can work with the same data formats and share underlying statistical engines.
    """)

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>© 2025 Athena Intelligence</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 