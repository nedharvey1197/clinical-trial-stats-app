import streamlit as st

# Set page config
st.set_page_config(
    page_title="Clinical Trial Statistics Suite",
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
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .app-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .header {
        text-align: center;
        margin-bottom: 40px;
        color: #1976d2;
    }
    .subtitle {
        font-size: 1.2em;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Custom sidebar styling */
    .css-1d391kg, .css-163ttbj {
        background-color: #f5f7f9;
    }
    .stRadio [role=radiogroup] {
        margin-top: 1em;
    }
    .stRadio [role=radiogroup] label {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 8px;
        display: block;
        border-left: 3px solid #1976d2;
        transition: all 0.2s ease;
    }
    .stRadio [role=radiogroup] label:hover {
        background: #e3f2fd;
        transform: translateX(3px);
    }
    
    /* Make the buttons more prominent */
    .stButton>button {
        background-color: #1976d2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.5em;
        font-size: 1.1em;
        font-weight: bold;
        margin: 0.5em 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<h1 class='header'>Clinical Trial Statistics Suite</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Comprehensive tools for analyzing clinical trial data</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='app-card'>
            <h3>📊 Statistical Modeling</h3>
            <p>Analyze clinical trial data with robust statistical methods.</p>
            <ul>
                <li>T-tests, ANOVA, Repeated Measures</li>
                <li>Automated assumption checking</li>
                <li>Effect size calculations</li>
                <li>Publication-ready visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Key Features")
        st.markdown("- **Data Import**: Upload your own CSV data or use built-in examples")
        st.markdown("- **Model Selection**: Choose from 10+ statistical models")
        st.markdown("- **Quality Assessment**: Automated data quality checks")
        st.markdown("- **Detailed Reporting**: Export results in multiple formats")

    with col2:
        st.markdown("""
        <div class='app-card'>
            <h3>🔍 Interactive Visualization</h3>
            <p>Explore clinical trial data through dynamic visualizations.</p>
            <ul>
                <li>Interactive plots</li>
                <li>Custom visualizations</li>
                <li>Real-time data exploration</li>
                <li>Advanced statistical insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("https://cdn.pixabay.com/photo/2018/05/18/15/30/webdesign-3411373_1280.jpg", 
                 caption="Interactive data visualization brings clinical trial data to life")

    st.markdown("---")
    st.markdown("""
    ### How to Get Started
    
    1. Select an application from the **sidebar navigation**
    2. Choose a statistical model or visualization type
    3. Upload your data or use provided examples
    4. Explore the results and download reports
    
    Both applications share the same underlying statistical engines and support similar data formats.
    """)

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>© 2023 Clinical Trial Statistics App | Created for clinical researchers</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 