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

def main():
    st.markdown("<h1 class='header'>Clinical Trial Statistics Suite</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Choose an application from the sidebar or explore the features below</p>", unsafe_allow_html=True)

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

    st.markdown("---")
    st.markdown("""
    ### About the Suite
    This suite of applications provides comprehensive tools for analyzing clinical trial data. The **Clinical Trial Analysis App** focuses on robust statistical modeling and analysis, while the **Interactive Dashboard** offers dynamic visualization and exploration tools.

    Both applications can work with the same data formats and share underlying statistical engines. Use the sidebar navigation to access these tools.
    """)

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>© 2023 Clinical Trial Statistics App</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 