import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ------------------ Mock Data Setup ------------------

# Simulated trial insights
design_patterns = pd.DataFrame({
    "Design Pattern": [
        "Dual-endpoint primary + exploratory",
        "Stratified population (age/BMI/metabolome)",
        "Microbiome integration"
    ],
    "Prevalence in Similar Trials": ["73%", "61%", "19%"],
    "Interpretation": [
        "Primary endpoint focused on % weight loss; exploratory endpoints assess metabolic flexibility or related biomarkers",
        "Use of baseline stratification by age, BMI, or metabolic phenotype to improve signal detection",
        "Microbiome data collected as exploratory or mechanistic input; rarely used as pivotal endpoint"
    ]
})

regulatory_findings = pd.DataFrame({
    "Regulatory Area": [
        "GLP analog precedent",
        "Behavior + Microbiome combo",
        "Rejection risk"
    ],
    "Regulatory Insight": [
        "Recent approvals (past 5 years) used a single co-intervention for clarity in effect attribution",
        "No approved Phase 3 trials yet combine behavioral + microbiome + pharmacologic interventions",
        "Submissions are at risk if surrogate linkage (e.g. biomarker-pathway-outcome) is unclear"
    ]
})

expansion_targets = pd.DataFrame({
    "Expansion Opportunity": ["PCOS", "NAFLD/NASH", "Post-bariatric regain", "Cognitive/Neurobehavioral"],
    "Precedent Trials": [
        "NCT04812345, NCT04298721",
        "NCT04089924",
        "Sparse trials",
        "NCT04999104 (Phase 1 only)"
    ],
    "Market Justification": [
        "Strong mechanistic overlap with GLP axis; hormonal regulation",
        "FDA receptive to GLP+ candidates with metabolic comorbidity scope",
        "High unmet need with emerging payer attention",
        "Novel territory, potential for behavioral-cognitive synergy"
    ]
})

recommendations = pd.DataFrame({
    "Design Element": [
        "Population Stratification",
        "Endpoint Strategy",
        "Biomarker Plan",
        "Comparator Arm"
    ],
    "Copilot Recommendation": [
        "Focus on individuals < 50 years, BMI class 2–3, stratified by behavioral phenotype",
        "Primary: % total body weight loss; Secondary: appetite modulation and insulin sensitivity",
        "Include gut-derived SCFA levels, plasma GLP-1, behavioral adherence scale",
        "Include active placebo group with behavioral coaching to isolate co-intervention effects"
    ]
})

# ------------------ Streamlit Layout ------------------
st.set_page_config(layout="wide")
st.title("🧠 Sentinel Knowledge Dashboard")
st.subheader("Context-Aware Planning Support for GLP-q Obesity Therapy")

with st.expander("🔍 User Context Summary"):
    st.markdown("""
    This view represents a Phase 2 trial planning scenario for a novel GLP-q intervention designed to promote lasting obesity reversal in adults under 50. The trial integrates microbiome-related and behavioral components, with an intent to extract insights relevant for Phase 3 readiness and market expansion.
    
    **Inferred Profile:**  
    - **Stage:** Completed Phase 1 → Planning Phase 2  
    - **Therapy Type:** GLP-q analog with microbiome + behavioral integration  
    - **Trial Intent:** Identify dual-path efficacy and design signals to inform PH3  
    - **Strategic Focus:** Maximize future indication breadth via exploratory endpoints
    """)

# Design Patterns Section
st.markdown("### 🔗 Phase 2 Design Patterns")
st.markdown("""
**Definition**: This table reflects common design choices across recent Phase 2 obesity/metabolic trials with similar MoAs. 
**Prevalence**: Refers to the percentage of such trials using each pattern.
**Interpretation**: Provides rationale and insight behind the prevalence of each pattern.
""")
st.dataframe(design_patterns, use_container_width=True)

# Regulatory Findings Section
st.markdown("### 🛂 Regulatory Fit Check")
st.markdown("""
**Purpose**: Summarizes relevant regulatory precedents and risks based on the design's novelty. Draws from FDA/EMA communications and trial rejections.
""")
st.dataframe(regulatory_findings, use_container_width=True)

# Expansion Opportunities Section
st.markdown("### 📈 Adjacent Indication Expansion Targets")
st.markdown("""
**Definition**: Opportunities to use Phase 2 data to support future label expansion. These areas show mechanistic or commercial overlap with the core obesity target.
""")
st.dataframe(expansion_targets, use_container_width=True)

# Copilot Design Recommendations Section
st.markdown("### 🧠 Copilot-Suggested Trial Design Elements")
st.markdown("""
**How Generated**: Based on subgraph analysis of past trials, endpoint trends, and biomarker linkages in GLP-related therapies. Designed to balance evidence strength with regulatory expectations.
""")
st.dataframe(recommendations, use_container_width=True)

# Subgraph Visualization (placeholder)
st.markdown("### 🕸️ Knowledge Subgraph Snapshot")
st.markdown("""
This mock network shows entities and concepts inferred as relevant based on the current trial's intent, MoA, and strategic posture.
""")
G = nx.Graph()
G.add_edges_from([
    ("GLP-q", "Obesity"),
    ("GLP-q", "Microbiome"),
    ("Microbiome", "Behavioral Therapy"),
    ("Obesity", "PCOS"),
    ("Obesity", "NAFLD/NASH"),
    ("Obesity", "Cognitive"),
    ("GLP-q", "Regulatory Warning")
])
fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', font_size=10)
st.pyplot(fig)

# Weakness Review Placeholder
st.markdown("### 🧩 Observed Schema/Graph Weaknesses (Preliminary)")
st.markdown("""
- **Ontology gap:** No clear support for behavioral-cognitive synergy modeling
- **Graph gap:** Missing linkage between 'market fit' signals and trial design nodes
- **Modeling limitation:** Lack of outcome → biomarker → MoA chaining restricts deeper reasoning
- **Action:** Extend ontology with class chains for `BehavioralMechanism`, `MarketAccessSignal`, and path-based SHACL constraints
""")
