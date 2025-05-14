import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ------------------ Mock Data Setup ------------------

# Simulated trial insights
design_patterns = pd.DataFrame({
    "Pattern": [
        "Dual-endpoint primary + exploratory",
        "Stratified population (age/BMI/metabolome)",
        "Microbiome integration"
    ],
    "% Trials": ["73%", "61%", "19%"],
    "Notes": [
        "Primary: % weight loss; Exploratory: metabolic flexibility",
        "Linked to higher effect detection",
        "Often exploratory, not yet pivotal"
    ]
})

regulatory_findings = pd.DataFrame({
    "Area": [
        "GLP analog precedent",
        "Behavior + Microbiome combo",
        "Rejection risk"
    ],
    "Finding": [
        "2 approvals in last 5 years used single co-intervention",
        "No Phase 3 yet approved with triple-axis approach",
        "High if no surrogate linkage shown"
    ]
})

expansion_targets = pd.DataFrame({
    "Target": ["PCOS", "NAFLD/NASH", "Post-bariatric regain", "Cognitive/Neurobehavioral"],
    "Precedent Trials": [
        "NCT04812345, NCT04298721",
        "NCT04089924",
        "Sparse trials",
        "NCT04999104 (Phase 1)"
    ],
    "Rationale": [
        "Overlaps with obesity and hormone regulation",
        "Receptive to dual filing",
        "High unmet need",
        "Pioneering zone"
    ]
})

recommendations = pd.DataFrame({
    "Design Feature": [
        "Population Stratification",
        "Endpoint Strategy",
        "Biomarker Plan",
        "Comparator Arm"
    ],
    "Recommendation": [
        "Age < 50; BMI class 2–3; behavior phenotype profiling",
        "% weight loss (primary); appetite modulation + insulin sensitivity (secondary)",
        "Fecal SCFA, GLP-1 levels, behavioral assessment scale",
        "Placebo + behavioral coaching"
    ]
})

# ------------------ Streamlit Layout ------------------
st.set_page_config(layout="wide")
st.title("🧠 Sentinel Knowledge Dashboard")
st.subheader("Context-Aware Planning Support for GLP-1 Obesity Therapy")

with st.expander("🔍 User Context Summary"):
    st.markdown("""
    - **Stage:** Completed Phase 1 → Planning Phase 2  
    - **Therapy:** Novel GLP-1 analog + microbiome + behavioral  
    - **Goal:** Maximize multi-endpoint validation for PH3  
    - **Strategy:** Expand future indication scope
    """)

# Design Patterns
st.markdown("### 🔗 Phase 2 Design Patterns (Recent Trials)")
st.dataframe(design_patterns, use_container_width=True)

# Regulatory Findings
st.markdown("### 🛂 Regulatory Fit Check")
st.dataframe(regulatory_findings, use_container_width=True)

# Expansion Opportunities
st.markdown("### 📈 Commercial Opportunity Map")
st.dataframe(expansion_targets, use_container_width=True)

# Copilot Recommendations
st.markdown("### 🧠 Copilot-Suggested Trial Design Elements")
st.dataframe(recommendations, use_container_width=True)

# Subgraph Visualization (placeholder)
st.markdown("### 🕸️ Knowledge Subgraph Snapshot (Mock)")
G = nx.Graph()
G.add_edges_from([
    ("GLP-1", "Obesity"),
    ("GLP-1", "Microbiome"),
    ("Microbiome", "Behavioral Therapy"),
    ("Obesity", "PCOS"),
    ("Obesity", "NAFLD/NASH"),
    ("Obesity", "Cognitive"),
    ("GLP-1", "Regulatory Warning")
])
fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', font_size=10)
st.pyplot(fig)

# Weakness Review Placeholder
st.markdown("### 🧩 Observed Schema/Graph Weaknesses (Preliminary)")
st.warning("\n- No current ontology support for behavioral-cognitive synergy modeling\n- 'Market Fit' nodes not structurally aligned to regulatory/clinical data\n- Outcome linkage to biomarker cascade missing in schema (need OWL + SHACL support)\n")
