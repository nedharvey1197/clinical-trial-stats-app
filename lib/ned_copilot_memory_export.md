# 🧠 **Condensed Full Memory Export: Ned’s Clinical Copilot Context**  
_Last updated: May 2025_

## 🔹 **User Identity & Operating Style**
- **Name**: Ned  
- **Role**: Vision-driven systems thinker; strategic, not detail-focused  
- **Working Style**: Moves fast, intuition-led; relies on AI to tie loose ends  
- **Preferred AI Behavior**:
  - Be a thought partner, not a cheerleader  
  - Challenge weak logic and incomplete ideas  
  - Avoid flattery; provide forward-thinking, honest feedback  
  - Work in clear modules with context-aware references  
- **AI Operating Modes**:
  - “Support mode” (default): Don’t restructure unless told  
  - “Development sprint mode”: Full-speed code or architecture work when explicitly enabled  

## 🔹 **Core Products**
### 1. **Clinical Trial Optimization Copilot** (Primary)
- Audience: Clinical leaders at small-to-mid biotech
- Goal: Acts as a **Virtual Development Officer**  
- Function: Transforms sparse input (NCT IDs, trial docs, scraped websites) into:
  - Trial context
  - Insightful synthesis (5Ws, 5 Rights, PICO)
  - Parametric design evaluation (simulation, power calc, feasibility)

### 2. **Development Copilot** (Secondary)
- Goal: AI-assisted coding, memory tracking, and system reasoning  
- Tools: GitHub Codespaces, Docker, modular BE/FE separation  

## 🔹 **Front-End Architecture**
- **Stack**: React (Vite), TailwindCSS, Shadcn UI  
- **Key Components**: `StageOneIntake`, `StageTwo5Ws`, `StageThreeSynopsis`, `Dashboard`  
- **State Tracking**: Session, completeness, scenario design  
- **Constraints**:
  - No logic in `App.jsx`
  - No overwrites without warning
  - Drop-in refactors only unless otherwise cleared
- **Session Behavior**:
  - Uploads trigger background document parsing
  - Copilot provides real-time summarization & insight notifications
  - Tabs reflect staged intake → synthesis → design evaluation

## 🔹 **Back-End Architecture**
- **Stack**: FastAPI, PostgreSQL, Neo4j, Databricks, Celery, Airflow  
- **Modules**:
  - `trial_design.py`: Intent + structure modeling  
  - `evaluation_bundle.py`: Outcome analysis logic  
  - `logger.py`: Lightweight versioning & audit trail  
  - `moa_tagging_pipeline.py`: Canonical mapping, class/tag assignment  
  - `abstract_etl.py`: ASCO, PubMed, MeSH-based document ingestion  
- **Data Stores**:
  - **AACT (Postgres)**: ClinicalTrials.gov structured data
  - **Neo4j**: Knowledge graph (Trial, MoA, Endpoint, Population nodes)
  - **Databricks Delta Lake**: Deep document and enrichment storage
- **Architectural Priorities**:
  - Parametric trial modeling
  - Bayesian optimization of design inputs
  - Version-controlled evaluation
  - Reasoning from structured and unstructured knowledge

## 🔹 **Data Pipelines & ETL**
- **Sources**:
  - AACT Postgres: trial summary, arms, outcomes, eligibility
  - ClinicalTrials.gov API: fallback JSON/NCT scraping
  - PubMed / ASCO: abstract ingestion and MeSH tagging
  - FDA: CRL and guidance documents for regulatory context

- **Tools**:
  - NLP: Stanza, SciSpacy, LangChain, LlamaIndex
  - MeSH Normalization: XML fallback, synonym expansion
  - Graph Sync: PostgreSQL ↔ Neo4j dual store with `DataManager` abstraction
  - Taxonomies: MoA hierarchy (Target → Action → Pathway → Effect), disease area enums

## 🔹 **Schema & Modeling Priorities**
- **Frameworks**:
  - 5Ws + 5 Rights alignment  
  - PICO-informed design synthesis  
  - MoA classification → parametric identifiers  
  - Outcome-based evaluation bundling
- **Materialized Views**:
  - `trial_summary_mat_full`
  - `trial_arms_mat_full_v1`
  - `trial_safety_mat_full_v1`
  - `NLP_intervention_name_tags`
- **Design Goals**:
  - Support for trial archetype discovery
  - Statistical evaluation of design intent
  - Structured insight summarization from sparse documents

## 🔹 **Knowledge Graph Schema (Neo4j)**
- **Core Nodes**:
  - `Trial`, `NewTrial`, `Session`, `DesignState`, `Scenario`
  - `Condition`, `Intervention`, `Endpoint`, `Population`
  - `MoATarget`, `MoAAction`, `MoAPathway`, `MoAEffect`

- **Relationships**:
  - `HAS_SCENARIO`, `HAS_EVALUATION`, `MEASURES`, `TARGETS`, `SIMILAR_TO`, etc.
- **Reasoning Priorities**:
  - MoA-tagged similarity
  - Embedding-enhanced recall
  - Governance & validation tracking via semantic relationships

## 🔹 **NLP & Document Enrichment**
- **Goals**:
  - Extract 5Ws + design schema from PDFs, abstracts, or copied text  
  - Generate structured JSON from raw input (promptable by Copilot)  
  - Identify MoA, endpoints, controls, eligibility, dropouts
- **Fallbacks**:
  - Use XML MeSH parsing if SPARQL or APIs fail  
  - Generate one row per MeSH term with synonym aggregation  

## 🔹 **Current Top Priorities**
1. **MoA tagging sync**  
   - Neo4j node updates from `moa_chain.yaml`  
   - PostgreSQL → Neo4j mapping, canonical ID resolution  

2. **Abstract ingestion & enrichment**  
   - NLP term normalization, MeSH matching  
   - Plugin-based ingestion for cardiology, oncology, respiratory  

3. **Stateful UI flow & 5Ws completion logic**  
   - Stage-wise completion tracking  
   - Background parsing integration with live summarization  

4. **Evaluation bundle generalization**  
   - Extend t-tests and ANOVA modules  
   - Prepare for future Bayesian and simulation-based evaluators  

## 🔹 **Process & Behavior Guidelines**
- **Development Practices**:
  - Maintain schema discipline (shared, versioned, validated)
  - Avoid module divergence; map endpoints to canonical logic
  - All feature proposals must be forward-compatible
- **AI Behavior**:
  - Always reflect session state and project priorities
  - Push back on weak logic or unnecessary complexity
  - Summarize each thread with:
    1. Top 1–3 priorities (w/ status)
    2. Immediate next recommended step
