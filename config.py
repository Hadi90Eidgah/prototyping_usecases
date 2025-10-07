# Configuration for the Research Impact Dashboard (Fine-tuned Layout)

# --- Data Loading ---
DATABASE_PATH = 'streamlit_research_database.db'
NODES_CSV_PATH = 'streamlit_nodes.csv'
EDGES_CSV_PATH = 'streamlit_edges.csv'
SUMMARY_CSV_PATH = 'streamlit_summary.csv'
CACHE_TTL = 3600  # Cache data for 1 hour

# --- Node and Edge Types ---
NODE_TYPE_GRANT = 'grant'
NODE_TYPE_TREATMENT = 'treatment'
NODE_TYPE_PUBLICATION = 'publication'
EDGE_TYPE_FUNDED_BY = 'funded_by'
EDGE_TYPE_LEADS_TO_TREATMENT = 'leads_to_treatment'
EDGE_TYPE_CITES = 'cites'
EDGE_TYPE_ENABLES_TREATMENT = 'enables_treatment'

# --- Visualization Colors ---
NODE_COLORS = {
    'grant': '#4299e1',                # Blue
    'publication': '#a0aec0',          # Gray
    'treatment': '#38b2ac',            # Teal
    'grant_funded_pub': '#718096',     # Muted gray-blue
    'treatment_pathway_pub': '#ed8936',# Orange
    'ecosystem_pub': '#718096'         # Gray for ecosystem nodes
}

# --- Node Sizes ---
NODE_SIZES = {
    'grant': 30,                # Main funding node
    'publication': 10,
    'treatment': 35,            # Final approved therapy
    'grant_funded_pub': 15,     # Slightly larger for visibility
    'treatment_pathway_pub': 15,
    'ecosystem_pub': 8
}

# --- Edge Colors ---
EDGE_COLORS = {
    'funded_by': 'rgba(66, 153, 225, 0.7)',           # Subtle blue-gray
    'leads_to_treatment': 'rgba(160, 174, 192, 0.25)', # Bright blue (impact path)
    'cites': 'rgba(160, 174, 192, 0.25)',            # Light gray for citations
    'enables_treatment': 'rgba(56, 178, 172, 0.8)'   # Teal-green to treatment
}

# --- Edge Widths ---
EDGE_WIDTHS = {
    'funded_by': 1.5,           # Grant → Funded Papers
    'leads_to_treatment': 1.8,    # Pathway links remain strong
    'cites': 0.8,               # Thin ecosystem edges
    'enables_treatment': 1.5    # ⬅️ Same thickness as funded_by (was 2)
}

# --- Layout Positions (Distances) ---
# Refined distances: bring treatment closer to pathway papers
NODE_POSITIONS_X = {
    'grant': -4,                # Was -5 → moved slightly right
    'grant_funded_pub': -2.5,
    'ecosystem_pub_cluster': [-0.5, 0.5, 1.5, 0],
    'treatment_pathway_pub': 3.5,
    'treatment': 4.8            # ⬅️ Was 6 → now closer to orange pathway nodes
}

NODE_POSITIONS_Y = {
    'grant': 0,
    'grant_funded_pub': [2, 0.7, -0.6, -1.9],
    'ecosystem_pub_cluster': [1, 0.5, -0.5, -1.5],
    'treatment_pathway_pub': [1, 0, -1],
    'treatment': 0
}
