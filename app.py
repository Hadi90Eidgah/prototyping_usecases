# Research Impact Dashboard (Final working version for new CSV data)

import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import numpy as np
import os
from config import *
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Research Impact Dashboard",
    page_icon="🔬",
    layout="wide"
)

# --- Load External CSS ---
def load_css(file_name):
    """Load CSS and re-inject after render to override Streamlit theming."""
    with open(file_name) as f:
        css = f.read()

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <script>
            setTimeout(function() {{
                var css = `{css}`;
                var style = document.createElement('style');
                style.innerHTML = css;
                document.head.appendChild(style);
            }}, 1500);
        </script>
        """,
        unsafe_allow_html=True
    )


load_css('style.css')


# --- Helper: build summary dynamically ---
def build_summary_from_nodes(nodes_df):
    """Generate a summary DataFrame from nodes when no summary CSV exists"""
    try:
        # Ensure required columns exist
        if 'node_type' not in nodes_df.columns:
            st.warning("⚠️ 'node_type' missing in nodes — assigning defaults.")
            nodes_df['node_type'] = 'publication'
            if len(nodes_df) > 0:
                nodes_df.loc[nodes_df.index[0], 'node_type'] = 'grant'
                nodes_df.loc[nodes_df.index[-1], 'node_type'] = 'treatment'

        for col in ['network_id', 'disease', 'treatment_name', 'grant_id',
                    'funding_amount', 'year', 'approval_year']:
            if col not in nodes_df.columns:
                nodes_df[col] = np.nan

        grants = nodes_df[nodes_df['node_type'] == 'grant'].copy()
        summary_df = grants[['network_id', 'disease', 'treatment_name', 'grant_id',
                             'funding_amount', 'year', 'approval_year']].rename(
            columns={'year': 'grant_year'}
        )

        if 'node_id' in nodes_df.columns:
            pub_counts = (
                nodes_df[nodes_df['node_type'] == 'publication']
                .groupby('network_id')['node_id']
                .count()
                .rename('total_publications')
            )
            summary_df = summary_df.merge(pub_counts, on='network_id', how='left')
        else:
            summary_df['total_publications'] = 0

        summary_df['research_duration'] = (
            summary_df['approval_year'] - summary_df['grant_year']
        )

        summary_df = summary_df.fillna(0)

        # --- Add readable names for single-network datasets ---
        if 'disease' in summary_df.columns:
            summary_df.loc[:, 'disease'] = 'General Research Network'
        if 'treatment_name' in summary_df.columns:
            summary_df.loc[:, 'treatment_name'] = 'Primary Treatment'
        if 'grant_id' in summary_df.columns:
            summary_df.loc[:, 'grant_id'] = 'AUTO-GEN-001'

        return summary_df

    except Exception as e:
        st.error(f"Error building summary: {e}")
        return pd.DataFrame()


# --- Visualization Utilities (Structured Layout) ---
def get_node_positions(network_nodes, network_id):
    """Assign structured positions based on node type and publication year."""
    node_positions = {}
    np.random.seed(42 + int(str(network_id).encode().hex(), 16) % 1000)

    # --- 1️⃣ Grant node (left side) ---
    grants = network_nodes[network_nodes["node_type"] == NODE_TYPE_GRANT]
    if not grants.empty:
        grant = grants.iloc[0]
        node_positions[grant["node_id"]] = (
            NODE_POSITIONS_X["grant"],
            NODE_POSITIONS_Y["grant"]
        )

    # --- 2️⃣ Grant-funded publications ---
    funded_pubs = network_nodes[
        network_nodes["node_type"].isin(["publication", "grant_funded_pub"])
    ]
    for i, (_, node) in enumerate(funded_pubs.iterrows()):
        x = NODE_POSITIONS_X["grant_funded_pub"]
        y = NODE_POSITIONS_Y["grant_funded_pub"][i % len(NODE_POSITIONS_Y["grant_funded_pub"])]
        node_positions[node["node_id"]] = (x, y)

    # --- 3️⃣ Ecosystem publications ---
    ecosystem_pubs = network_nodes[
        network_nodes["node_type"].isin(["ecosystem_pub"])
    ]
    for i, (_, node) in enumerate(ecosystem_pubs.iterrows()):
        x = NODE_POSITIONS_X["ecosystem_pub_cluster"][i % len(NODE_POSITIONS_X["ecosystem_pub_cluster"])]
        y = NODE_POSITIONS_Y["ecosystem_pub_cluster"][i % len(NODE_POSITIONS_Y["ecosystem_pub_cluster"])]
        node_positions[node["node_id"]] = (x, y)

    # --- 4️⃣ Treatment pathway publications ---
    treat_pubs = network_nodes[
        network_nodes["node_type"].isin(["treatment_pathway_pub"])
    ]
    for i, (_, node) in enumerate(treat_pubs.iterrows()):
        x = NODE_POSITIONS_X["treatment_pathway_pub"]
        y = NODE_POSITIONS_Y["treatment_pathway_pub"][i % len(NODE_POSITIONS_Y["treatment_pathway_pub"])]
        node_positions[node["node_id"]] = (x, y)

    # --- 5️⃣ Treatment node (right side) ---
    treatments = network_nodes[network_nodes["node_type"] == NODE_TYPE_TREATMENT]
    if not treatments.empty:
        treat = treatments.iloc[0]
        node_positions[treat["node_id"]] = (
            NODE_POSITIONS_X["treatment"],
            NODE_POSITIONS_Y["treatment"]
        )

    return node_positions


def create_edge_trace(edges, node_positions, edge_type, name, showlegend):
    """Create a Plotly scatter trace for edges."""
    edge_x, edge_y = [], []
    for _, edge in edges.iterrows():
        if edge["source_id"] in node_positions and edge["target_id"] in node_positions:
            x0, y0 = node_positions[edge["source_id"]]
            x1, y1 = node_positions[edge["target_id"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    if not edge_x:
        return None

    return go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=EDGE_WIDTHS.get(edge_type, 1.5), color=EDGE_COLORS.get(edge_type, "gray")),
        hoverinfo="none",
        mode="lines",
        name=name,
        showlegend=showlegend
    )


def create_node_trace(nodes, node_positions, node_type, name, showlegend):
    """Create a Plotly scatter trace for nodes."""
    node_x, node_y, hover_texts = [], [], []

    for _, node in nodes.iterrows():
        node_id = node["node_id"]
        if node_id not in node_positions:
            continue
        x, y = node_positions[node_id]
        node_x.append(x)
        node_y.append(y)
        hover_texts.append(node.get("title", node_id))

    return go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=hover_texts,
        marker=dict(
            size=NODE_SIZES.get(node_type, 10),
            color=NODE_COLORS.get(node_type, "#718096"),
            line=dict(width=2, color="#e2e8f0")
        ),
        name=name,
        showlegend=showlegend
    )


def create_network_visualization(nodes_df, edges_df, network_id, grant_id=None, treatment_name=None):
    """Generate the structured left→right network visualization."""
    network_nodes = nodes_df[nodes_df["network_id"] == network_id]
    network_edges = edges_df

    if network_nodes.empty or network_edges.empty:
        raise ValueError("No nodes or edges found for this network.")

    node_positions = get_node_positions(network_nodes, network_id)

    # Edge layers
    edge_traces = [
        create_edge_trace(network_edges, node_positions, EDGE_TYPE_FUNDED_BY, "Grant Funding", True),
        create_edge_trace(network_edges, node_positions, EDGE_TYPE_LEADS_TO_TREATMENT, "Research Impact Pathway", True),
        create_edge_trace(network_edges, node_positions, EDGE_TYPE_ENABLES_TREATMENT, "Treatment Enablement", True),
    ]

    # Node layers
    node_traces = [
        create_node_trace(network_nodes[network_nodes["node_type"] == NODE_TYPE_GRANT], node_positions, NODE_TYPE_GRANT, "Grant", True),
        create_node_trace(network_nodes[network_nodes["node_type"] == NODE_TYPE_PUBLICATION], node_positions, "grant_funded_pub", "Grant-Funded Papers", True),
        create_node_trace(network_nodes[network_nodes["node_type"] == NODE_TYPE_TREATMENT], node_positions, NODE_TYPE_TREATMENT, "Approved Treatment", True),
    ]

    fig = go.Figure(data=[t for t in edge_traces + node_traces if t is not None])
    fig.update_layout(
        title=f"Research Impact Network — {grant_id or ''} → {treatment_name or ''}",
        showlegend=True,
        hovermode="closest",
        margin=dict(b=40, l=40, r=40, t=70),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-6, 7]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]),
        height=600,
        plot_bgcolor="rgba(14,17,23,1)",
        paper_bgcolor="rgba(14,17,23,1)",
        font=dict(color="#e2e8f0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(45,55,72,0.9)",
            bordercolor="rgba(74,85,104,0.5)",
            borderwidth=1,
            font=dict(size=11, color="#e2e8f0")
        )
    )
    return fig



# --- Data Loading ---
@st.cache_data(ttl=CACHE_TTL)
def load_database():
    """Load data from database or CSV files"""
    try:
        # --- New logic: load directly from CSVs if they exist ---
        if os.path.exists(NODES_CSV_PATH) and os.path.exists(EDGES_CSV_PATH):
            nodes_df = pd.read_csv(NODES_CSV_PATH)
            st.write("🧩 Node columns detected in CSV:", list(nodes_df.columns))

            # Ensure essential columns exist
            if 'node_type' not in nodes_df.columns:
                nodes_df['node_type'] = 'publication'
                if len(nodes_df) > 0:
                    nodes_df.loc[nodes_df.index[0], 'node_type'] = 'grant'
                    nodes_df.loc[nodes_df.index[-1], 'node_type'] = 'treatment'
            for col in ['disease', 'treatment_name', 'grant_id', 'approval_year', 'funding_amount']:
                if col not in nodes_df.columns:
                    nodes_df[col] = np.nan

            edges_df = pd.read_csv(EDGES_CSV_PATH)
            st.write("🧩 Edge columns before rename:", list(edges_df.columns))

            # --- Normalize edge columns to match the app's expected format ---
            edges_df = edges_df.rename(columns={
                'source': 'source_id',
                'target': 'target_id',
                'relation': 'edge_type'
            })
            st.write("✅ Edge columns after rename:", list(edges_df.columns))

            summary_df = build_summary_from_nodes(nodes_df)
            st.success("✅ Loaded nodes and edges from CSV files.")
            st.write("📊 Summary preview:", summary_df.head())

            return nodes_df, edges_df, summary_df

        # --- Legacy database loading path ---
        elif os.path.exists(DATABASE_PATH):
            conn = sqlite3.connect(DATABASE_PATH)
            nodes_df = pd.read_sql('SELECT * FROM nodes', conn)
            edges_df = pd.read_sql('SELECT * FROM edges', conn)
            summary_df = pd.read_sql('SELECT * FROM network_summary', conn)
            conn.close()
            return nodes_df, edges_df, summary_df

        # --- Fallback if other paths missing ---
        else:
            nodes_df = pd.read_csv(NODES_CSV_PATH)
            edges_df = pd.read_csv(EDGES_CSV_PATH)
            summary_df = pd.read_csv(SUMMARY_CSV_PATH)
            return nodes_df, edges_df, summary_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# --- Main App ---
def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">Research Impact Network Analysis</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.1rem; color: #a0aec0; margin-bottom: 3rem; font-weight: 300;">'
        'Mapping Research Pathways from Grant Funding to Breakthrough Treatments</p>',
        unsafe_allow_html=True
    )

    # Load data
    nodes_df, edges_df, summary_df = load_database()

    # Sidebar metrics
    st.sidebar.markdown("### Database Statistics")
    st.sidebar.write(f"Total connections: {len(edges_df)}")
    if 'edge_type' in edges_df.columns:
        st.sidebar.write(f"Treatment pathways: {len(edges_df[edges_df['edge_type'] == EDGE_TYPE_LEADS_TO_TREATMENT])}")
    else:
        st.sidebar.write("Treatment pathways: (edge_type column missing)")

    if summary_df.empty:
        st.error("No data available. Please check your database files.")
        return

    # Normalize strings for consistent filtering
    for col in ['disease', 'treatment_name', 'grant_id']:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].astype(str).str.strip()

    st.markdown("## Select the Citation Network")
    st.markdown("### Available Research Networks")

    selected_network = summary_df.iloc[0]["network_id"]
    selected_summary = summary_df.iloc[0]

    st.markdown(f"""
    <div class="selection-card grant-card">
        <div class="network-title">{selected_summary['disease']}</div>
        <div class="treatment-name">{selected_summary['treatment_name']}</div>
        <div class="network-details">
            Grant ID: {selected_summary['grant_id']}<br>
            Publications: {selected_summary['total_publications']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Analyze Citation Network", use_container_width=True):
        st.session_state.selected_network = selected_network
        st.success(f"Selected network: {selected_summary['disease']} → {selected_summary['treatment_name']}")

        # --- Visualization Section ---
        st.markdown("### 🕸️ Research Network Visualization")
        with st.spinner("Creating network visualization..."):
            try:
                fig = create_network_visualization(
                    nodes_df,
                    edges_df,
                    selected_network,
                    grant_id=selected_summary["grant_id"],
                    treatment_name=selected_summary["treatment_name"]
                )
                if fig and fig.data:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No nodes or edges to visualize.")
            except Exception as e:
                st.error(f"Error generating visualization: {e}")


if __name__ == "__main__":
    main()
