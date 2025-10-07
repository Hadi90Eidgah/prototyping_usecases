# Research Impact Dashboard - Optimized

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
    import time

    # Read the CSS file
    with open(file_name) as f:
        css = f.read()

    # Inject CSS initially
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # Re-inject CSS after Streamlit finishes rendering (for Streamlit Cloud overrides)
    st.markdown(
        f"""
        <script>
            setTimeout(function() {{
                var css = `{css}`;
                var style = document.createElement('style');
                style.innerHTML = css;
                document.head.appendChild(style);
                console.log('✅ Forced CSS reinjection {int(time.time())}');
            }}, 1500);
        </script>
        """,
        unsafe_allow_html=True
    )


load_css('style.css')

# --- Data Loading ---
@st.cache_data(ttl=CACHE_TTL)
def load_database():
    """Load data from database or CSV files"""
    try:
        if os.path.exists(DATABASE_PATH):
            conn = sqlite3.connect(DATABASE_PATH)
            nodes_df = pd.read_sql('SELECT * FROM nodes', conn)
            edges_df = pd.read_sql('SELECT * FROM edges', conn)
            summary_df = pd.read_sql('SELECT * FROM network_summary', conn)
            conn.close()
            return nodes_df, edges_df, summary_df
        else:
            nodes_df = pd.read_csv(NODES_CSV_PATH)
            edges_df = pd.read_csv(EDGES_CSV_PATH)
            summary_df = pd.read_csv(SUMMARY_CSV_PATH)
            return nodes_df, edges_df, summary_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

def create_sample_data():
    """Create fallback sample data"""
    summary_df = pd.DataFrame({
        'network_id': [1, 2, 3],
        'disease': ['Cancer', 'Alzheimer Disease', 'Diabetes'],
        'treatment_name': ['CAR-T Cell Therapy', 'Aducanumab Plus', 'Smart Insulin Patch'],
        'grant_id': ['INST-R01-877572', 'INST-U01-352422', 'INST-U01-448787'],
        'grant_year': [2015, 2019, 2015],
        'approval_year': [2024, 2023, 2025],
        'funding_amount': [2807113, 2304762, 1983309],
        'total_publications': [37, 37, 37],
        'research_duration': [9, 4, 10]
    })
    
    nodes_df = pd.DataFrame({
        'node_id': ['GRANT_1', 'GRANT_2', 'GRANT_3', 'TREAT_1', 'TREAT_2', 'TREAT_3'],
        'node_type': ['grant', 'grant', 'grant', 'treatment', 'treatment', 'treatment'],
        'network_id': [1, 2, 3, 1, 2, 3]
    })
    
    edges_df = pd.DataFrame({
        'source_id': ['GRANT_1', 'GRANT_2', 'GRANT_3'],
        'target_id': ['TREAT_1', 'TREAT_2', 'TREAT_3'],
        'edge_type': ['leads_to', 'leads_to', 'leads_to'],
        'network_id': [1, 2, 3]
    })
    
    return nodes_df, edges_df, summary_df

# --- Visualization Functions ---

def get_node_positions(network_nodes, network_id):
    """Calculate node positions based on node type."""
    node_positions = {}
    np.random.seed(42 + network_id)

    # Position grant
    grants = network_nodes[network_nodes['node_type'] == NODE_TYPE_GRANT]
    if not grants.empty:
        node_positions[grants.iloc[0]['node_id']] = (NODE_POSITIONS_X['grant'], NODE_POSITIONS_Y['grant'])

    # Position publications
    publications = network_nodes[network_nodes['node_type'] == NODE_TYPE_PUBLICATION]
    grant_funded_pubs = [p for p in publications['node_id'] if p.startswith('PUB_')]
    treatment_pathway_pubs = [p for p in publications['node_id'] if p.startswith('TREAT_PUB_')]
    ecosystem_pubs = [p for p in publications['node_id'] if p.startswith('ECO_')]

    for i, pub_id in enumerate(grant_funded_pubs):
        node_positions[pub_id] = (NODE_POSITIONS_X['grant_funded_pub'], NODE_POSITIONS_Y['grant_funded_pub'][i % len(NODE_POSITIONS_Y['grant_funded_pub'])])

    for i, pub_id in enumerate(treatment_pathway_pubs):
        node_positions[pub_id] = (NODE_POSITIONS_X['treatment_pathway_pub'], NODE_POSITIONS_Y['treatment_pathway_pub'][i % len(NODE_POSITIONS_Y['treatment_pathway_pub'])])

    # --- Ecosystem publications positioned by year ---
    ecosystem_pubs_df = publications[publications['node_id'].str.startswith('ECO_')]

    if not ecosystem_pubs_df.empty:
        min_year = ecosystem_pubs_df['year'].min()
        max_year = ecosystem_pubs_df['year'].max()

        for _, node in ecosystem_pubs_df.iterrows():
            year = node.get('year', np.nan)

            if pd.notna(year):
                # Map publication year to horizontal position between grant and treatment areas
                x_pos = np.interp(
                    year,
                    [min_year, max_year],
                    [NODE_POSITIONS_X['grant_funded_pub'] + 0.5,
                     NODE_POSITIONS_X['treatment_pathway_pub'] - 0.5]
                )
            else:
                # Fallback for missing year
                x_pos = np.random.uniform(
                    NODE_POSITIONS_X['grant_funded_pub'] + 0.5,
                    NODE_POSITIONS_X['treatment_pathway_pub'] - 0.5
                )

            # Add small vertical jitter for natural dispersion
            y_pos = np.random.uniform(-1.5, 1.5)

            node_positions[node['node_id']] = (
                x_pos + np.random.normal(0, 0.2),
                y_pos + np.random.normal(0, 0.2)
            )


    # Position treatment
    treatments = network_nodes[network_nodes['node_type'] == NODE_TYPE_TREATMENT]
    if not treatments.empty:
        node_positions[treatments.iloc[0]['node_id']] = (NODE_POSITIONS_X['treatment'], NODE_POSITIONS_Y['treatment'])

    return node_positions

def create_edge_trace(edges, node_positions, edge_type, name, showlegend):
    """Create a Plotly scatter trace for edges."""
    edge_x, edge_y = [], []
    for _, edge in edges.iterrows():
        if edge['source_id'] in node_positions and edge['target_id'] in node_positions:
            x0, y0 = node_positions[edge['source_id']]
            x1, y1 = node_positions[edge['target_id']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    if not edge_x:
        return None

    return go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=EDGE_WIDTHS[edge_type], color=EDGE_COLORS[edge_type]),
        hoverinfo='none',
        mode='lines',
        name=name,
        showlegend=showlegend
    )

def create_node_trace(nodes, node_positions, node_type, name, text_template, showlegend):
    """Create a Plotly scatter trace for nodes with detailed hover text."""
    node_x, node_y, hover_texts = [], [], []

    for _, node in nodes.iterrows():
        node_id = node['node_id']
        if node_id not in node_positions:
            continue

        x, y = node_positions[node_id]
        node_x.append(x)
        node_y.append(y)

        # --- Create richer hover content ---
        if node_type == 'grant_funded_pub' or node_id.startswith('PUB_'):
            hover_text = (
                f"<b>Grant-Funded Publication</b><br>"
                f"<b>Title:</b> {node.get('title', 'N/A')}<br>"
                f"<b>Year:</b> {int(node['year']) if pd.notna(node.get('year')) else 'N/A'}<br>"
                f"<b>PubMed ID:</b> {int(node['pmid']) if pd.notna(node.get('pmid')) else 'N/A'}<br>"
                f"<b>Authors:</b> {node.get('authors', 'N/A')}"
            )

        elif node_type == 'treatment_pathway_pub' or node_id.startswith('TREAT_PUB_'):
            hover_text = (
                f"<b>Treatment Pathway Paper</b><br>"
                f"<b>Title:</b> {node.get('title', 'N/A')}<br>"
                f"<b>Year:</b> {int(node['year']) if pd.notna(node.get('year')) else 'N/A'}<br>"
                f"<b>Journal:</b> {node.get('journal', 'N/A')}<br>"
                f"<b>Authors:</b> {node.get('authors', 'N/A')}"
            )

        elif node_type == 'ecosystem_pub' or node_id.startswith('ECO_'):
            hover_text = (
                f"<b>Research Ecosystem</b><br>"
                f"<b>Title:</b> {node.get('title', 'N/A')}<br>"
                f"<b>Year:</b> {int(node['year']) if pd.notna(node.get('year')) else 'N/A'}<br>"
                f"<b>Journal:</b> {node.get('journal', 'N/A')}<br>"
                f"<b>Authors:</b> {node.get('authors', 'N/A')}"
            )

        elif node_type == 'grant':
            hover_text = (
                f"<b>Research Grant</b><br>"
                f"<b>ID:</b> {node.get('grant_id', 'N/A')}<br>"
                f"<b>PI:</b> {node.get('pi_name', 'N/A')}<br>"
                f"<b>Funding:</b> ${int(node['funding_amount']) if pd.notna(node.get('funding_amount')) else 'N/A'}<br>"
                f"<b>Disease:</b> {node.get('disease', 'N/A')}"
            )

        elif node_type == 'treatment':
            hover_text = (
                f"<b>Approved Treatment</b><br>"
                f"<b>Name:</b> {node.get('treatment_name', 'N/A')}<br>"
                f"<b>Approval Year:</b> {int(node['approval_year']) if pd.notna(node.get('approval_year')) else 'N/A'}"
            )

        else:
            hover_text = text_template

        hover_texts.append(hover_text)

    return go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=hover_texts,   # <-- dynamic content here
        marker=dict(size=NODE_SIZES[node_type],
                    color=NODE_COLORS[node_type],
                    line=dict(width=2, color='#e2e8f0')),
        name=name,
        showlegend=showlegend
    )


def create_network_visualization(nodes_df, edges_df, network_id, grant_id=None, treatment_name=None):
    """Create the network visualization."""
    network_nodes = nodes_df[nodes_df['network_id'] == network_id]
    network_edges = edges_df[edges_df['network_id'] == network_id]

    if network_nodes.empty:
        st.error(f"No data found for network {network_id}")
        return go.Figure()

    # Get grant ID and treatment name from the network data if not provided
    if grant_id is None:
        grant_node = network_nodes[network_nodes['node_type'] == NODE_TYPE_GRANT]
        grant_id = grant_node.iloc[0]['node_id'] if not grant_node.empty else f"Network {network_id}"
    
    if treatment_name is None:
        treatment_node = network_nodes[network_nodes['node_type'] == NODE_TYPE_TREATMENT]
        treatment_name = treatment_node.iloc[0]['node_id'] if not treatment_node.empty else "Treatment"

    node_positions = get_node_positions(network_nodes, network_id)

    edge_traces = [
        create_edge_trace(network_edges[network_edges['edge_type'] == EDGE_TYPE_FUNDED_BY], node_positions, EDGE_TYPE_FUNDED_BY, 'Grant Funding', True),
        create_edge_trace(network_edges[network_edges['edge_type'] == EDGE_TYPE_LEADS_TO_TREATMENT], node_positions, EDGE_TYPE_LEADS_TO_TREATMENT, 'Research Impact Pathway', True),
        create_edge_trace(network_edges[network_edges['edge_type'] == EDGE_TYPE_CITES], node_positions, EDGE_TYPE_CITES, 'Citation', False),
        create_edge_trace(network_edges[network_edges['edge_type'] == EDGE_TYPE_ENABLES_TREATMENT], node_positions, EDGE_TYPE_ENABLES_TREATMENT, 'Treatment Enablement', True),
    ]

    node_traces = [
        create_node_trace(network_nodes[network_nodes['node_type'] == NODE_TYPE_GRANT], node_positions, NODE_TYPE_GRANT, 'Grant', 'Research Grant<br>Funding Source', True),
        create_node_trace(network_nodes[network_nodes['node_id'].str.startswith('PUB_')], node_positions, 'grant_funded_pub', 'Grant-Funded Papers', 'Grant-Funded Paper', True),
        create_node_trace(network_nodes[network_nodes['node_id'].str.startswith('TREAT_PUB_')], node_positions, 'treatment_pathway_pub', 'Treatment Approval Papers', 'Treatment Development Paper', True),
        create_node_trace(network_nodes[network_nodes['node_id'].str.startswith('ECO_')], node_positions, 'ecosystem_pub', 'Research Ecosystem', 'Research Ecosystem<br>Supporting Literature', True),
        create_node_trace(network_nodes[network_nodes['node_type'] == NODE_TYPE_TREATMENT], node_positions, NODE_TYPE_TREATMENT, 'Approved Treatment', 'Approved Treatment<br>Clinical Application', True),
    ]

    fig = go.Figure(data=[t for t in edge_traces + node_traces if t is not None])

    fig.update_layout(
        title={'text': f"Research Impact Network - {grant_id} → {treatment_name}", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20, 'color': '#e2e8f0', 'family': 'Inter, sans-serif'}},
        showlegend=True,
        hovermode='closest',
        margin=dict(b=40, l=40, r=40, t=70),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-6, 7]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]),
        height=600,
        plot_bgcolor='rgba(14, 17, 23, 0)',
        paper_bgcolor='rgba(14, 17, 23, 0)',
        font=dict(color='#e2e8f0'),
        legend=dict( orientation="v",yanchor="top",y=-0.15, xanchor="center", x=0.5, bgcolor="rgba(45, 55, 72, 0.9)", bordercolor="rgba(74, 85, 104, 0.5)", borderwidth=1, font=dict(size=11, color='#e2e8f0'))
    )
    
    fig.update_layout(
    template=None,
    paper_bgcolor='rgba(14, 17, 23, 1)',  # solid dark
    plot_bgcolor='rgba(14, 17, 23, 1)'    # solid dark
    )
    return fig

# --- UI Components ---
def display_network_metrics(summary_df, edges_df, network_id):
    """Display key metrics for the selected network"""
    try:
        network_summary = summary_df[summary_df['network_id'] == network_id].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Publications", value=f"{network_summary['total_publications']}", delta="Research papers")
        with col2:
            st.metric(label="Research Duration", value=f"{network_summary['research_duration']} years", delta=f"Approved {network_summary['approval_year']}")

    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">Research Impact Network Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #a0aec0; margin-bottom: 3rem; font-weight: 300;">Mapping Research Pathways from Grant Funding to Breakthrough Treatments</p>', unsafe_allow_html=True)

    # Load data
    nodes_df, edges_df, summary_df = load_database()

    st.sidebar.markdown("### Database Statistics")
    st.sidebar.write(f"Total connections: {len(edges_df)}")
    st.sidebar.write(f"Treatment pathways: {len(edges_df[edges_df['edge_type'] == EDGE_TYPE_LEADS_TO_TREATMENT])}")

    if summary_df.empty:
        st.error("No data available. Please check your database files.")
        return

    st.markdown("## Select the Citation Network")
    
    # Compact search controls
    with st.container():
        st.markdown('<div class="search-controls">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            search_type = st.selectbox("Search by:", ["Disease", "Treatment", "Grant"], key="search_type")
        
        with col2:
            if search_type == "Disease":
                search_options = summary_df['disease'].unique().tolist()
                search_placeholder = "Select a disease..."
            elif search_type == "Treatment":
                search_options = summary_df['treatment_name'].unique().tolist()
                search_placeholder = "Select a treatment..."
            else:  # Grant
                search_options = summary_df['grant_id'].unique().tolist()
                search_placeholder = "Select a grant..."

            selected_search = st.selectbox(f"Select {search_type.lower()}:", [""] + search_options, key="search_selection", format_func=lambda x: search_placeholder if x == "" else x)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Only show networks if a search selection has been made
    if selected_search and selected_search != "":
        if search_type == "Disease":
            filtered_networks = summary_df[summary_df['disease'] == selected_search]
        elif search_type == "Treatment":
            filtered_networks = summary_df[summary_df['treatment_name'] == selected_search]
        else:  # Grant
            filtered_networks = summary_df[summary_df['grant_id'] == selected_search]

        if not filtered_networks.empty:
            st.markdown("### Available Research Networks")
            
            # Create a more compact grid layout
            num_networks = len(filtered_networks)
            if num_networks <= 3:
                cols = st.columns(num_networks)
            else:
                cols = st.columns(3)
            
            selected_network = None

            for i, (_, network) in enumerate(filtered_networks.iterrows()):
                with cols[i % len(cols)]:
                    with st.container():
                        st.markdown(f"""<div class="selection-card grant-card">
                            <div class="network-title">{network['disease']}</div>
                            <div class="treatment-name">{network['treatment_name']}</div>
                            <div class="network-details">Grant ID: {network['grant_id']}<br>Duration: {network['research_duration']} years</div>
                        </div>""", unsafe_allow_html=True)
                        if st.button(f"Analyze Citation Network", key=f"btn_{network['network_id']}", use_container_width=True):
                            selected_network = network['network_id']
        else:
            st.info("No networks found for the selected criteria.")
    else:
        # Show instruction message when no selection is made
        st.info("👆 Please select a disease, treatment, or grant from the dropdown above to view available research networks.")
        selected_network = None

    # Handle network selection and visualization
    if selected_network:
        if 'selected_network' not in st.session_state:
            st.session_state.selected_network = selected_network
        else:
            st.session_state.selected_network = selected_network

    # Only show network analysis if a network has been selected
    if 'selected_network' in st.session_state and st.session_state.selected_network:
        network_id = st.session_state.selected_network
        
        # Verify the network still exists in the current filtered data
        if selected_search and selected_search != "":
            # Use filtered networks
            if search_type == "Disease":
                current_networks = summary_df[summary_df['disease'] == selected_search]
            elif search_type == "Treatment":
                current_networks = summary_df[summary_df['treatment_name'] == selected_search]
            else:  # Grant
                current_networks = summary_df[summary_df['grant_id'] == selected_search]
            
            if network_id in current_networks['network_id'].values:
                selected_summary = current_networks[current_networks['network_id'] == network_id].iloc[0]
                
                st.markdown(f"## Citation Network: {selected_summary['disease']} Research Impact")
                display_network_metrics(summary_df, edges_df, network_id)

                st.markdown("### 🕸️ Research Network Visualization")
                with st.spinner("Creating network visualization..."):
                    fig = create_network_visualization(nodes_df, edges_df, network_id, 
                                                     grant_id=selected_summary['grant_id'], 
                                                     treatment_name=selected_summary['treatment_name'])
                    if fig.data:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Unable to create network visualization")
                # --- Citation Explorer Section ---
                st.markdown("### 🔍 Explore Direct Citations Between Publications")
                
                # Get the network’s ecosystem and treatment-pathway papers
                network_nodes = nodes_df[nodes_df['network_id'] == network_id]
                network_edges = edges_df[edges_df['network_id'] == network_id]
                
                # Identify direct citation edges (ecosystem or treatment papers citing funded ones)
                direct_citations = network_edges[
                    (network_edges['edge_type'] == EDGE_TYPE_CITES) &
                    (network_edges['target_id'].str.startswith('PUB_')) &
                    (network_edges['source_id'].str.startswith(('ECO_', 'TREAT_PUB_')))
                ]
                
                if not direct_citations.empty:
                    st.write(f"Found **{len(direct_citations)}** ecosystem or treatment papers that directly cited grant-funded papers.")
                
                    # Create a sliding window container for each citing paper
                    for _, edge in direct_citations.iterrows():
                        citing_paper = network_nodes[network_nodes['node_id'] == edge['source_id']].iloc[0]
                        cited_paper = network_nodes[network_nodes['node_id'] == edge['target_id']].iloc[0]
                
                        with st.expander(f"📄 {citing_paper.get('title', 'Untitled')} ({int(citing_paper['year']) if pd.notna(citing_paper.get('year')) else 'N/A'})"):
                            st.markdown(f"**Journal:** {citing_paper.get('journal', 'N/A')}")
                            st.markdown(f"**Authors:** {citing_paper.get('authors', 'N/A')}")
                            st.markdown(f"**PubMed ID:** {int(citing_paper['pmid']) if pd.notna(citing_paper.get('pmid')) else 'N/A'}")
                            st.markdown("---")
                            st.markdown(f"🧩 **Cites grant-funded paper:** {cited_paper.get('title', 'N/A')} ({int(cited_paper['year']) if pd.notna(cited_paper.get('year')) else 'N/A'})")
                else:
                    st.info("No direct ecosystem or treatment citations to grant-funded papers found in this network.")

            else:
                # Selected network is not in current filter, clear selection
                st.session_state.selected_network = None
        else:
            # No search selection made, don't show network analysis
            pass

if __name__ == "__main__":
    main()

