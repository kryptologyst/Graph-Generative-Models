"""Interactive demo for graph generative models using Streamlit."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import torch
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import load_dataset, generate_sbm_graph, generate_barabasi_albert_graph
from src.models import create_model
from src.utils import get_device, set_seed


def load_model(model_path: str, model_type: str, in_channels: int) -> torch.nn.Module:
    """Load a trained model.
    
    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model
        in_channels: Number of input features
        
    Returns:
        torch.nn.Module: Loaded model
    """
    checkpoint = torch.load(model_path, map_location=get_device())
    
    # Create model
    model = create_model(
        model_type=model_type,
        in_channels=in_channels,
        hidden_channels=[128, 64],
        latent_dim=64,
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


def visualize_graph(
    G: nx.Graph,
    title: str = "Graph",
    layout: str = "spring",
    node_size: int = 10,
    edge_width: float = 0.5,
) -> go.Figure:
    """Visualize a graph using Plotly.
    
    Args:
        G: NetworkX graph
        title: Title of the plot
        layout: Layout algorithm
        node_size: Size of nodes
        edge_width: Width of edges
        
    Returns:
        go.Figure: Plotly figure
    """
    if G.number_of_nodes() == 0:
        # Return empty plot
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Extract edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Extract nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [f"Node {node}" for node in G.nodes()]
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=edge_width, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            reversescale=True,
            color=[],
            size=node_size,
            colorbar=dict(
                thickness=15,
                xanchor="left",
                titleside="right"
            ),
            line=dict(width=2)
        )
    )
    
    # Color nodes by degree
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(G.degree(node))
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = [f"Node {node}<br>Degree: {deg}" for node, deg in zip(G.nodes(), node_adjacencies)]
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Interactive graph visualization",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(color="#888", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    )
    
    return fig


def visualize_embeddings(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    method: str = "tsne",
    title: str = "Node Embeddings",
) -> go.Figure:
    """Visualize node embeddings.
    
    Args:
        embeddings: Node embeddings
        labels: Node labels (optional)
        method: Dimensionality reduction method
        title: Title of the plot
        
    Returns:
        go.Figure: Plotly figure
    """
    # Convert to numpy
    emb_np = embeddings.cpu().numpy()
    
    # Reduce dimensionality
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    emb_2d = reducer.fit_transform(emb_np)
    
    # Create plot
    if labels is not None:
        labels_np = labels.cpu().numpy()
        fig = px.scatter(
            x=emb_2d[:, 0],
            y=emb_2d[:, 1],
            color=labels_np,
            title=title,
            labels={"x": f"{method.upper()} 1", "y": f"{method.upper()} 2"},
        )
    else:
        fig = px.scatter(
            x=emb_2d[:, 0],
            y=emb_2d[:, 1],
            title=title,
            labels={"x": f"{method.upper()} 1", "y": f"{method.upper()} 2"},
        )
    
    return fig


def generate_graph_interactive(
    model: torch.nn.Module,
    model_type: str,
    num_samples: int = 1,
) -> List[nx.Graph]:
    """Generate graphs interactively.
    
    Args:
        model: Trained model
        model_type: Type of model
        num_samples: Number of graphs to generate
        
    Returns:
        List[nx.Graph]: Generated graphs
    """
    generated_graphs = []
    
    with torch.no_grad():
        if model_type == "graphvae":
            for _ in range(num_samples):
                # Sample from prior
                z = torch.randn(1, 64, device=get_device())
                
                # Generate graph
                node_features, adj_matrix, node_probs = model.decode(z)
                
                # Convert to NetworkX graph
                G = adj_matrix_to_graph(adj_matrix, node_probs)
                generated_graphs.append(G)
        
        elif model_type == "vgae":
            # For VGAE, we can generate new edges for existing nodes
            # This is a simplified approach
            st.warning("VGAE generates edges for existing nodes, not new graphs")
            return []
        
        else:
            st.error(f"Graph generation not implemented for {model_type}")
            return []
    
    return generated_graphs


def adj_matrix_to_graph(
    adj_matrix: torch.Tensor,
    node_probs: torch.Tensor,
    threshold: float = 0.5,
) -> nx.Graph:
    """Convert adjacency matrix to NetworkX graph.
    
    Args:
        adj_matrix: Adjacency matrix
        node_probs: Node existence probabilities
        threshold: Threshold for edge existence
        
    Returns:
        nx.Graph: NetworkX graph
    """
    # Filter nodes based on existence probability
    node_mask = node_probs > threshold
    num_nodes = node_mask.sum().item()
    
    if num_nodes == 0:
        return nx.Graph()
    
    # Filter adjacency matrix
    adj_np = adj_matrix[node_mask][:, node_mask].cpu().numpy()
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Add edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_np[i, j] > threshold:
                G.add_edge(i, j)
    
    return G


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Graph Generative Models Demo",
        page_icon="🕸️",
        layout="wide",
    )
    
    st.title("🕸️ Graph Generative Models Demo")
    st.markdown("Interactive visualization and generation of graphs using neural networks")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["vgae", "graphvae", "graphrnn"],
        help="Select the type of graph generative model"
    )
    
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["cora", "citeseer", "pubmed", "sbm", "barabasi_albert"],
        help="Select the dataset to use"
    )
    
    # Model path
    model_path = st.sidebar.text_input(
        "Model Path",
        value="./outputs/best_model.pt",
        help="Path to the trained model checkpoint"
    )
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please train a model first using the training script")
        return
    
    # Load data
    st.sidebar.subheader("Data Loading")
    if st.sidebar.button("Load Dataset"):
        with st.spinner("Loading dataset..."):
            try:
                if dataset_name in ["cora", "citeseer", "pubmed"]:
                    data, _ = load_dataset(dataset_name)
                elif dataset_name == "sbm":
                    data = generate_sbm_graph()
                elif dataset_name == "barabasi_albert":
                    data = generate_barabasi_albert_graph()
                
                st.session_state.data = data
                st.success(f"Dataset {dataset_name} loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
    
    # Load model
    if "data" in st.session_state:
        st.sidebar.subheader("Model Loading")
        if st.sidebar.button("Load Model"):
            with st.spinner("Loading model..."):
                try:
                    model = load_model(
                        model_path,
                        model_type,
                        st.session_state.data.x.size(1)
                    )
                    st.session_state.model = model
                    st.success("Model loaded successfully!")
                    
                except Exception as e:
                    st.error(f"Error loading model: {e}")
    
    # Main content
    if "data" in st.session_state and "model" in st.session_state:
        data = st.session_state.data
        model = st.session_state.model
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔍 Embeddings", "🎲 Generation", "📈 Analysis"])
        
        with tab1:
            st.header("Data Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Graph Statistics")
                stats = {
                    "Nodes": data.num_nodes,
                    "Edges": data.edge_index.size(1),
                    "Features": data.x.size(1),
                    "Density": f"{2 * data.edge_index.size(1) / (data.num_nodes * (data.num_nodes - 1)):.4f}",
                }
                
                for key, value in stats.items():
                    st.metric(key, value)
            
            with col2:
                st.subheader("Graph Visualization")
                
                # Convert to NetworkX for visualization
                G = nx.Graph()
                G.add_nodes_from(range(data.num_nodes))
                G.add_edges_from(data.edge_index.t().cpu().numpy())
                
                # Layout selection
                layout = st.selectbox("Layout", ["spring", "circular", "random"])
                
                # Visualize
                fig = visualize_graph(G, title=f"{dataset_name.upper()} Graph", layout=layout)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Node Embeddings")
            
            # Generate embeddings
            with torch.no_grad():
                if model_type == "vgae":
                    z = model.encode(data.x, data.train_pos_edge_index)
                else:
                    z = model.encode(data.x, data.train_pos_edge_index)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Embedding Visualization")
                
                # Method selection
                method = st.selectbox("Reduction Method", ["tsne", "pca"])
                
                # Labels
                labels = data.y if hasattr(data, 'y') else None
                
                # Visualize
                fig = visualize_embeddings(z, labels, method)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Embedding Statistics")
                
                # Compute statistics
                emb_stats = {
                    "Dimension": z.size(1),
                    "Mean": z.mean().item(),
                    "Std": z.std().item(),
                    "Min": z.min().item(),
                    "Max": z.max().item(),
                }
                
                for key, value in emb_stats.items():
                    st.metric(key, f"{value:.4f}")
        
        with tab3:
            st.header("Graph Generation")
            
            if model_type in ["graphvae", "graphrnn"]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Generation Parameters")
                    
                    num_samples = st.slider("Number of Samples", 1, 10, 1)
                    threshold = st.slider("Edge Threshold", 0.1, 0.9, 0.5, 0.1)
                    
                    if st.button("Generate Graphs"):
                        with st.spinner("Generating graphs..."):
                            generated_graphs = generate_graph_interactive(model, model_type, num_samples)
                            st.session_state.generated_graphs = generated_graphs
                            st.success(f"Generated {len(generated_graphs)} graphs!")
                
                with col2:
                    st.subheader("Generated Graphs")
                    
                    if "generated_graphs" in st.session_state:
                        graphs = st.session_state.generated_graphs
                        
                        # Select graph to visualize
                        if graphs:
                            graph_idx = st.selectbox("Select Graph", range(len(graphs)))
                            G = graphs[graph_idx]
                            
                            # Visualize
                            fig = visualize_graph(
                                G,
                                title=f"Generated Graph {graph_idx + 1}",
                                node_size=15,
                                edge_width=1.0
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Statistics
                            st.write("**Graph Statistics:**")
                            st.write(f"- Nodes: {G.number_of_nodes()}")
                            st.write(f"- Edges: {G.number_of_edges()}")
                            st.write(f"- Density: {nx.density(G):.4f}")
                            st.write(f"- Clustering: {nx.average_clustering(G):.4f}")
            
            else:
                st.info(f"Graph generation not available for {model_type}")
        
        with tab4:
            st.header("Analysis")
            
            if "generated_graphs" in st.session_state and st.session_state.generated_graphs:
                st.subheader("Generated vs Original Graph Comparison")
                
                # Original graph
                G_orig = nx.Graph()
                G_orig.add_nodes_from(range(data.num_nodes))
                G_orig.add_edges_from(data.edge_index.t().cpu().numpy())
                
                # Generated graphs
                generated_graphs = st.session_state.generated_graphs
                
                # Compare statistics
                orig_stats = {
                    "Nodes": G_orig.number_of_nodes(),
                    "Edges": G_orig.number_of_edges(),
                    "Density": nx.density(G_orig),
                    "Clustering": nx.average_clustering(G_orig),
                }
                
                gen_stats = []
                for G in generated_graphs:
                    gen_stats.append({
                        "Nodes": G.number_of_nodes(),
                        "Edges": G.number_of_edges(),
                        "Density": nx.density(G),
                        "Clustering": nx.average_clustering(G),
                    })
                
                # Create comparison plot
                metrics = ["Nodes", "Edges", "Density", "Clustering"]
                
                fig = go.Figure()
                
                # Original
                fig.add_trace(go.Scatter(
                    x=metrics,
                    y=[orig_stats[m] for m in metrics],
                    mode="markers+lines",
                    name="Original",
                    marker=dict(size=10, color="red")
                ))
                
                # Generated
                for i, stats in enumerate(gen_stats):
                    fig.add_trace(go.Scatter(
                        x=metrics,
                        y=[stats[m] for m in metrics],
                        mode="markers+lines",
                        name=f"Generated {i+1}",
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    title="Graph Statistics Comparison",
                    xaxis_title="Metrics",
                    yaxis_title="Values",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Please load a dataset and model to start the demo")
        
        # Show instructions
        st.subheader("Instructions")
        st.markdown("""
        1. **Select Model Type**: Choose between VGAE, GraphVAE, or GraphRNN
        2. **Select Dataset**: Choose from available datasets
        3. **Load Dataset**: Click the "Load Dataset" button
        4. **Load Model**: Click the "Load Model" button
        5. **Explore**: Use the tabs to explore different aspects of the model
        
        **Note**: Make sure you have trained a model first using the training script.
        """)


if __name__ == "__main__":
    main()
