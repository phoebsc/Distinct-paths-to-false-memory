import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import sys


# Create a graph
G = nx.Graph()

# Define number of nodes
num_nodes = 8
nodes = list(range(num_nodes))

# Add all nodes
G.add_nodes_from(nodes)

# Define the central node
center_node = 0

# Add edges: central node connects to many others with high weights
for node in nodes:
    if node != center_node:
        weight = np.random.uniform(0.9, 1.0)  # Stronger connections for center
        G.add_edge(center_node, node, weight=weight)

# Add some random weaker connections between non-central nodes
for _ in range(num_nodes * 2):
    u, v = np.random.choice(nodes[1:], size=2, replace=False)
    if not G.has_edge(u, v):
        weight = np.random.uniform(0.2, 0.25)
        G.add_edge(u, v, weight=weight)

# Positioning nodes using spring layout
pos = nx.spring_layout(G, seed=42)

# Draw nodes with central node larger
node_sizes = [80 if n == center_node else np.random.uniform(20,30) for n in G.nodes()]

"""
plotting block
"""
# Plot using plotly
edge_traces = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    weight = G[edge[0]][edge[1]]['weight']
    transparency = weight
    edge_trace = go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        line=dict(width=1, color=f'rgba(50, 50, 50, {transparency})'),
        mode='lines')
    edge_traces.append(edge_trace)

node_x = [pos[i][0] for i in G.nodes()]
node_y = [pos[i][1] for i in G.nodes()]
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        size=node_sizes,
        color='#793547',
        opacity=1,  # Make markers solid
        line=dict(width=2, color='#793547'),
        colorbar=dict(thickness=15, title='Centrality', xanchor='left', titleside='right')
    )
)

fig = go.Figure(data=edge_traces + [node_trace],
                layout=go.Layout(
                    title='Network Graph based on Cosine Similarity',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    paper_bgcolor='white',  # White background
                    plot_bgcolor='white'  # White background for plot area
                ))

fig.write_image("./figures/svg/network_figure.svg")

"""
surprisal plot
"""
