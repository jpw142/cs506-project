import sys
import csv
import json
import pandas as pd
import textwrap
import plotly.graph_objects as go
import numpy as np
from plotly.colors import qualitative

def wrap_text(text, width=70):
    wrapped = textwrap.wrap(text, width=width)
    return '<br>'.join(wrapped)[:width * 3] + "..."

def load_contract_info(contract_csv, ids):
    id_to_name, id_to_desc = {}, {}
    with open(contract_csv, newline='', encoding='latin1') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            id_to_name[row['NoticeId']] = row['Title']
            id_to_desc[row['NoticeId']] = row.get('Description', '')
    cleaned = [id_.replace('CAP:', '') for id_ in ids]
    names = [id_to_name.get(cid, "Unknown") for cid in cleaned]
    descs = [id_to_desc.get(cid, "") for cid in cleaned]
    return names, descs

def plot_clusters_interactive(reduced_embeddings, labels, ids, names, descs, output_html):
    df = pd.DataFrame({
        'UMAP-1': reduced_embeddings[:, 0],
        'UMAP-2': reduced_embeddings[:, 1],
        'Cluster': labels,
        'ID': ids,
        'Name': names,
        'Description': descs
    })

    df['IsCapability'] = df['ID'].str.startswith('CAP:')
    unique_clusters = sorted(df['Cluster'].unique())
    colors = qualitative.Plotly
    fig = go.Figure()

    for i, cluster in enumerate(unique_clusters):
        pts = df[(df['Cluster'] == cluster) & (~df['IsCapability'])]
        if pts.empty:
            continue
        color = colors[i % len(colors)]
        label = "Noise" if cluster == -1 else f"Cluster {cluster}"
        hover_texts = [
            f"ID: {id_}<br>Name: {name}<br><br>Description:<br>{wrap_text(desc)}<br><br>Cluster: {cluster}"
            for id_, name, desc in zip(pts['ID'], pts['Name'], pts['Description'])
        ]
        fig.add_trace(go.Scattergl(
            x=pts['UMAP-1'],
            y=pts['UMAP-2'],
            mode='markers',
            marker=dict(size=8 if cluster != -1 else 7, color=color, opacity=0.6 if cluster != -1 else 0.3),
            name=label,
            text=hover_texts,
            hoverinfo='text',
            showlegend=True
        ))

    cap_pts = df[df['IsCapability']]
    if not cap_pts.empty:
        hover_texts = [
            f"Capability: {id_.replace('CAP:', '')}<br><br>Description:<br>{wrap_text(desc)}<br><br>Cluster: {cluster}"
            for id_, desc, cluster in zip(cap_pts['ID'], cap_pts['Description'], cap_pts['Cluster'])
        ]
        fig.add_trace(go.Scattergl(
            x=cap_pts['UMAP-1'],
            y=cap_pts['UMAP-2'],
            mode='markers',
            marker=dict(symbol='diamond', size=16, color='black', line=dict(width=1, color='white')),
            name="Capabilities (CAP)",
            text=hover_texts,
            hoverinfo='text',
            showlegend=True
        ))

    fig.update_layout(
        title='HDBSCAN Clusters (2D UMAP Projection)',
        xaxis_title='UMAP-1',
        yaxis_title='UMAP-2',
        legend=dict(itemsizing='constant', orientation='v', yanchor='top', y=1, xanchor='left', x=1.05),
        width=900,
        height=600,
        hovermode='closest',
        hoverlabel=dict(bgcolor='white', align='left', namelength=-1)
    )

    fig.write_html(output_html)
    print(f"Plot saved to {output_html}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python plotting.py <input_json> <cluster_json> <csv_file> <output_html>")
        sys.exit(1)

    input_json, cluster_json, csv_file, output_html = sys.argv[1:]

    print("Opening Data")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    ids = list(data.keys())
    embeddings = np.array([data[id_] for id_ in ids])
    
    with open(cluster_json, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
    labels = [cluster_data[id_] for id_ in ids]

    print("Reducing Dimensionality")
    import umap
    reducer = umap.UMAP(n_components=2)
    reduced = reducer.fit_transform(embeddings)

    print("Formatting")
    names, descs = load_contract_info(csv_file, ids)
    plot_clusters_interactive(reduced, labels, ids, names, descs, output_html)

if __name__ =="__main__":
    main()
 
