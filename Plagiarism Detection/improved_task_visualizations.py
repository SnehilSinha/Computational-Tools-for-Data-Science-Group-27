"""
Improved Visualization Methods for Task-Specific Clustering
===========================================================

This module provides several enhanced visualization techniques for 
better understanding of task-specific plagiarism clustering results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def enhanced_similarity_heatmap(task_dfs):
    """
    Create similarity heatmaps showing document-to-document similarity within each task
    """
    n_tasks = len(task_dfs)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 4))
    if n_tasks == 1:
        axes = [axes]
    
    for idx, task_df in enumerate(task_dfs):
        task_name = task_df['Task'].iloc[0]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(task_df['clean_text'])
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create heatmap
        im = axes[idx].imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        
        # Add labels with file names (shortened) and categories
        labels = [f"{row['File'][:10]}({row['Category']})" 
                 for _, row in task_df.iterrows()]
        
        axes[idx].set_xticks(range(len(labels)))
        axes[idx].set_yticks(range(len(labels)))
        axes[idx].set_xticklabels(labels, rotation=45, ha='right')
        axes[idx].set_yticklabels(labels)
        axes[idx].set_title(f'Task {task_name.upper()} - Document Similarity')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    plt.tight_layout()
    plt.show()

def interactive_3d_clustering(task_dfs):
    """
    Create interactive 3D scatter plots for each task using plotly
    """
    for task_df in task_dfs:
        task_name = task_df['Task'].iloc[0]
        
        # Create features: TF-IDF + similarity to original + document stats
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(task_df['clean_text'])
        
        # Find original document and calculate similarities
        orig_mask = task_df['Category'] == 'orig'
        if orig_mask.any():
            orig_tfidf = tfidf_matrix[orig_mask]
            similarities = cosine_similarity(orig_tfidf, tfidf_matrix)[0]
        else:
            similarities = np.zeros(len(task_df))
        
        # Document statistics
        doc_lengths = task_df['clean_text'].str.len()
        unique_words = task_df['clean_text'].apply(lambda x: len(set(x.split())))
        
        # Use PCA for 3D visualization
        if tfidf_matrix.shape[1] >= 3:
            pca = PCA(n_components=3, random_state=42)
            coords_3d = pca.fit_transform(tfidf_matrix.toarray())
        else:
            coords_3d = np.column_stack([
                similarities,
                doc_lengths / doc_lengths.max(),
                unique_words / unique_words.max()
            ])
        
        # Create interactive 3D plot
        fig = go.Figure()
        
        colors = {'orig': 'black', 'non': 'blue', 'heavy': 'red', 
                 'light': 'orange', 'cut': 'green'}
        
        for category in task_df['Category'].unique():
            mask = task_df['Category'] == category
            fig.add_trace(go.Scatter3d(
                x=coords_3d[mask, 0],
                y=coords_3d[mask, 1], 
                z=coords_3d[mask, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors.get(category, 'gray'),
                    opacity=0.8
                ),
                name=category,
                text=[f"File: {row['File']}<br>Category: {row['Category']}<br>Similarity: {sim:.3f}" 
                      for _, row, sim in zip(task_df[mask].iterrows(), task_df[mask].iterrows(), similarities[mask])],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Add cluster boundaries if available
        if 'task_cluster' in task_df.columns:
            for cluster_id in task_df['task_cluster'].unique():
                cluster_mask = task_df['task_cluster'] == cluster_id
                cluster_coords = coords_3d[cluster_mask]
                if len(cluster_coords) > 1:
                    center = cluster_coords.mean(axis=0)
                    fig.add_trace(go.Scatter3d(
                        x=[center[0]], y=[center[1]], z=[center[2]],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='x'),
                        name=f'Cluster {cluster_id} Center',
                        showlegend=False
                    ))
        
        fig.update_layout(
            title=f'Task {task_name.upper()} - 3D Clustering Visualization',
            scene=dict(
                xaxis_title='PC1/Similarity',
                yaxis_title='PC2/Length',
                zaxis_title='PC3/Uniqueness'
            ),
            height=600
        )
        
        fig.show()

def similarity_distribution_plots(task_dfs):
    """
    Create distribution plots showing similarity patterns for each task
    """
    n_tasks = len(task_dfs)
    fig, axes = plt.subplots(2, n_tasks, figsize=(4*n_tasks, 8))
    if n_tasks == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, task_df in enumerate(task_dfs):
        task_name = task_df['Task'].iloc[0]
        
        # Calculate similarities to original
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(task_df['clean_text'])
        
        orig_mask = task_df['Category'] == 'orig'
        if orig_mask.any():
            orig_tfidf = tfidf_matrix[orig_mask]
            similarities = cosine_similarity(orig_tfidf, tfidf_matrix)[0]
            task_df_with_sim = task_df.copy()
            task_df_with_sim['similarity'] = similarities
        else:
            continue
        
        # Top plot: Distribution by category
        ax1 = axes[0, idx]
        categories = ['non', 'cut', 'light', 'heavy']
        colors = ['blue', 'green', 'orange', 'red']
        
        for category, color in zip(categories, colors):
            cat_data = task_df_with_sim[task_df_with_sim['Category'] == category]
            if len(cat_data) > 0:
                ax1.hist(cat_data['similarity'], alpha=0.7, color=color, 
                        label=f'{category} (n={len(cat_data)})', bins=10, density=True)
        
        ax1.set_title(f'Task {task_name.upper()} - Similarity Distribution')
        ax1.set_xlabel('Similarity to Original')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Box plot by category
        ax2 = axes[1, idx]
        category_data = []
        category_labels = []
        
        for category in categories:
            cat_data = task_df_with_sim[task_df_with_sim['Category'] == category]
            if len(cat_data) > 0:
                category_data.append(cat_data['similarity'])
                category_labels.append(f'{category}\n(n={len(cat_data)})')
        
        if category_data:
            bp = ax2.boxplot(category_data, labels=category_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(category_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax2.set_title(f'Task {task_name.upper()} - Similarity by Category')
        ax2.set_ylabel('Similarity to Original')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def cluster_performance_dashboard(task_dfs):
    """
    Create a comprehensive dashboard showing clustering performance
    """
    n_tasks = len(task_dfs)
    
    # Calculate metrics for each task
    task_metrics = []
    
    for task_df in task_dfs:
        if 'task_cluster' not in task_df.columns:
            continue
            
        task_name = task_df['Task'].iloc[0]
        
        # Calculate cluster purity and other metrics
        clusters = task_df['task_cluster'].unique()
        cluster_purities = []
        cluster_sizes = []
        
        for cluster_id in clusters:
            cluster_data = task_df[task_df['task_cluster'] == cluster_id]
            
            # Purity: fraction of most common category in cluster
            most_common_category = cluster_data['Category'].mode()[0]
            purity = (cluster_data['Category'] == most_common_category).mean()
            
            cluster_purities.append(purity)
            cluster_sizes.append(len(cluster_data))
            
            # Plagiarism detection accuracy
            is_plagiarized = cluster_data['Category'].isin(['cut', 'light', 'heavy'])
            plagiarism_ratio = is_plagiarized.mean()
        
        avg_purity = np.mean(cluster_purities)
        avg_size = np.mean(cluster_sizes)
        
        task_metrics.append({
            'task': task_name,
            'n_clusters': len(clusters),
            'avg_purity': avg_purity,
            'avg_size': avg_size,
            'purities': cluster_purities,
            'sizes': cluster_sizes
        })
    
    # Create dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top-left: Average purity by task
    tasks = [m['task'] for m in task_metrics]
    purities = [m['avg_purity'] for m in task_metrics]
    
    axes[0, 0].bar(tasks, purities, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Average Cluster Purity by Task')
    axes[0, 0].set_ylabel('Purity')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(purities):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Top-right: Number of clusters by task
    n_clusters = [m['n_clusters'] for m in task_metrics]
    axes[0, 1].bar(tasks, n_clusters, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Number of Clusters by Task')
    axes[0, 1].set_ylabel('Number of Clusters')
    for i, v in enumerate(n_clusters):
        axes[0, 1].text(i, v + 0.05, str(v), ha='center', va='bottom')
    
    # Bottom-left: Purity distribution across all clusters
    all_purities = []
    for m in task_metrics:
        all_purities.extend(m['purities'])
    
    axes[1, 0].hist(all_purities, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribution of Cluster Purities')
    axes[1, 0].set_xlabel('Purity')
    axes[1, 0].set_ylabel('Number of Clusters')
    axes[1, 0].axvline(np.mean(all_purities), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_purities):.3f}')
    axes[1, 0].legend()
    
    # Bottom-right: Cluster size distribution
    all_sizes = []
    for m in task_metrics:
        all_sizes.extend(m['sizes'])
    
    axes[1, 1].hist(all_sizes, bins=10, color='gold', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution of Cluster Sizes')
    axes[1, 1].set_xlabel('Cluster Size')
    axes[1, 1].set_ylabel('Number of Clusters')
    axes[1, 1].axvline(np.mean(all_sizes), color='red', linestyle='--',
                      label=f'Mean: {np.mean(all_sizes):.1f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return task_metrics

def network_similarity_visualization(task_dfs):
    """
    Create network graphs showing document relationships within each task
    """
    import networkx as nx
    from matplotlib.patches import Circle
    
    n_tasks = len(task_dfs)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]
    
    for idx, task_df in enumerate(task_dfs):
        task_name = task_df['Task'].iloc[0]
        
        # Calculate similarity matrix
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(task_df['clean_text'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i, (_, row) in enumerate(task_df.iterrows()):
            G.add_node(i, 
                      file=row['File'][:10], 
                      category=row['Category'],
                      cluster=row.get('task_cluster', 0))
        
        # Add edges for high similarity (threshold = 0.3)
        threshold = 0.3
        for i in range(len(task_df)):
            for j in range(i+1, len(task_df)):
                if similarity_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Color nodes by category
        colors = {'orig': 'black', 'non': 'blue', 'heavy': 'red', 
                 'light': 'orange', 'cut': 'green'}
        node_colors = [colors.get(task_df.iloc[node]['Category'], 'gray') 
                      for node in G.nodes()]
        
        # Draw network
        ax = axes[idx]
        
        # Draw edges with thickness proportional to similarity
        edges = G.edges()
        edge_weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in edge_weights], 
                              alpha=0.5, edge_color='gray', ax=ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=300, alpha=0.8, ax=ax)
        
        # Add labels
        labels = {i: task_df.iloc[i]['File'][:8] for i in range(len(task_df))}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Task {task_name.upper()} - Document Similarity Network\n'
                    f'(edges: similarity > {threshold})')
        ax.axis('off')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=8, label=cat)
                          for cat, color in colors.items() 
                          if cat in task_df['Category'].values]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Example usage function
def demonstrate_improved_visualizations(task_dfs):
    """
    Demonstrate all improved visualization techniques
    """
    print("🎨 ENHANCED TASK-SPECIFIC CLUSTERING VISUALIZATIONS")
    print("=" * 60)
    
    print("\n1. 📊 Document Similarity Heatmaps")
    print("-" * 35)
    enhanced_similarity_heatmap(task_dfs)
    
    print("\n2. 📈 Similarity Distribution Analysis")
    print("-" * 35)
    similarity_distribution_plots(task_dfs)
    
    print("\n3. 📋 Clustering Performance Dashboard")
    print("-" * 35)
    cluster_performance_dashboard(task_dfs)
    
    print("\n4. 🕸️  Document Similarity Networks")
    print("-" * 35)
    network_similarity_visualization(task_dfs)
    
    print("\n5. 🌐 Interactive 3D Clustering (Plotly)")
    print("-" * 35)
    print("Note: Interactive 3D plots will open in browser/notebook")
    interactive_3d_clustering(task_dfs)
    
    print("\n✅ All enhanced visualizations completed!")

if __name__ == "__main__":
    print("Enhanced Task-Specific Clustering Visualization Module")
    print("Use demonstrate_improved_visualizations(task_dfs) to see all methods")