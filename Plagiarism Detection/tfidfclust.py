import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class Tfidfclustering:
    def __init__(self, df):
        
        # This is the dataframe containing the cleaned text data
        self.df = df
        
        
    
    def tfidf_vectorization(self):
        # Limit features to avoid overfitting
        # Include unigrams and bigrams
        # Ignore terms appearing in less than 2 documents and more than 80% of documents
        vectorizer = TfidfVectorizer(
            max_features=1000,  
            stop_words='english',
            ngram_range=(1, 2),  
            min_df=2,  
            max_df=0.8  
        )

        # Fit and transform the cleaned text
        tfidf_matrix = vectorizer.fit_transform(self.df['clean_text'])
        #print(f"TF-IDF shape: {tfidf_matrix.shape}")

        # Convert to dense array for easier manipulation
        tfidf_dense = tfidf_matrix.toarray()
        feature_names = vectorizer.get_feature_names_out()
        #print(f"Num of features: {len(feature_names)}")
        return tfidf_dense, feature_names
    
    
    
    def perform_kmeans_clustering(self, X, n_clusters_range=range(2, 11), task_filter=None):
        """Perform K-means clustering and find optimal number of clusters"""
        results = []

        data_subset = X
        df_subset = self.df

        if task_filter:
            task_mask = self.df['Task'] == task_filter
            data_subset = X[task_mask]
            df_subset = self.df[task_mask]

        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data_subset)

            # Calculate metrics
            silhouette_avg = silhouette_score(data_subset, cluster_labels)
            inertia = kmeans.inertia_

            # Calculate clustering purity with respect to plagiarism categories
            if 'category_id' in df_subset.columns:
                ari = adjusted_rand_score(df_subset['category_id'].values, cluster_labels)
                nmi = normalized_mutual_info_score(df_subset['category_id'].values, cluster_labels)
            else:
                ari = nmi = None

            results.append({
                'n_clusters': n_clusters,
                'silhouette_score': silhouette_avg,
                'inertia': inertia,
                'ari': ari,
                'nmi': nmi,
                'labels': cluster_labels
            })

        return results, df_subset
    
    def hierarchical_clustering(self, X, n_clusters):
        """Perform hierarchical clustering with different linkage methods"""
        linkage_methods = ['ward', 'complete', 'average']
        hierarchical_results = []
        
        for linkage_method in linkage_methods:
            agglomerative = AgglomerativeClustering(
                n_clusters=n_clusters, 
                linkage=linkage_method
            )
            hier_labels = agglomerative.fit_predict(X)
            
            # Calculate metrics
            silhouette_avg = silhouette_score(X, hier_labels)
            ari = adjusted_rand_score(self.df['category_id'].values, hier_labels)
            nmi = normalized_mutual_info_score(self.df['category_id'].values, hier_labels)
            
            hierarchical_results.append({
                'linkage': linkage_method,
                'silhouette_score': silhouette_avg,
                'ari': ari,
                'nmi': nmi,
                'labels': hier_labels
            })
        
        return hierarchical_results
    
    def visualize_clustering(self, X, cluster_labels, title="Clustering Visualization"):
        """Create PCA and t-SNE visualizations of clustering results"""
        # PCA for dimensionality reduction
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        X_tsne = tsne.fit_transform(X)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # PCA plot colored by clusters
        scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                     c=cluster_labels, cmap='tab10', alpha=0.7)
        axes[0, 0].set_title(f'{title} (PCA)')
        axes[0, 0].set_xlabel('First Principal Component')
        axes[0, 0].set_ylabel('Second Principal Component')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # PCA plot colored by true categories
        category_colors = {'orig': 'black', 'non': 'blue', 'heavy': 'red', 'light': 'orange', 'cut': 'green'}
        for category, color in category_colors.items():
            mask = self.df['Category'] == category
            axes[0, 1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=color, label=category, alpha=0.7)
        axes[0, 1].set_title('True Categories (PCA)')
        axes[0, 1].set_xlabel('First Principal Component')
        axes[0, 1].set_ylabel('Second Principal Component')
        axes[0, 1].legend()
        
        # t-SNE plot colored by clusters
        scatter3 = axes[1, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                     c=cluster_labels, cmap='tab10', alpha=0.7)
        axes[1, 0].set_title(f'{title} (t-SNE)')
        axes[1, 0].set_xlabel('t-SNE Component 1')
        axes[1, 0].set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter3, ax=axes[1, 0])
        
        # t-SNE plot colored by true categories
        for category, color in category_colors.items():
            mask = self.df['Category'] == category
            axes[1, 1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                              c=color, label=category, alpha=0.7)
        axes[1, 1].set_title('True Categories (t-SNE)')
        axes[1, 1].set_xlabel('t-SNE Component 1')
        axes[1, 1].set_ylabel('t-SNE Component 2')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return X_pca, X_tsne


# This is for task-specific clustering as the previous class functions were more focused for task clustering than plagiarism
class TfIDFClustTask(Tfidfclustering):
    def __init__(self, df):
        super().__init__(df)
        
    def task_clustering(self, task_name, n_clusters=None):
        """
        Perform clustering within a specific task to identify plagiarism patterns
        """
        # Filter data for specific task
        task_mask = self.df['Task'] == task_name
        task_df = self.df[task_mask].copy()
    
        if len(task_df) < 4:
            return None, None, f"Not enough documents for task {task_name} (need at least 4, have {len(task_df)})"

        print(f"\n=== TASK {task_name.upper()} CLUSTERING ANALYSIS ===")
        print(f"Documents: {len(task_df)}")
        print(f"Categories: {task_df['Category'].value_counts().to_dict()}")

        # Create TF-IDF matrix for this task only
        task_vectorizer = TfidfVectorizer(
            max_features=500,  
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  
            max_df=0.8
        )
        feature_matrix = task_vectorizer.fit_transform(task_df['clean_text']).toarray()
        feature_names = task_vectorizer.get_feature_names_out()
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            max_clusters = min(len(task_df) // 2, 6)
            cluster_range = range(2, max_clusters + 1) if max_clusters >= 2 else [2]
            
            best_score = -1
            best_k = 2
            
            for k in cluster_range:
                if k >= len(task_df):
                    continue
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(feature_matrix)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(feature_matrix, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            
            n_clusters = best_k
            print(f"Optimal clusters: {n_clusters} (silhouette score: {best_score:.3f})")
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        task_df['task_cluster'] = cluster_labels
        
        # Add similarity-to-original analysis if original document exists
        orig_docs = task_df[task_df['Category'] == 'orig']
        if len(orig_docs) > 0:
            original_text = orig_docs['clean_text'].iloc[0]
            vectorizer_sim = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            all_texts = [original_text] + task_df['clean_text'].tolist()
            tfidf_matrix_sim = vectorizer_sim.fit_transform(all_texts)
            similarity_to_orig = cosine_similarity(tfidf_matrix_sim[0:1], tfidf_matrix_sim[1:])[0]
            task_df['similarity_to_original'] = similarity_to_orig
            
            print("\nSIMILARITY TO ORIGINAL BY CATEGORY:")
            for category in ['orig', 'non', 'cut', 'light', 'heavy']:
                cat_data = task_df[task_df['Category'] == category]
                if len(cat_data) > 0:
                    avg_sim = cat_data['similarity_to_original'].mean()
                    std_sim = cat_data['similarity_to_original'].std()
                    print(f"  {category:>6}: {avg_sim:.3f} ± {std_sim:.3f}")
        
        # Analyze clusters for plagiarism patterns
        print("\nCluster Analysis:")
        plagiarism_clusters = []
        
        for cluster_id in sorted(set(cluster_labels)):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = task_df[cluster_mask]
            
            category_counts = cluster_data['Category'].value_counts()
            plagiarized_count = cluster_data[cluster_data['Category'].isin(['cut', 'light', 'heavy'])].shape[0]
            total_count = len(cluster_data)
            plagiarism_ratio = plagiarized_count / total_count if total_count > 0 else 0
            
            print(f"  Cluster {cluster_id} (n={total_count}):")
            print(f"    Plagiarism ratio: {plagiarism_ratio:.1%}")
            print(f"    Categories: {category_counts.to_dict()}")
            
            # Add similarity analysis if available
            if 'similarity_to_original' in task_df.columns:
                avg_similarity = cluster_data['similarity_to_original'].mean()
                print(f"    Avg similarity to original: {avg_similarity:.3f}")
                
                # Enhanced cluster characterization
                if avg_similarity > 0.8:
                    cluster_type = "HIGH SIMILARITY (potential cut plagiarism)"
                elif avg_similarity > 0.5:
                    cluster_type = "MEDIUM SIMILARITY (potential light plagiarism)"
                elif avg_similarity > 0.2:
                    cluster_type = "LOW SIMILARITY (potential heavy plagiarism or original work)"
                else:
                    cluster_type = "VERY LOW SIMILARITY (likely original work)"
                print(f"    → {cluster_type}")
            
            if plagiarism_ratio > 0.6:
                plagiarism_clusters.append(cluster_id)
                print(f"    → POTENTIAL PLAGIARISM CLUSTER")
            
            sample_files = cluster_data['File'].head(3).tolist()
            print(f"    Sample files: {sample_files}")
            print()
        
        # Calculate metrics
        task_df['is_plagiarized'] = task_df['Category'].isin(['cut', 'light', 'heavy']).astype(int)
        task_df['predicted_plagiarized'] = task_df['task_cluster'].isin(plagiarism_clusters).astype(int)
        
        if task_df['is_plagiarized'].sum() > 0:
            precision = precision_score(task_df['is_plagiarized'], task_df['predicted_plagiarized'])
            recall = recall_score(task_df['is_plagiarized'], task_df['predicted_plagiarized'])
            f1 = f1_score(task_df['is_plagiarized'], task_df['predicted_plagiarized'])
            accuracy = accuracy_score(task_df['is_plagiarized'], task_df['predicted_plagiarized'])
            
            print("PLAGIARISM DETECTION METRICS:")
            print(f"  Accuracy:  {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1-Score:  {f1:.3f}")
        else:
            precision = recall = f1 = accuracy = 0
            print("No plagiarized documents in this task for evaluation.")
        
        return task_df, {
            'task': task_name,
            'n_docs': len(task_df),
            'n_clusters': n_clusters,
            'plagiarism_clusters': plagiarism_clusters,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def analyze_all_tasks(self):
        """Run task-specific clustering for all tasks"""
        print("="*80)
        print("TASK-SPECIFIC CLUSTERING FOR PLAGIARISM DETECTION")
        print("="*80)
        
        all_task_results = []
        all_task_dataframes = []
        
        for task in sorted(self.df['Task'].unique()):
            task_df, metrics = self.task_clustering(task)
            
            if task_df is not None and metrics is not None:
                all_task_results.append(metrics)
                all_task_dataframes.append(task_df)
        
        if all_task_results:
            results_summary = pd.DataFrame(all_task_results)
            print("\n" + "="*80)
            print("TASK-SPECIFIC PLAGIARISM DETECTION SUMMARY")
            print("="*80)
            print(results_summary[['task', 'n_docs', 'n_clusters', 'accuracy', 'precision', 'recall', 'f1']].round(3))
            
            # Overall performance
            avg_metrics = results_summary[['accuracy', 'precision', 'recall', 'f1']].mean()
            print(f"\nAVERAGE PERFORMANCE ACROSS ALL TASKS:")
            print(f"  Accuracy:  {avg_metrics['accuracy']:.3f}")
            print(f"  Precision: {avg_metrics['precision']:.3f}")
            print(f"  Recall:    {avg_metrics['recall']:.3f}")
            print(f"  F1-Score:  {avg_metrics['f1']:.3f}")
            
            return all_task_dataframes, results_summary
        else:
            print("No successful task-specific clustering results.")
            return [], None
    
    def comprehensive_plagiarism_detection(self, task_name, similarity_threshold=0.7):
        """
        Comprehensive plagiarism detection using similarity-based classification
        """
        task_mask = self.df['Task'] == task_name
        task_df = self.df[task_mask].copy()
        
        if len(task_df) < 4:
            return None
        
        print(f"\n=== COMPREHENSIVE PLAGIARISM DETECTION: TASK {task_name.upper()} ===")
        
        # Find original document and calculate similarities
        orig_docs = task_df[task_df['Category'] == 'orig']
        if len(orig_docs) == 0:
            print(f"No original document found for task {task_name}")
            return None
        
        original_text = orig_docs['clean_text'].iloc[0]
        vectorizer_sim = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        all_texts = [original_text] + task_df['clean_text'].tolist()
        tfidf_matrix = vectorizer_sim.fit_transform(all_texts)
        similarity_to_orig = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        task_df['similarity_to_original'] = similarity_to_orig
        
        # Rule-based classification
        def classify_document(row):
            similarity = row['similarity_to_original']
            category = row['Category']
            
            if category == 'orig':
                return 'original', 'Original document'
            
            if similarity > 0.8:
                return 'high_risk', f'Very high similarity ({similarity:.3f}) - likely CUT plagiarism'
            elif similarity > 0.6:
                return 'medium_risk', f'High similarity ({similarity:.3f}) - likely LIGHT plagiarism'
            elif similarity > 0.4:
                return 'low_risk', f'Medium similarity ({similarity:.3f}) - possible HEAVY plagiarism'
            else:
                return 'likely_original', f'Low similarity ({similarity:.3f}) - likely original work'
        
        # Apply classification
        classifications = task_df.apply(classify_document, axis=1)
        task_df['risk_level'] = [c[0] for c in classifications]
        task_df['risk_reason'] = [c[1] for c in classifications]
        
        # Evaluate against ground truth
        detection_results = []
        for _, row in task_df.iterrows():
            true_category = row['Category']
            predicted_risk = row['risk_level']
            
            is_actually_plagiarized = true_category in ['cut', 'light', 'heavy']
            is_predicted_plagiarized = predicted_risk in ['high_risk', 'medium_risk', 'low_risk']
            
            correct = (is_actually_plagiarized and is_predicted_plagiarized) or \
                     (not is_actually_plagiarized and not is_predicted_plagiarized)
            
            detection_results.append({
                'file': row['File'],
                'true_category': true_category,
                'predicted_risk': predicted_risk,
                'similarity': row['similarity_to_original'],
                'correct': correct,
                'reason': row['risk_reason']
            })
        
        # Calculate performance
        correct_predictions = sum(r['correct'] for r in detection_results)
        total_predictions = len(detection_results)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'task': task_name,
            'accuracy': accuracy,
            'detection_results': detection_results,
            'task_df': task_df
        }
    
    def analyze_all_tasks_comprehensive(self):
        """
        Run comprehensive similarity-based detection for all tasks
        """
        print("="*80)
        print("COMPREHENSIVE PLAGIARISM DETECTION RESULTS")
        print("="*80)
        
        final_results = {}
        for task in sorted(self.df['Task'].unique()):
            result = self.comprehensive_plagiarism_detection(task)
            if result:
                final_results[task] = result
        
        if final_results:
            print("\n" + "="*80)
            print("OVERALL PERFORMANCE SUMMARY")
            print("="*80)
            
            total_accuracy = np.mean([r['accuracy'] for r in final_results.values()])
            print(f"Average Accuracy Across All Tasks: {total_accuracy:.1%}")
            
            print("\nPer-Task Performance:")
            for task, result in final_results.items():
                print(f"  Task {task.upper()}: {result['accuracy']:.1%}")
        
        return final_results


