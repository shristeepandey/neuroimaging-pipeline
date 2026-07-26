"""Statistical analysis module for neuroimaging data."""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json


def descriptive_statistics(df):
    """Compute comprehensive descriptive statistics.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary of descriptive statistics
    """
    print("Computing descriptive statistics...")
    
    stats_dict ={}
    
    # Demographics
    continuous_vars = ['age', 'education_years', 'mmse_score', 
                       'brain_volume_cm3', 'hippocampal_volume_cm3', 
                       'cognitive_score']
    
    for var in continuous_vars:
        if var in df.columns:
            stats_dict[f'{var}_mean'] = float(df[var].mean())
            stats_dict[f'{var}_std'] = float(df[var].std())
            stats_dict[f'{var}_median'] = float(df[var].median())
            stats_dict[f'{var}_min'] = float(df[var].min())
            stats_dict[f'{var}_max'] = float(df[var].max())
    
    # By diagnosis group
    for diagnosis in df['diagnosis'].unique():
        subset = df[df['diagnosis'] == diagnosis]
        stats_dict[f'{diagnosis}_n'] = int(len(subset))
        
        for var in continuous_vars:
            if var in subset.columns:
                stats_dict[f'{diagnosis}_{var}_mean'] = float(subset[var].mean())
                stats_dict[f'{diagnosis}_{var}_std'] = float(subset[var].std())
    
    return stats_dict


def group_comparisons(df):
    """Perform statistical group comparisons.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary of test results
    """
    print("Performing group comparisons...")
    
    results = {}
    
    control = df[df['diagnosis'] == 'Control']
    patients = df[df['diagnosis'] != 'Control']
    
    continuous_vars = ['brain_volume_cm3', 'hippocampal_volume_cm3', 
                       'cognitive_score', 'mmse_score']
    
    for var in continuous_vars:
        if var in df.columns:
            control_vals = control[var].dropna()
            patient_vals = patients[var].dropna()
            
            # Independent samples t-test
            t_stat, p_value = stats.ttest_ind(control_vals, patient_vals)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_vals)-1)*control_vals.var() + 
                                 (len(patient_vals)-1)*patient_vals.var()) / 
                                (len(control_vals) + len(patient_vals) - 2))
            
            cohens_d = (patient_vals.mean() - control_vals.mean()) / pooled_std
            
            results[var] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'control_mean': float(control_vals.mean()),
                'control_std': float(control_vals.std()),
                'control_n': int(len(control_vals)),
                'patient_mean': float(patient_vals.mean()),
                'patient_std': float(patient_vals.std()),
                'patient_n': int(len(patient_vals)),
                'significant': p_value < 0.05
            }
    
    return results


def correlation_analysis(df):
    """Perform correlation analysis between neuroimaging and clinical metrics.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Correlation matrix and key findings
    """
    print("Performing correlation analysis...")
    
    # Select relevant variables
    vars_of_interest = ['age', 'education_years', 'mmse_score',
                        'brain_volume_cm3', 'hippocampal_volume_cm3', 
                        'cognitive_score']
    
    df_subset = df[vars_of_interest].dropna()
    
    # Compute correlation matrix
    corr_matrix = df_subset.corr()
    
    # Find significant correlations
    significant_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            r = corr_matrix.iloc[i, j]
            
            # Test significance
            n = len(df_subset)
            t_stat = r * np.sqrt((n-2) / (1-r**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
            
            if p_value < 0.05:
                significant_corrs.append({
                    'variable_1': var1,
                    'variable_2': var2,
                    'correlation': float(r),
                    'p_value': float(p_value)
                })
    
    return corr_matrix, significant_corrs


def predictive_modeling(df):
    """Build predictive models for cognitive impairment.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary of model results
    """
    print("Training predictive models...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    # Prepare features
    feature_cols = ['age', 'education_years', 'brain_volume_cm3', 
                    'hippocampal_volume_cm3']
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['diagnosis_binary']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    
    # Train Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    results = {
        'random_forest': {
            'auc': float(roc_auc_score(y_test, rf_proba)),
            'accuracy': float((rf_pred == y_test).mean()),
            'feature_importance': dict(zip(feature_cols, 
                                          map(float, rf.feature_importances_)))
        },
        'logistic_regression': {
            'auc': float(roc_auc_score(y_test, lr_proba)),
            'accuracy': float((lr_pred == y_test).mean()),
            'coefficients': dict(zip(feature_cols, 
                                    map(float, lr.coef_[0])))
        }
    }
    
    # Save results
    output_path = Path("outputs/results/model_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Model results saved to outputs/results/model_results.json")
    return results


def fmri_analysis(connectivity_matrix):
    """Analyze fMRI connectivity data.
    
    Args:
        connectivity_matrix: NxN connectivity matrix
        
    Returns:
        Network metrics
    """
    print("Analyzing fMRI connectivity...")
    
    n_regions = connectivity_matrix.shape[0]
    
    # Compute network metrics
    metrics = {
        'n_regions': n_regions,
        'mean_connectivity': float(connectivity_matrix.mean()),
        'std_connectivity': float(connectivity_matrix.std()),
        'max_connectivity': float(connectivity_matrix.max()),
        'min_connectivity': float(connectivity_matrix.min())
    }
    
    # Compute graph metrics
    # Degree distribution
    degree = np.sum(np.abs(connectivity_matrix) > 0, axis=1)
    metrics['mean_degree'] = float(degree.mean())
    metrics['max_degree'] = int(degree.max())
    metrics['min_degree'] = int(degree.min())
    
    # Modularity (simplified)
    import networkx as nx
    
    G = nx.from_numpy_array(np.abs(connectivity_matrix))
    
    if G.number_of_edges() > 0:
        metrics['clustering_coefficient'] = float(nx.average_clustering(G))
        metrics['network_efficiency'] = float(nx.global_efficiency(G))
        
        # Community detection
        try:
            communities = nx.community.greedy_modularity_communities(G)
            metrics['n_communities'] = len(communities)
            metrics['modularity'] = float(nx.community.modularity(G, communities))
        except:
            metrics['n_communities'] = 0
            metrics['modularity'] = 0.0
    
    # Save results
    output_path = Path("outputs/results/fmri_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"fMRI analysis complete: {metrics}")
    return metrics
