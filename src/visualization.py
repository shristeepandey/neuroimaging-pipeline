"""Visualization module for neuroimaging data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def set_plotting_style():
    """Set consistent plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def plot_demographics(df, output_path="outputs/figures/demographics.png"):
    """Plot demographic distributions.
    
    Args:
        df: DataFrame with participant data
        output_path: Path to save figure
    """
    set_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Participant Demographics', fontsize=14, fontweight='bold')
    
    # Age distribution
    axes[0, 0].hist(df['age'], bins=20, color='#6366f1', edgecolor='white', alpha=0.8)
    axes[0, 0].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].axvline(df['age'].mean(), color='red', linestyle='--', label=f'Mean: {df["age"].mean():.1f}')
    axes[0, 0].legend()
    
    # Sex distribution
    sex_counts = df['sex'].value_counts()
    axes[0, 1].bar(sex_counts.index, sex_counts.values, color=['#6366f1', '#ec4899'])
    axes[0, 1].set_xlabel('Sex')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Sex Distribution')
    
    # Diagnosis distribution
    dx_counts = df['diagnosis'].value_counts()
    axes[1, 0].barh(dx_counts.index, dx_counts.values, color=['#10b981', '#f59e0b', '#ef4444'])
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_title('Diagnosis Distribution')
    
    # Education distribution
    axes[1, 1].hist(df['education_years'], bins=15, color='#8b5cf6', edgecolor='white', alpha=0.8)
    axes[1, 1].set_xlabel('Education (years)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Education Distribution')
    axes[1, 1].axvline(df['education_years'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["education_years"].mean():.1f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Demographics plot saved to {output_path}")
    plt.close()


def plot_clinical_metrics(df, output_path="outputs/figures/clinical_metrics.png"):
    """Plot clinical and cognitive metrics by diagnosis.
    
    Args:
        df: DataFrame with participant data
        output_path: Path to save figure
    """
    set_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Clinical Metrics by Diagnosis', fontsize=14, fontweight='bold')
    
    metrics = ['mmse_score', 'cognitive_score', 'brain_volume_cm3', 'hippocampal_volume_cm3']
    titles = ['MMSE Score', 'Cognitive Score', 'Brain Volume', 'Hippocampal Volume']
    ylabels = ['Score', 'Score', 'cm³', 'cm³']
    
    for ax, metric, title, ylabel in zip(axes.flatten(), metrics, titles, ylabels):
        df.boxplot(column=metric, by='diagnosis', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Diagnosis')
        ax.set_ylabel(ylabel)
        plt.sca(ax)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Clinical metrics plot saved to {output_path}")
    plt.close()


def plot_correlation_matrix(corr_matrix, output_path="outputs/figures/correlation_matrix.png"):
    """Plot correlation matrix heatmap.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        output_path: Path to save figure
    """
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Neuroimaging & Clinical Metrics Correlation Matrix', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Correlation matrix saved to {output_path}")
    plt.close()


def plot_connectivity(connectivity_matrix, output_path="outputs/figures/connectivity.png"):
    """Plot fMRI connectivity matrix.
    
    Args:
        connectivity_matrix: NxN connectivity matrix
        output_path: Path to save figure
    """
    set_plotting_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Full connectivity
    im1 = axes[0].imshow(connectivity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Functional Connectivity Matrix')
    axes[0].set_xlabel('Brain Region')
    axes[0].set_ylabel('Brain Region')
    plt.colorbar(im1, ax=axes[0])
    
    # Thresholded connectivity
    threshold = np.percentile(np.abs(connectivity_matrix), 75)
    connectivity_thresh = np.where(np.abs(connectivity_matrix) >= threshold, 
                                   connectivity_matrix, 0)
    
    im2 = axes[1].imshow(connectivity_thresh, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title(f'Thresholded Connectivity (75th percentile)')
    axes[1].set_xlabel('Brain Region')
    axes[1].set_ylabel('Brain Region')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Connectivity plot saved to {output_path}")
    plt.close()


def plot_model_results(results, output_path="outputs/figures/model_results.png"):
    """Plot model performance comparison.
    
    Args:
        results: Dictionary with model results
        output_path: Path to save figure
    """
    set_plotting_style()
    
    models = ['Random Forest', 'Logistic Regression']
    auc_scores = [results['random_forest']['auc'], results['logistic_regression']['auc']]
    accuracies = [results['random_forest']['accuracy'], results['logistic_regression']['accuracy']]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, auc_scores, width, label='AUC', color='#6366f1')
    bars2 = ax.bar(x + width/2, accuracies, width, label='Accuracy', color='#ec4899')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Model results plot saved to {output_path}")
    plt.close()


def create_summary_dashboard(df, metrics, results, corr_matrix, output_path="outputs/figures/dashboard.png"):
    """Create comprehensive visualization dashboard.
    
    Args:
        df: DataFrame with participant data
        metrics: Brain metrics dictionary
        results: Model results dictionary
        corr_matrix: Correlation matrix
        output_path: Path to save figure
    """
    set_plotting_style()
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Diagnosis distribution
    ax1 = fig.add_subplot(gs[0, 0])
    dx_counts = df['diagnosis'].value_counts()
    ax1.barh(dx_counts.index, dx_counts.values, color=['#10b981', '#f59e0b', '#ef4444'])
    ax1.set_title('Diagnosis Distribution')
    ax1.set_xlabel('Count')
    
    # 2. Age vs Cognitive Score
    ax2 = fig.add_subplot(gs[0, 1])
    for diagnosis in df['diagnosis'].unique():
        subset = df[df['diagnosis'] == diagnosis]
        ax2.scatter(subset['age'], subset['cognitive_score'], 
                   label=diagnosis, alpha=0.6, s=50)
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Cognitive Score')
    ax2.set_title('Age vs Cognitive Score')
    ax2.legend()
    
    # 3. Brain Volume Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    df.boxplot(column='brain_volume_cm3', by='diagnosis', ax=ax3)
    ax3.set_title('Brain Volume by Diagnosis')
    ax3.set_xlabel('Diagnosis')
    ax3.set_ylabel('Brain Volume (cm³)')
    plt.sca(ax3)
    plt.xticks(rotation=45)
    
    # 4. Correlation heatmap (simplified)
    ax4 = fig.add_subplot(gs[1, :2])
    vars_subset = ['mmse_score', 'cognitive_score', 'brain_volume_cm3', 'hippocampal_volume_cm3']
    corr_subset = corr_matrix.loc[vars_subset, vars_subset]
    sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, ax=ax4, cbar_kws={"shrink": 0.8})
    ax4.set_title('Correlation Matrix')
    
    # 5. Model Performance
    ax5 = fig.add_subplot(gs[1, 2])
    models = ['RF', 'LR']
    aucs = [results['random_forest']['auc'], results['logistic_regression']['auc']]
    ax5.bar(models, aucs, color=['#6366f1', '#ec4899'])
    ax5.set_ylabel('AUC')
    ax5.set_title('Model AUC Scores')
    ax5.set_ylim(0, 1.1)
    for i, v in enumerate(aucs):
        ax5.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # 6. Feature Importance
    ax6 = fig.add_subplot(gs[2, :])
    features = list(results['random_forest']['feature_importance'].keys())
    importance = list(results['random_forest']['feature_importance'].values())
    ax6.barh(features, importance, color='#8b5cf6')
    ax6.set_xlabel('Importance')
    ax6.set_title('Random Forest Feature Importance')
    
    plt.suptitle('Neuroimaging Pipeline Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Dashboard saved to {output_path}")
    plt.close()
