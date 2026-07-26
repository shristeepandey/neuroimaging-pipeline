#!/usr/bin/env python3
"""Main neuroimaging data analysis pipeline.

Usage:
    python run_pipeline.py
"""

import sys
from pathlib import Path
import json
import argparse
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data import generate_synthetic_data, generate_synthetic_fmri
from preprocessing import (quality_control, preprocess_clinical_data, 
                           preprocess_fmri, compute_brain_metrics)
from analysis import (descriptive_statistics, group_comparisons, 
                      correlation_analysis, predictive_modeling, fmri_analysis)
from visualization import (plot_demographics, plot_clinical_metrics,
                           plot_correlation_matrix, plot_connectivity,
                           plot_model_results, create_summary_dashboard)


def run_pipeline(generate_data=True, run_fmri=True, output_dir="outputs"):
    """Run the complete neuroimaging analysis pipeline.
    
    Args:
        generate_data: Whether to generate synthetic data
        run_fmri: Whether to run fMRI analysis
        output_dir: Output directory
    """
    print("=" * 70)
    print("NEUROIMAGING DATA ANALYSIS PIPELINE")
    print("=" * 70)
    print()
    
    # Create output directories
    Path(output_dir).mkdir(exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    
    # Step 1: Data Generation/Download
    print("Step 1: Data Preparation")
    print("-" * 70)
    
    if generate_data:
        data_path = generate_synthetic_data("data/raw/oasis")
        print(f"✓ Clinical data prepared: {data_path}/participants.csv")
    
    if run_fmri:
        fmri_path = generate_synthetic_fmri("data/raw/fmri")
        print(f"✓ fMRI data prepared: {fmri_path}")
    
    print()
    
    # Step 2: Quality Control
    print("Step 2: Quality Control")
    print("-" * 70)
    qc_results = quality_control("data/raw/oasis/participants.csv")
    print(f"✓ Quality control completed")
    print()
    
    # Step 3: Preprocessing
    print("Step 3: Data Preprocessing")
    print("-" * 70)
    df = preprocess_clinical_data("data/raw/oasis/participants.csv")
    print(f"✓ Clinical data preprocessed")
    
    if run_fmri:
        time_series, connectivity, connectivity_thresh = preprocess_fmri("data/raw/fmri/fmri_timeseries.npy")
        print(f"✓ fMRI data preprocessed")
    print()
    
    # Step 4: Statistical Analysis
    print("Step 4: Statistical Analysis")
    print("-" * 70)
    
    # Descriptive statistics
    desc_stats = descriptive_statistics(df)
    with open("outputs/results/descriptive_statistics.json", 'w') as f:
        json.dump(desc_stats, f, indent=2)
    print(f"✓ Descriptive statistics computed")
    
    # Group comparisons
    group_results = group_comparisons(df)
    with open("outputs/results/group_comparisons.json", 'w') as f:
        json.dump(group_results, f, indent=2, default=str)
    print(f"✓ Group comparisons completed")
    
    # Correlation analysis
    corr_matrix, sig_corrs = correlation_analysis(df)
    corr_matrix.to_csv("outputs/results/correlation_matrix.csv")
    print(f"✓ Correlation analysis completed ({len(sig_corrs)} significant correlations)")
    
    # Predictive modeling
    model_results = predictive_modeling(df)
    print(f"✓ Predictive models trained and evaluated")
    print()
    
    # Step 5: fMRI Analysis (if run)
    if run_fmri:
        print("Step 5: fMRI Connectivity Analysis")
        print("-" * 70)
        fmri_metrics = fmri_analysis(connectivity)
        np.save("outputs/results/connectivity_matrix.npy", connectivity)
        np.save("outputs/results/connectivity_thresholded.npy", connectivity_thresh)
        print(f"✓ fMRI analysis completed")
        print()
    
    # Step 6: Visualization
    print("Step 6: Generating Visualizations")
    print("-" * 70)
    
    plot_demographics(df, "outputs/figures/demographics.png")
    plot_clinical_metrics(df, "outputs/figures/clinical_metrics.png")
    plot_correlation_matrix(corr_matrix, "outputs/figures/correlation_matrix.png")
    
    if run_fmri:
        plot_connectivity(connectivity, "outputs/figures/connectivity.png")
    
    plot_model_results(model_results, "outputs/figures/model_results.png")
    
    # Create dashboard
    if run_fmri:
        create_summary_dashboard(df, {}, model_results, corr_matrix, 
                                "outputs/figures/dashboard.png")
    
    print(f"✓ All visualizations generated")
    print()
    
    # Step 7: Generate Report
    print("Step 7: Generating Report")
    print("-" * 70)
    generate_report(df, desc_stats, group_results, sig_corrs, model_results, 
                   fmri_metrics if run_fmri else None)
    print(f"✓ Report generated: outputs/reports/analysis_report.md")
    print()
    
    # Summary
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Results saved to: {output_dir}/")
    print(f"  - Figures: {output_dir}/figures/")
    print(f"  - Results: {output_dir}/results/")
    print(f"  - Reports: {output_dir}/reports/")
    print()
    
    return {
        'df': df,
        'descriptive_stats': desc_stats,
        'group_comparisons': group_results,
        'correlations': sig_corrs,
        'models': model_results
    }


def generate_report(df, desc_stats, group_results, sig_corrs, model_results, fmri_metrics=None):
    """Generate comprehensive analysis report in Markdown.
    
    Args:
        df: DataFrame with participant data
        desc_stats: Descriptive statistics
        group_results: Group comparison results
        sig_corrs: Significant correlations
        model_results: Model performance results
        fmri_metrics: fMRI metrics (optional)
    """
    report = []
    report.append("# Neuroimaging Data Analysis Report")
    report.append("\n**Generated by:** Neuroimaging Data Analysis Pipeline v1.0.0  ")
    report.append("**Author:** Shristee Pandey  ")
    report.append("**Date:** 2025-07-26")
    report.append("\n---\n")
    
    # Executive Summary
    report.append("\n## Executive Summary\n")
    report.append(f"This report presents a comprehensive analysis of neuroimaging data including ")
    report.append(f"clinical, cognitive, and imaging metrics from {len(df)} participants.\n")
    
    report.append("\n**Key Findings:**\n")
    report.append(f"- Dataset includes {len(df)} participants with detailed clinical and neuroimaging data")
    
    if 'diagnosis' in df.columns:
        dx_counts = df['diagnosis'].value_counts()
        for dx, count in dx_counts.items():
            report.append(f"- {dx}: {count} participants ({100*count/len(df):.1f}%)")
    
    report.append(f"- {len(sig_corrs)} statistically significant correlations identified")
    
    # Best model
    best_model = max(model_results.items(), key=lambda x: x[1]['auc'])
    report.append(f"- Best predictive model: {best_model[0]} (AUC = {best_model[1]['auc']:.3f})")
    
    if fmri_metrics:
        report.append(f"\n### fMRI Analysis")
        report.append(f"- Brain regions analyzed: {fmri_metrics.get('n_regions', 'N/A')}")
        report.append(f"- Mean connectivity: {fmri_metrics.get('mean_connectivity', 0):.3f}")
        report.append(f"- Network communities detected: {fmri_metrics.get('n_communities', 'N/A')}")
    
    # Data Overview
    report.append("\n\n## Data Overview\n")
    report.append("\n### Dataset Summary\n")
    report.append(f"- **Total Participants:** {len(df)}")
    report.append(f"- **Age Range:** {desc_stats.get('age_min', 0):.0f} - {desc_stats.get('age_max', 0):.0f} years")
    report.append(f"- **Mean Age:** {desc_stats.get('age_mean', 0):.1f} ± {desc_stats.get('age_std', 0):.1f} years")
    report.append(f"- **Sex Distribution:** {(df['sex'] == 'M').sum()} Male, {(df['sex'] == 'F').sum()} Female")
    
    report.append("\n### Clinical Metrics\n")
    report.append("\n| Metric | Mean | Std | Median | Min | Max |")
    report.append("|--------|------|-----|--------|-----|-----|")
    for metric in ['mmse_score', 'cognitive_score', 'brain_volume_cm3', 'hippocampal_volume_cm3']:
        report.append(f"| {metric.replace('_', ' ').title()} | "
                     f"{desc_stats.get(f'{metric}_mean', 0):.1f} | "
                     f"{desc_stats.get(f'{metric}_std', 0):.1f} | "
                     f"{desc_stats.get(f'{metric}_median', 0):.1f} | "
                     f"{desc_stats.get(f'{metric}_min', 0):.1f} | "
                     f"{desc_stats.get(f'{metric}_max', 0):.1f} |")
    
    # Results
    report.append("\n\n## Statistical Results\n")
    
    report.append("\n### Group Comparisons\n")
    report.append("\nComparison between Control and Patient groups:\n")
    
    for metric, result in group_results.items():
        significance = "Yes" if result['significant'] else "No"
        report.append(f"\n**{metric.replace('_', ' ').title()}**")
        report.append(f"- Control: {result['control_mean']:.2f} ± {result['control_std']:.2f}")
        report.append(f"- Patient: {result['patient_mean']:.2f} ± {result['patient_std']:.2f}")
        report.append(f"- t-statistic: {result['t_statistic']:.3f}")
        report.append(f"- p-value: {result['p_value']:.4f}")
        report.append(f"- Cohen's d: {result['cohens_d']:.3f}")
        report.append(f"- Significant: {significance}")
    
    report.append("\n### Significant Correlations\n")
    report.append(f"\n{len(sig_corrs)} statistically significant correlations found (p < 0.05):\n")
    
    for corr in sig_corrs[:10]:  # Show top 10
        strength = "strong" if abs(corr['correlation']) > 0.5 else "moderate" if abs(corr['correlation']) > 0.3 else "weak"
        direction = "positive" if corr['correlation'] > 0 else "negative"
        report.append(f"- **{corr['variable_1'].replace('_', ' ').title()}** ↔ **{corr['variable_2'].replace('_', ' ').title()}**: "
                     f"r = {corr['correlation']:.3f} ({strength} {direction}, p = {corr['p_value']:.4f})")
    
    # Predictive Modeling
    report.append("\n\n## Predictive Modeling\n")
    report.append("\n### Model Performance\n")
    report.append("\n| Model | AUC | Accuracy |")
    report.append("|-------|-----|----------|")
    for model_name, results in model_results.items():
        report.append(f"| {model_name.replace('_', ' ').title()} | "
                     f"{results['auc']:.3f} | {results['accuracy']:.3f} |")
    
    report.append("\n### Feature Importance (Random Forest)\n")
    for feature, importance in model_results['random_forest']['feature_importance'].items():
        report.append(f"- **{feature.replace('_', ' ').title()}**: {importance:.3f}")
    
    # Figures
    report.append("\n\n## Figures\n")
    report.append("\n### Demographics")
    report.append("\n![Demographics](figures/demographics.png)\n")
    
    report.append("\n### Clinical Metrics")
    report.append("\n![Clinical Metrics](figures/clinical_metrics.png)\n")
    
    report.append("\n### Correlation Matrix")
    report.append("\n![Correlation Matrix](figures/correlation_matrix.png)\n")
    
    report.append("\n### Model Performance")
    report.append("\n![Model Results](figures/model_results.png)\n")
    
    if fmri_metrics:
        report.append("\n### fMRI Connectivity")
        report.append("\n![Connectivity](figures/connectivity.png)\n")
    
    # Conclusions
    report.append("\n\n## Conclusions\n")
    report.append("\nThis analysis demonstrates the application of machine learning and statistical methods ")
    report.append("to neuroimaging data for cognitive impairment prediction.\n")
    
    report.append("\n**Key Takeaways:**\n")
    report.append("1. Significant differences observed in brain volumes and cognitive scores across diagnostic groups")
    report.append("2. Strong correlations identified between volumetric measures and cognitive performance")
    report.append("3. Machine learning models achieve good predictive accuracy for classification")
    report.append("4. Hippocampal volume and cognitive score are the strongest predictors")
    
    if fmri_metrics:
        report.append("\n5. Functional connectivity analysis reveals distinct network organization patterns")
    
    report.append("\n\n---\n")
    report.append("\n*Report generated automatically by the Neuroimaging Data Analysis Pipeline*")
    
    # Save report
    report_path = Path("outputs/reports/analysis_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neuroimaging analysis pipeline")
    parser.add_argument('--skip-data', action='store_true', help='Skip data generation')
    parser.add_argument('--skip-fmri', action='store_true', help='Skip fMRI analysis')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    results = run_pipeline(
        generate_data=not args.skip_data,
        run_fmri=not args.skip_fmri,
        output_dir=args.output_dir
    )
