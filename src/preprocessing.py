"""Quality control and preprocessing module for neuroimaging data."""

import numpy as np
import pandas as pd
from pathlib import Path
import json


def quality_control(data_path="data/raw/oasis/participants.csv"):
    """Perform quality control checks on neuroimaging data.
    
    Args:
        data_path: Path to participants CSV
        
    Returns:
        DataFrame with QC results
    """
    df = pd.read_csv(data_path)
    
    qc_results = []
    
    for _, row in df.iterrows():
        subject_id = row['subject_id']
        
        # Check 1: Scan quality
        scan_quality = row.get('scan_quality', 'Unknown')
        quality_flag = scan_quality in ['High', 'Medium']
        
        # Check 2: Missing data
        missing_data = row[['age', 'sex', 'diagnosis', 'mmse_score']].isnull().any()
        
        # Check 3: MMSE score validity
        mmse = row.get('mmse_score', np.nan)
        mmse_valid = 0 <= mmse <= 30 if not pd.isna(mmse) else False
        
        # Check 4: Age range
        age = row.get('age', 0)
        age_valid = 18 <= age <= 100
        
        # Overall QC status
        passed = quality_flag and not missing_data and mmse_valid and age_valid
        
        qc_results.append({
            'subject_id': subject_id,
            'scan_quality': scan_quality,
            'missing_data': missing_data,
            'mmse_valid': mmse_valid,
            'age_valid': age_valid,
            'passed_qc': passed
        })
    
    qc_df = pd.DataFrame(qc_results)
    
    # Summary
    passed = qc_df['passed_qc'].sum()
    total = len(qc_df)
    
    print(f"Quality Control Summary:")
    print(f"  Total subjects: {total}")
    print(f"  Passed QC: {passed} ({100*passed/total:.1f}%)")
    print(f"  Failed QC: {total-passed} ({100*(total-passed)/total:.1f}%)")
    
    return qc_df


def preprocess_clinical_data(data_path="data/raw/oasis/participants.csv"):
    """Preprocess clinical and demographic data.
    
    Args:
        data_path: Path to raw data
        
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(data_path)
    
    print("Preprocessing clinical data...")
    
    # Handle missing values
    df['mmse_score'] = df['mmse_score'].fillna(df['mmse_score'].median())
    
    # Convert categorical variables
    df['sex_encoded'] = (df['sex'] == 'M').astype(int)
    
    # Normalize continuous variables
    continuous_vars = ['age', 'education_years', 'mmse_score', 
                       'brain_volume_cm3', 'hippocampal_volume_cm3', 
                       'cognitive_score']
    
    for var in continuous_vars:
        if var in df.columns:
            df[f'{var}_zscore'] = (df[var] - df[var].mean()) / df[var].std()
    
    # Create diagnostic groups
    df['diagnosis_binary'] = (df['diagnosis'] != 'Control').astype(int)
    
    print(f"Preprocessed {len(df)} subjects")
    return df


def preprocess_fmri(time_series_path="data/raw/fmri/fmri_timeseries.npy"):
    """Preprocess fMRI time series data.
    
    Args:
        time_series_path: Path to fMRI time series
        
    Returns:
        Preprocessed time series and connectivity matrix
    """
    print("Preprocessing fMRI data...")
    
    time_series = np.load(time_series_path)
    
    # Step 1: Detrend
    from scipy.signal import detrend
    time_series = detrend(time_series, axis=0)
    
    # Step 2: Standardize
    time_series = (time_series - time_series.mean(axis=0)) / time_series.std(axis=0)
    
    # Step 3: Compute connectivity matrix
    from scipy.stats import pearsonr
    
    n_regions = time_series.shape[1]
    connectivity = np.zeros((n_regions, n_regions))
    
    for i in range(n_regions):
        for j in range(n_regions):
            if i != j:
                r, _ = pearsonr(time_series[:, i], time_series[:, j])
                connectivity[i, j] = r
            else:
                connectivity[i, j] = 1.0
    
    # Step 4: Threshold connectivity matrix
    threshold = np.percentile(np.abs(connectivity[np.triu_indices(n_regions, k=1)]), 75)
    connectivity_thresh = np.where(np.abs(connectivity) >= threshold, connectivity, 0)
    
    print(f"Preprocessed fMRI: {time_series.shape[0]} timepoints, {n_regions} regions")
    print(f"Mean connectivity: {connectivity.mean():.3f}")
    print(f"Threshold (75th percentile): {threshold:.3f}")
    
    return time_series, connectivity, connectivity_thresh


def compute_brain_metrics(df):
    """Compute summary brain metrics.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary of computed metrics
    """
    print("Computing brain metrics...")
    
    metrics = {}
    
    # Volume metrics by diagnosis
    for diagnosis in df['diagnosis'].unique():
        subset = df[df['diagnosis'] == diagnosis]
        metrics[f'{diagnosis}_brain_volume'] = subset['brain_volume_cm3'].mean()
        metrics[f'{diagnosis}_hippocampal_volume'] = subset['hippocampal_volume_cm3'].mean()
        metrics[f'{diagnosis}_cognitive_score'] = subset['cognitive_score'].mean()
    
    # Correlation metrics
    metrics['volume_cognition_corr'] = df['brain_volume_cm3'].corr(df['cognitive_score'])
    metrics['hippo_volume_cognition_corr'] = df['hippocampal_volume_cm3'].corr(df['cognitive_score'])
    metrics['age_volume_corr'] = df['age'].corr(df['brain_volume_cm3'])
    
    # Save metrics
    import json
    output_path = Path("outputs/results/brain_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Brain metrics computed and saved to {output_path}")
    return metrics
