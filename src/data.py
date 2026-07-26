"""Data download module for public neuroimaging datasets."""

import os
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path


def download_oasis_data(output_dir="data/raw/oasis"):
    """Download OASIS brain MRI dataset summary.
    
    Args:
        output_dir: Directory to save downloaded data
        
    Returns:
        Path to downloaded data directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # OASIS-1 dataset summary
    url = "https://www.nitrc.org/frs/?group_id=196&id=1213&filename=oasis_downloads.zip"
    
    print(f"Downloading OASIS dataset to {output_path}...")
    
    # Download dataset
    zip_path = output_path / "oasis_downloads.zip"
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        
        print(f"Dataset downloaded and extracted to {output_path}")
        return output_path
    except Exception as e:
        print(f"Warning: Could not download dataset: {e}")
        print("Using synthetic data for demonstration...")
        return generate_synthetic_data(output_path)


def download_sample_fmri(output_dir="data/raw/fmri"):
    """Download sample fMRI data from nilearn.
    
    Args:
        output_dir: Directory to save data
        
    Returns:
        Path to data directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        from nilearn import datasets
        
        # Download sample fMRI dataset
        print("Fetching sample fMRI dataset...")
        dataset = datasets.fetch_adhd(n_subjects=1, data_dir=str(output_path))
        
        print(f"Sample fMRI data downloaded to {output_path}")
        return output_path
    except ImportError:
        print("nilearn not installed. Install with: pip install nilearn")
        return generate_synthetic_fmri(output_path)


def generate_synthetic_data(output_dir):
    """Generate synthetic brain MRI data for demonstration.
    
    Args:
        output_dir: Directory to save synthetic data
        
    Returns:
        Path to data directory
    """
    import numpy as np
    import pandas as pd
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic neuroimaging data...")
    
    # Generate synthetic clinical data
    n_subjects = 100
    np.random.seed(42)
    
    data = {
        'subject_id': [f'sub-{i:03d}' for i in range(1, n_subjects + 1)],
        'age': np.random.randint(18, 85, n_subjects),
        'sex': np.random.choice(['M', 'F'], n_subjects),
        'diagnosis': np.random.choice(['Control', 'MCI', 'Dementia'], n_subjects, p=[0.4, 0.35, 0.25]),
        'education_years': np.random.randint(8, 20, n_subjects),
        'mmse_score': np.random.randint(15, 30, n_subjects),
        'brain_volume_cm3': np.random.normal(1100, 120, n_subjects),
        'hippocampal_volume_cm3': np.random.normal(7.2, 0.8, n_subjects),
        'cognitive_score': np.random.normal(50, 10, n_subjects),
        'scan_quality': np.random.choice(['High', 'Medium', 'Low'], n_subjects, p=[0.7, 0.25, 0.05])
    }
    
    df = pd.DataFrame(data)
    csv_path = output_path / 'participants.csv'
    df.to_csv(csv_path, index=False)
    
    # Generate summary statistics
    summary_path = output_path / 'dataset_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("NEUROIMAGING DATASET SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Subjects: {n_subjects}\n\n")
        
        f.write("Demographics:\n")
        f.write(f"- Age: {data['age'].mean():.1f} ± {data['age'].std():.1f} years\n")
        f.write(f"- Sex: {(data['sex'] == 'M').sum()} Male, {(data['sex'] == 'F').sum()} Female\n")
        f.write(f"- Education: {data['education_years'].mean():.1f} ± {data['education_years'].std():.1f} years\n\n")
        
        f.write("Diagnosis Distribution:\n")
        for dx, count in pd.Series(data['diagnosis']).value_counts().items():
            f.write(f"- {dx}: {count} ({100*count/n_subjects:.1f}%)\n")
        
        f.write("\nNeuroimaging Metrics:\n")
        f.write(f"- Brain Volume: {data['brain_volume_cm3'].mean():.1f} ± {data['brain_volume_cm3'].std():.1f} cm³\n")
        f.write(f"- Hippocampal Volume: {data['hippocampal_volume_cm3'].mean():.2f} ± {data['hippocampal_volume_cm3'].std():.2f} cm³\n")
        f.write(f"- Cognitive Score: {data['cognitive_score'].mean():.1f} ± {data['cognitive_score'].std():.1f}\n")
    
    print(f"Synthetic data generated and saved to {output_path}")
    return output_path


def generate_synthetic_fmri(output_dir):
    """Generate synthetic fMRI time series data.
    
    Args:
        output_dir: Directory to save synthetic data
        
    Returns:
        Path to data directory
    """
    import numpy as np
    from scipy import signal
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic fMRI data...")
    
    # Simulate fMRI time series for a brain parcel atlas
    n_timepoints = 200
    n_regions = 68  # AAL atlas has 68 regions
    
    # Generate synthetic BOLD signal
    t = np.arange(n_timepoints)
    
    # Base signal with task-related activation
    base_signal = np.sin(2 * np.pi * 0.05 * t) * 0.5
    
    # Add regional variation
    time_series = np.zeros((n_timepoints, n_regions))
    for i in range(n_regions):
        # Random phase and amplitude for each region
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.3, 1.2)
        
        # Generate signal
        signal_region = amplitude * np.sin(2 * np.pi * 0.05 * t + phase)
        
        # Add noise
        noise = np.random.normal(0, 0.2, n_timepoints)
        
        time_series[:, i] = signal_region + noise + base_signal
    
    # Save time series
    np.save(output_path / 'fmri_timeseries.npy', time_series)
    
    # Generate connectome (correlation matrix)
    from scipy.stats import pearsonr
    
    connectivity = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            if i != j:
                connectivity[i, j], _ = pearsonr(time_series[:, i], time_series[:, j])
    
    np.save(output_path / 'connectivity_matrix.npy', connectivity)
    
    print(f"Synthetic fMRI data generated: {n_timepoints} timepoints, {n_regions} regions")
    return output_path


if __name__ == "__main__":
    download_sample_fmri()
