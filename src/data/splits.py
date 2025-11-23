import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def create_patient_splits(csv_path, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Create patient-level train/val/test splits ensuring no patient overlap
    """
    df = pd.read_csv(csv_path)
    
    # Get unique patients
    unique_patients = df['Patient ID'].unique()
    print(f"Total unique patients: {len(unique_patients)}")
    
    # Stratify by pneumonia patients if possible
    pneumonia_patients = df[df['Finding Labels'].str.contains('Pneumonia', na=False)]['Patient ID'].unique()
    print(f"Patients with pneumonia: {len(pneumonia_patients)}")
    
    # First split: train vs (val+test)
    train_patients, temp_patients = train_test_split(
        unique_patients, 
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=np.isin(unique_patients, pneumonia_patients)  # Stratify by pneumonia presence
    )
    
    # Second split: val vs test
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=test_size/(val_size + test_size),
        random_state=random_state
    )
    
    # Verify splits
    train_df = df[df['Patient ID'].isin(train_patients)]
    val_df = df[df['Patient ID'].isin(val_patients)]
    test_df = df[df['Patient ID'].isin(test_patients)]
    
    print("\n=== SPLIT VERIFICATION ===")
    print(f"Train: {len(train_df)} images, {len(train_patients)} patients")
    print(f"Val: {len(val_df)} images, {len(val_patients)} patients")
    print(f"Test: {len(test_df)} images, {len(test_patients)} patients")
    
    # Check pneumonia distribution
    for name, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pneumonia_count = subset['Finding Labels'].str.contains('Pneumonia', na=False).sum()
        print(f"{name} pneumonia: {pneumonia_count} ({pneumonia_count/len(subset)*100:.2f}%)")
    
    return {
        'train_patients': train_patients,
        'val_patients': val_patients, 
        'test_patients': test_patients
    }

if __name__ == "__main__":
    splits = create_patient_splits("data/raw/archive/Data_Entry_2017.csv")
    
    # Save splits for reproducibility
    np.savez('data/processed/patient_splits.npz', **splits)
    print("Splits saved to data/processed/patient_splits.npz")
