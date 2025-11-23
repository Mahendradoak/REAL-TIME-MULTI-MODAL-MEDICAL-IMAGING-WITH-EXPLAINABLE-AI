import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os

class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, image_folders, transform=None, subset_patients=None):
        """
        Args:
            csv_file: Path to Data_Entry_2017.csv
            image_folders: List of paths to images_001, images_002, etc.
            transform: PyTorch transforms
            subset_patients: List of patient IDs (for train/val/test split)
        """
        self.df = pd.read_csv(csv_file)
        
        # Filter by patient IDs if specified
        if subset_patients is not None:
            self.df = self.df[self.df['Patient ID'].isin(subset_patients)]
        
        # Create binary pneumonia labels
        self.df['has_pneumonia'] = self.df['Finding Labels'].str.contains('Pneumonia', na=False).astype(int)
        
        self.image_folders = image_folders
        self.transform = transform
        
        # Create image path mapping for faster lookup
        self.image_path_map = {}
        for folder in image_folders:
            if os.path.exists(folder):
                print(f"Found folder: {folder} with {len(os.listdir(folder))} files")
                for img_file in os.listdir(folder):
                    if img_file.endswith('.png'):
                        self.image_path_map[img_file] = os.path.join(folder, img_file)
            else:
                print(f"Warning: Folder not found: {folder}")
        
        print(f"Total images mapped: {len(self.image_path_map)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['Image Index']
        label = row['has_pneumonia']
        
        # Load image
        image_path = self.image_path_map.get(image_name)
        if image_path is None:
            raise FileNotFoundError(f"Image {image_name} not found in mapped paths")
        
        image = Image.open(image_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

# Test the dataset
def test_data_loading():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Start small for testing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Your specific paths
    csv_path = "data/raw/archive/Data_Entry_2017.csv"
    
    # Check what folders exist in your archive
    archive_path = "data/raw/archive"
    print(f"Contents of {archive_path}:")
    if os.path.exists(archive_path):
        contents = os.listdir(archive_path)
        for item in contents:
            item_path = os.path.join(archive_path, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/ ({len(os.listdir(item_path))} items)")
            else:
                print(f"  📄 {item}")
    
    # Look for image folders (they might be named differently)
    image_folders = []
    for item in contents:
        item_path = os.path.join(archive_path, item)
        if os.path.isdir(item_path) and "images" in item.lower():
            # Check if it contains images directly or has a subfolder
            subcontents = os.listdir(item_path)
            if any(f.endswith('.png') for f in subcontents):
                image_folders.append(item_path)
            else:
                # Check for images subfolder
                images_subfolder = os.path.join(item_path, "images")
                if os.path.exists(images_subfolder):
                    image_folders.append(images_subfolder)
    
    print(f"\nImage folders found: {image_folders}")
    
    if not image_folders:
        print("❌ No image folders found! Please check your directory structure.")
        return
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at: {csv_path}")
        return
    
    print(f"✅ CSV file found: {csv_path}")
    
    try:
        dataset = PneumoniaDataset(csv_path, image_folders, transform=transform)
        print(f"✅ Dataset created successfully! Size: {len(dataset)}")
        
        # Test loading a small batch
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        images, labels = next(iter(dataloader))
        print(f"✅ Batch loaded successfully!")
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels: {labels}")
        print(f"   Pneumonia cases in batch: {labels.sum().item()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_data_loading()
