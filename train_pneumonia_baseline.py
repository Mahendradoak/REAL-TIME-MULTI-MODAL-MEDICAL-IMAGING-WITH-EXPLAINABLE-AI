import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
import sys
import os
sys.path.append('src')

from data.dataset import PneumoniaDataset
from models.pneumonia_net import PneumoniaNet
import torchvision.transforms as transforms
from tqdm import tqdm

class PneumoniaTrainerOptimized:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {device}")
        
        # Load patient splits
        splits = np.load('data/processed/patient_splits.npz')
        self.train_patients = splits['train_patients']
        self.val_patients = splits['val_patients']
        
        # FAST: Calculate class weights from CSV (not loading images!)
        csv_path = "data/raw/archive/Data_Entry_2017.csv"
        print("📊 Calculating class weights from CSV...")
        df = pd.read_csv(csv_path)
        train_df = df[df['Patient ID'].isin(self.train_patients)]
        
        # Calculate pneumonia distribution
        has_pneumonia = train_df['Finding Labels'].str.contains('Pneumonia', na=False)
        pos_count = has_pneumonia.sum()
        neg_count = len(train_df) - pos_count
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"📈 Train samples: {len(train_df):,}")
        print(f"🔴 Pneumonia cases: {pos_count:,} ({pos_count/len(train_df)*100:.2f}%)")
        print(f"🟢 Normal cases: {neg_count:,} ({neg_count/len(train_df)*100:.2f}%)")
        print(f"⚖️ Positive weight: {pos_weight:.2f}")
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        # Create datasets
        image_folders = [f"data/raw/archive/images_{i:03d}/images" for i in range(1, 13)]
        
        print("📁 Creating training dataset...")
        self.train_dataset = PneumoniaDataset(
            csv_path, image_folders, 
            transform=self.train_transform, 
            subset_patients=self.train_patients
        )
        
        print("📁 Creating validation dataset...")
        self.val_dataset = PneumoniaDataset(
            csv_path, image_folders,
            transform=self.val_transform,
            subset_patients=self.val_patients
        )
        
        print(f"✅ Train dataset: {len(self.train_dataset):,} images")
        print(f"✅ Val dataset: {len(self.val_dataset):,} images")
        
        # CPU-optimized settings
        batch_size = 8  # Safe for CPU
        print(f"🔢 Using batch size: {batch_size}")
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=0  # No multiprocessing for stability
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"📦 Train batches: {len(self.train_loader):,}")
        print(f"📦 Val batches: {len(self.val_loader):,}")
        
        # Model, loss, optimizer
        print("🧠 Creating model...")
        self.model = PneumoniaNet(pretrained=True).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        print("🎉 Setup complete! Ready to train.")
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc="🚂 Training", 
                   leave=True, ncols=100)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions for AUC calculation
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_predictions.extend(probs)
            all_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Print progress every 500 batches
            if batch_idx % 500 == 0 and batch_idx > 0:
                print(f'\n📊 Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        auc = roc_auc_score(all_targets, all_predictions) if len(set(all_targets)) > 1 else 0.0
        
        return avg_loss, auc
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="🔍 Validation", 
                       leave=True, ncols=100)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_predictions.extend(probs)
                all_targets.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        auc = roc_auc_score(all_targets, all_predictions) if len(set(all_targets)) > 1 else 0.0
        
        # Calculate other metrics
        binary_preds = (np.array(all_predictions) > 0.5).astype(int)
        report = classification_report(all_targets, binary_preds, output_dict=True, zero_division=0)
        
        return avg_loss, auc, report
    
    def train(self, num_epochs=3):
        print(f"\n🚀 === STARTING TRAINING FOR {num_epochs} EPOCHS ===")
        
        best_val_auc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\n🔥 EPOCH {epoch+1}/{num_epochs} 🔥")
            
            # Training
            train_loss, train_auc = self.train_epoch()
            print(f"\n📈 TRAIN  - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
            
            # Validation
            val_loss, val_auc, report = self.validate()
            print(f"📊 VAL    - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
            
            if '1' in report:
                precision = report['1']['precision']
                recall = report['1']['recall']
                f1 = report['1']['f1-score']
                print(f"🎯 METRICS - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                os.makedirs('models', exist_ok=True)
                torch.save(self.model.state_dict(), 'models/best_pneumonia_model.pth')
                print(f"💾 NEW BEST MODEL SAVED! AUC: {val_auc:.4f}")
        
        print(f"\n🏁 TRAINING COMPLETE!")
        print(f"🏆 Best Validation AUC: {best_val_auc:.4f}")
        
        # Success criteria evaluation
        if best_val_auc >= 0.85:
            print("🎉 EXCELLENT! AUC ≥ 0.85 - Ready for multi-modal work!")
            print("✅ Week 1 Foundation: COMPLETE")
            return "SUCCESS"
        elif best_val_auc >= 0.75:
            print("⚠️  GOOD FOUNDATION! AUC ≥ 0.75 - Approach validated, needs optimization")
            print("✅ Week 1 Foundation: COMPLETE")
            return "PARTIAL_SUCCESS"
        elif best_val_auc >= 0.65:
            print("🔧 PROMISING START! AUC ≥ 0.65 - Need hyperparameter tuning")
            return "NEEDS_OPTIMIZATION"
        else:
            print("❌ NEEDS DEBUGGING - AUC < 0.65")
            return "NEEDS_DEBUGGING"

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 PNEUMONIA DETECTION BASELINE TRAINER")
    print("=" * 60)
    
    # Check prerequisites
    if not os.path.exists('data/processed/patient_splits.npz'):
        print("❌ Patient splits not found. Run: python src/data/splits.py")
        exit(1)
    
    if not os.path.exists('data/raw/archive/Data_Entry_2017.csv'):
        print("❌ Dataset not found. Check data/raw/archive/ folder")
        exit(1)
    
    try:
        # Start training
        trainer = PneumoniaTrainerOptimized()
        result = trainer.train(num_epochs=3)
        
        print(f"\n🎯 FINAL RESULT: {result}")
        print("\n📊 SUMMARY:")
        print("  - If SUCCESS: Proceed to multi-modal fusion (Week 2)")
        print("  - If PARTIAL_SUCCESS: Optimize hyperparameters then proceed")
        print("  - If NEEDS_OPTIMIZATION: Tune learning rate, batch size, epochs")
        print("  - If NEEDS_DEBUGGING: Check data loading and model architecture")
        
    except Exception as e:
        print(f"❌ TRAINING ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nCommon solutions:")
        print("  1. Reduce batch size (currently 8)")
        print("  2. Check data paths")
        print("  3. Verify PyTorch installation")
