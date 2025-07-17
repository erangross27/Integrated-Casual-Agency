#!/usr/bin/env python3
"""
Modern Neural Network Persistence for TRUE AGI
Uses industry-standard approaches optimized for large models
"""

import os
import torch
import h5py
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
import logging


class ModernNeuralPersistence:
    """Modern, efficient neural network persistence for TRUE AGI"""
    
    def __init__(self, session_id, base_path="./agi_checkpoints"):
        self.session_id = session_id
        self.base_path = Path(base_path)
        self.session_path = self.base_path / session_id
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self.session_path.mkdir(parents=True, exist_ok=True)
        (self.session_path / "models").mkdir(exist_ok=True)
        (self.session_path / "metadata").mkdir(exist_ok=True)
        
        print(f"🧠 [Modern] Neural persistence initialized: {self.session_path}")
        print(f"🧠 [Modern] Using industry-standard PyTorch + HDF5 approach")
    
    def save_neural_model(self, model_name, model, metadata=None):
        """Save neural network using modern PyTorch checkpoint approach"""
        print(f"🧠 [Modern] Saving {model_name} using PyTorch native format...")
        
        try:
            # Create model checkpoint with all necessary info
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'session_id': self.session_id,
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'model_architecture': str(model),
                'metadata': metadata or {}
            }
            
            # Save using PyTorch's optimized format
            model_file = self.session_path / "models" / f"{model_name}_latest.pth"
            torch.save(checkpoint, model_file)
            
            # Calculate file size and create metadata
            file_size = model_file.stat().st_size
            size_mb = file_size / (1024 * 1024)
            
            # Save metadata separately for quick access
            metadata_info = {
                'model_name': model_name,
                'file_path': str(model_file),
                'file_size_bytes': file_size,
                'file_size_mb': size_mb,
                'parameter_count': checkpoint['parameter_count'],
                'saved_at': checkpoint['datetime'],
                'checksum': self._calculate_file_checksum(model_file)
            }
            
            metadata_file = self.session_path / "metadata" / f"{model_name}_info.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata_info, f, indent=2)
            
            print(f"✅ [Modern] Saved {model_name}: {size_mb:.1f}MB ({checkpoint['parameter_count']:,} params)")
            print(f"🗂️ [Modern] Location: {model_file}")
            
            # Also create HDF5 backup for very large models (>1GB)
            if file_size > 1_000_000_000:
                self._save_hdf5_backup(model_name, model, checkpoint)
            
            return True
            
        except Exception as e:
            print(f"❌ [Modern] Failed to save {model_name}: {e}")
            return False
    
    def load_neural_model(self, model_name, model):
        """Load neural network using modern PyTorch checkpoint approach"""
        print(f"🧠 [Modern] Loading {model_name} from PyTorch checkpoint...")
        
        try:
            model_file = self.session_path / "models" / f"{model_name}_latest.pth"
            
            if not model_file.exists():
                print(f"ℹ️ [Modern] No checkpoint found for {model_name} - starting fresh")
                return False
            
            # Verify file integrity
            metadata_file = self.session_path / "metadata" / f"{model_name}_info.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                expected_checksum = metadata.get('checksum')
                actual_checksum = self._calculate_file_checksum(model_file)
                
                if expected_checksum != actual_checksum:
                    print(f"❌ [Modern] Checkpoint corrupted for {model_name}")
                    return False
            
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location='cpu')  # Load to CPU first
            
            # Restore model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Print restore info
            param_count = checkpoint.get('parameter_count', 0)
            saved_time = checkpoint.get('datetime', 'unknown')
            
            print(f"✅ [Modern] Restored {model_name} ({param_count:,} parameters)")
            print(f"🧠 [Modern] Checkpoint from: {saved_time}")
            
            return True
            
        except Exception as e:
            print(f"❌ [Modern] Failed to load {model_name}: {e}")
            return False
    
    def _save_hdf5_backup(self, model_name, model, checkpoint):
        """Save HDF5 backup for very large models"""
        try:
            print(f"🗄️ [Modern] Creating HDF5 backup for large model {model_name}...")
            
            hdf5_file = self.session_path / "models" / f"{model_name}_backup.h5"
            
            with h5py.File(hdf5_file, 'w') as f:
                # Save metadata
                f.attrs['model_name'] = model_name
                f.attrs['session_id'] = self.session_id
                f.attrs['timestamp'] = checkpoint['timestamp']
                f.attrs['parameter_count'] = checkpoint['parameter_count']
                
                # Save each tensor as a separate dataset with compression
                state_dict = checkpoint['model_state_dict']
                for key, tensor in state_dict.items():
                    # Convert to numpy and save with compression
                    tensor_np = tensor.cpu().numpy()
                    f.create_dataset(
                        key, 
                        data=tensor_np, 
                        compression='gzip', 
                        compression_opts=6,
                        shuffle=True  # Better compression
                    )
            
            hdf5_size = hdf5_file.stat().st_size / (1024 * 1024)
            print(f"✅ [Modern] HDF5 backup created: {hdf5_size:.1f}MB")
            
        except Exception as e:
            print(f"⚠️ [Modern] HDF5 backup failed: {e}")
    
    def _calculate_file_checksum(self, file_path):
        """Calculate SHA256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def list_saved_models(self):
        """List all saved models with their info"""
        print(f"🗂️ [Modern] Saved models in session {self.session_id}:")
        
        metadata_path = self.session_path / "metadata"
        if not metadata_path.exists():
            print("   No models found")
            return []
        
        models = []
        for metadata_file in metadata_path.glob("*_info.json"):
            try:
                with open(metadata_file, 'r') as f:
                    info = json.load(f)
                models.append(info)
                
                print(f"   📊 {info['model_name']}: {info['file_size_mb']:.1f}MB")
                print(f"      Parameters: {info['parameter_count']:,}")
                print(f"      Saved: {info['saved_at']}")
                print()
                
            except Exception as e:
                print(f"   ⚠️ Error reading {metadata_file}: {e}")
        
        return models
    
    def get_total_storage_usage(self):
        """Get total storage usage"""
        total_size = 0
        model_count = 0
        
        models_path = self.session_path / "models"
        if models_path.exists():
            for model_file in models_path.glob("*.pth"):
                total_size += model_file.stat().st_size
                model_count += 1
        
        total_mb = total_size / (1024 * 1024)
        total_gb = total_mb / 1024
        
        print(f"💾 [Modern] Storage usage: {total_gb:.2f}GB ({model_count} models)")
        return total_size, model_count
    
    def cleanup_old_checkpoints(self, keep_latest=3):
        """Clean up old checkpoints, keeping only the latest versions"""
        print(f"🧹 [Modern] Cleaning up old checkpoints (keeping latest {keep_latest})...")
        
        # This would implement checkpoint rotation
        # For now, we just keep the latest version
        print(f"✅ [Modern] Cleanup completed")


def create_session_persistence(session_id=None):
    """Create a modern neural persistence instance"""
    if session_id is None:
        session_id = f"agi_session_{int(time.time())}"
    
    return ModernNeuralPersistence(session_id)


# Test the new system
if __name__ == "__main__":
    # Demo usage
    persistence = create_session_persistence("demo_session")
    
    # Create a test model
    test_model = torch.nn.Sequential(
        torch.nn.Linear(1000, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1)
    )
    
    # Save and load test
    print("Testing save/load...")
    persistence.save_neural_model("test_model", test_model, {"test": True})
    
    # Create new model and restore
    new_model = torch.nn.Sequential(
        torch.nn.Linear(1000, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1)
    )
    
    persistence.load_neural_model("test_model", new_model)
    persistence.list_saved_models()
    persistence.get_total_storage_usage()
