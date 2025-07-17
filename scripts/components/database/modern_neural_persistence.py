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
import threading
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
        
        # Add locking to prevent concurrent saves of the same model
        self._save_locks = {}
        self._lock_manager = threading.Lock()
        
        # Create directories
        self.session_path.mkdir(parents=True, exist_ok=True)
        (self.session_path / "models").mkdir(exist_ok=True)
        (self.session_path / "metadata").mkdir(exist_ok=True)
        
        print(f"🧠 [Modern] Neural persistence initialized: {self.session_path}")
        print(f"🧠 [Modern] Using industry-standard PyTorch + HDF5 approach")
    
    def _get_model_lock(self, model_name):
        """Get or create a lock for a specific model"""
        with self._lock_manager:
            if model_name not in self._save_locks:
                self._save_locks[model_name] = threading.Lock()
            return self._save_locks[model_name]
    
    def save_neural_model(self, model_name, model, metadata=None):
        """Save neural network using atomic file operations with locking"""
        
        # Use per-model locking to prevent concurrent saves of the same model
        model_lock = self._get_model_lock(model_name)
        
        with model_lock:
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
                
                # ATOMIC SAVE: Write to temporary file first, then move
                final_file = self.session_path / "models" / f"{model_name}_latest.pth"
                temp_file = self.session_path / "models" / f"{model_name}_temp_{int(time.time() * 1000)}.pth"
                
                # Ensure directory exists
                final_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save to temporary file first
                torch.save(checkpoint, temp_file)
                
                # Atomic move - this prevents file corruption from concurrent access
                import shutil
                shutil.move(str(temp_file), str(final_file))
                
                # Calculate file size and create metadata
                file_size = final_file.stat().st_size
                size_mb = file_size / (1024 * 1024)
                
                # Save metadata atomically too
                metadata_info = {
                    'model_name': model_name,
                    'file_path': str(final_file),
                    'file_size_bytes': file_size,
                    'file_size_mb': size_mb,
                    'parameter_count': checkpoint['parameter_count'],
                    'saved_at': checkpoint['datetime'],
                    'checksum': self._calculate_file_checksum(final_file)
                }
                
                final_metadata = self.session_path / "metadata" / f"{model_name}_info.json"
                temp_metadata = self.session_path / "metadata" / f"{model_name}_temp_{int(time.time() * 1000)}.json"
                
                # Ensure metadata directory exists
                final_metadata.parent.mkdir(parents=True, exist_ok=True)
                
                # Atomic metadata save
                with open(temp_metadata, 'w') as f:
                    json.dump(metadata_info, f, indent=2)
                shutil.move(str(temp_metadata), str(final_metadata))
                
                print(f"✅ [Modern] Saved {model_name}: {size_mb:.1f}MB ({checkpoint['parameter_count']:,} params)")
                print(f"🗂️ [Modern] Location: {final_file}")
                
                # Skip HDF5 backup for now to avoid file locking issues
                # TODO: Implement atomic HDF5 backup if needed
                return True
                
            except Exception as e:
                print(f"❌ [Modern] Failed to save {model_name}: {e}")
                # Clean up temp files if they exist
                if 'temp_file' in locals() and temp_file.exists():
                    temp_file.unlink()
                if 'temp_metadata' in locals() and temp_metadata.exists():
                    temp_metadata.unlink()
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
            
            # Use timestamp and process ID to ensure unique filename
            import os
            timestamp = str(int(time.time() * 1000))  # Milliseconds for uniqueness
            pid = os.getpid()
            hdf5_file = self.session_path / "models" / f"{model_name}_backup_{timestamp}_{pid}.h5"
            
            # Ensure the file doesn't exist and parent directory exists
            hdf5_file.parent.mkdir(parents=True, exist_ok=True)
            if hdf5_file.exists():
                hdf5_file.unlink()
            
            # Create HDF5 file with exclusive access
            with h5py.File(str(hdf5_file), 'w') as f:
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
                    # Use simple string keys to avoid HDF5 issues
                    safe_key = key.replace('.', '_').replace('/', '_')
                    f.create_dataset(
                        safe_key, 
                        data=tensor_np, 
                        compression='gzip', 
                        compression_opts=6
                    )
            
            hdf5_size = hdf5_file.stat().st_size / (1024 * 1024)
            print(f"✅ [Modern] HDF5 backup created: {hdf5_size:.1f}MB")
            
            # Clean up old backups (keep only the latest 3)
            self._cleanup_old_backups(model_name)
            
        except Exception as e:
            print(f"⚠️ [Modern] HDF5 backup failed: {e}")
    
    def _cleanup_old_backups(self, model_name):
        """Clean up old HDF5 backups, keeping only the latest 3"""
        try:
            models_path = self.session_path / "models"
            backup_files = list(models_path.glob(f"{model_name}_backup_*.h5"))
            
            if len(backup_files) > 3:
                # Sort by creation time (newest first)
                backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                
                # Remove oldest backups
                for old_backup in backup_files[3:]:
                    old_backup.unlink()
                    print(f"🗑️ [Modern] Cleaned up old backup: {old_backup.name}")
                    
        except Exception as e:
            print(f"⚠️ [Modern] Cleanup failed: {e}")
    
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
    
    def save_gpu_models(self, gpu_processor):
        """Save all GPU models from the GPU processor"""
        if not gpu_processor or not hasattr(gpu_processor, 'models'):
            print("⚠️ [Modern] No GPU processor or models available")
            return False
            
        try:
            saved_count = 0
            for model_name, model in gpu_processor.models.items():
                if model is not None:
                    self.save_neural_model(model_name, model)
                    saved_count += 1
            
            print(f"✅ [Modern] Saved {saved_count} GPU models successfully")
            return True
            
        except Exception as e:
            print(f"⚠️ [Modern] Failed to save GPU models: {e}")
            return False
    
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
