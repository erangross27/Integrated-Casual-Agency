#!/usr/bin/env python3
"""
Weights & Biases Analytics Logger for TRUE AGI
Persistent epoch-based tracking for continuous learning
"""

import wandb
import time
import torch
from typing import Dict, Any, Optional
import GPUtil
from pathlib import Path


class WandBAGILogger:
    """Weights & Biases logger for TRUE AGI experiments with continuous epoch-based tracking"""
    
    def __init__(self, project_name: str = "TRUE-AGI-System", resume_mode: bool = True):
        self.project_name = project_name
        self.step = 0
        self.epoch = 0
        self.initialized = False
        self.resume_mode = resume_mode
        self.run_name = "TRUE_AGI_Continuous_Learning"
        
        # Persistent epoch tracking file
        self.epoch_file = Path("agi_checkpoints/persistent_epoch.txt")
        self.epoch_file.parent.mkdir(exist_ok=True)
        
        # Load last epoch from file
        self._load_last_epoch()
        
        # Try to resume existing session or create new one
        try:
            if resume_mode:
                # Try to resume existing run
                self._resume_or_create_session()
            else:
                # Create new session
                self._create_new_session()
                
            self.initialized = True
            print(f"✅ [W&B] Analytics initialized - Project: {self.project_name}")
            print(f"🌐 [W&B] Dashboard: https://wandb.ai/your-username/{self.project_name}")
            print(f"📊 [W&B] Continuous Learning | Epoch: {self.epoch} | Step: {self.step}")
        except Exception as e:
            print(f"⚠️ [W&B] Failed to initialize: {e}")
            self.initialized = False
    
    def _load_last_epoch(self):
        """Load the last epoch number from persistent storage"""
        try:
            if self.epoch_file.exists():
                with open(self.epoch_file, 'r') as f:
                    data = f.read().strip().split(',')
                    self.epoch = int(data[0]) if data[0] else 0
                    self.step = int(data[1]) if len(data) > 1 and data[1] else 0
                print(f"📊 [W&B] Resuming from Epoch {self.epoch}, Step {self.step}")
            else:
                self.epoch = 0
                self.step = 0
                print(f"📊 [W&B] Starting fresh - Epoch 0")
        except Exception as e:
            print(f"⚠️ [W&B] Could not load epoch data: {e}")
            self.epoch = 0
            self.step = 0
    
    def _save_epoch_data(self):
        """Save current epoch and step to persistent storage"""
        try:
            with open(self.epoch_file, 'w') as f:
                f.write(f"{self.epoch},{self.step}")
        except Exception as e:
            print(f"⚠️ [W&B] Could not save epoch data: {e}")
    
    def _resume_or_create_session(self):
        """Resume existing session or create new one if none exists"""
        try:
            # Try to resume with a consistent run name
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                resume="allow",  # Resume if exists, create if not
                id=self.run_name.replace("_", "-").lower(),  # Consistent ID
                tags=["TRUE-AGI", "Continuous-Learning", "Neural-Networks", "Epoch-Based"],
                config={
                    "architecture": "TRUE_AGI_Continuous",
                    "learning_type": "continuous_environmental",
                    "epoch_save_interval": 50,
                    "persistent_session": True,
                    "framework": "PyTorch", 
                    "system": "TRUE AGI Continuous Learning",
                    "start_time": time.time(),
                    "tracking_mode": "epoch_based"
                }
            )
            
            # Get current epoch from W&B if resuming
            if wandb.run.resumed:
                wandb_epoch = wandb.run.summary.get("current_epoch", 0)
                wandb_step = wandb.run.summary.get("global_step", 0)
                # Use the higher value between file and W&B
                self.epoch = max(self.epoch, wandb_epoch)
                self.step = max(self.step, wandb_step)
                print(f"🔄 [W&B] Resumed existing run at epoch {self.epoch}, step {self.step}")
            else:
                print(f"🆕 [W&B] Created new continuous learning session")
            
            # Set the current step in W&B
            wandb.run.summary["current_epoch"] = self.epoch
            wandb.run.summary["global_step"] = self.step
            
        except Exception as e:
            # Fallback to new session
            print(f"⚠️ [W&B] Resume failed, creating new session: {e}")
            self._create_new_session()
    
    def _create_new_session(self):
        """Create a completely new W&B session"""
        wandb.init(
            project=self.project_name,
            name=f"{self.run_name}_backup_{int(time.time())}",
            tags=["TRUE-AGI", "Continuous-Learning", "Backup"],
            config={
                "framework": "PyTorch", 
                "system": "TRUE AGI Continuous Learning",
                "start_time": time.time(),
                "backup_session": True
            }
        )
    
    def start_new_epoch(self):
        """Start a new epoch and log it"""
        self.epoch += 1
        self.step = 0
        self._save_epoch_data()
        
        if self.initialized:
            wandb.log({"epoch": self.epoch}, step=self.step)
            wandb.run.summary["current_epoch"] = self.epoch
            print(f"🔄 [W&B] Started Epoch {self.epoch}")
    
    def increment_step(self):
        """Increment step counter and save if it's a milestone"""
        self.step += 1
        self._save_epoch_data()
        
        # Start new epoch every 100 steps for continuous learning progression
        if self.step > 0 and self.step % 100 == 0:
            self.start_new_epoch()
        
        # Save checkpoint every 50 steps
        elif self.step % 50 == 0:
            if self.initialized:
                wandb.run.summary["global_step"] = self.step
                print(f"💾 [W&B] Checkpoint saved at Step {self.step}, Epoch {self.epoch}")
    
    def log_learning_episode(self, episode_data: Dict[str, Any]):
        """Log a learning episode with epoch tracking"""
        if not self.initialized:
            return
        
        try:
            # Add epoch information to the episode data
            episode_data_with_epoch = {
                **episode_data,
                "epoch": self.epoch,
                "global_step": self.step,
                "timestamp": time.time()
            }
            
            wandb.log(episode_data_with_epoch, step=self.step)
            
        except Exception as e:
            print(f"⚠️ [W&B] Failed to log episode: {e}")
    
    def log_learning_metrics(self, metrics: Dict[str, Any]):
        """Log learning metrics with epoch tracking"""
        if not self.initialized:
            return
        
        try:
            # Add epoch information to metrics
            metrics_with_epoch = {
                **metrics,
                "epoch": self.epoch,
                "global_step": self.step
            }
            
            wandb.log(metrics_with_epoch, step=self.step)
            
        except Exception as e:
            print(f"⚠️ [W&B] Failed to log metrics: {e}")
    
    def log_gpu_performance(self):
        """Log GPU performance metrics with epoch tracking"""
        if not self.initialized:
            return
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_metrics = {
                    "gpu_utilization": gpu.load * 100,
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_total": gpu.memoryTotal,
                    "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "gpu_temperature": gpu.temperature,
                    "epoch": self.epoch,
                    "global_step": self.step
                }
                
                wandb.log(gpu_metrics, step=self.step)
                
        except Exception as e:
            print(f"⚠️ [W&B] Failed to log GPU metrics: {e}")
    
    def log_model_checkpoint(self, model_name: str, model_info: Dict[str, Any]):
        """Log model checkpoint information with epoch tracking"""
        if not self.initialized:
            return
        
        try:
            checkpoint_data = {
                f"{model_name}_parameters": model_info.get("parameter_count", 0),
                f"{model_name}_size_mb": model_info.get("size_mb", 0),
                f"{model_name}_saved_at": model_info.get("saved_at", time.time()),
                "epoch": self.epoch,
                "global_step": self.step,
                "model_type": model_name
            }
            
            wandb.log(checkpoint_data, step=self.step)
            print(f"📊 [W&B] Logged {model_name} checkpoint at Epoch {self.epoch}")
            
        except Exception as e:
            print(f"⚠️ [W&B] Failed to log model checkpoint: {e}")
    
    def log_learning_event(self, event_type: str, environment_state: Dict, action: Dict, outcome: str):
        """Log learning events for tracking AGI progress with epoch tracking"""
        if not self.initialized:
            return
            
        try:
            event_data = {
                "event_type": event_type,
                "outcome": outcome,
                "epoch": self.epoch,
                "global_step": self.step,
                "timestamp": time.time()
            }
            
            # Add environment state if available
            if environment_state:
                event_data.update({f"env_{k}": v for k, v in environment_state.items() if isinstance(v, (int, float, str, bool))})
            
            # Add action if available  
            if action:
                event_data.update({f"action_{k}": v for k, v in action.items() if isinstance(v, (int, float, str, bool))})
            
            wandb.log(event_data, step=self.step)
            
        except Exception as e:
            print(f"⚠️ [W&B] Failed to log learning event: {e}")
    
    def log_neural_network_info(self, model_name: str, model: Any):
        """Log neural network architecture and parameters with epoch tracking"""
        if not self.initialized:
            return
            
        try:
            # Count parameters if it's a PyTorch model
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            else:
                total_params = 0
                trainable_params = 0
            
            network_info = {
                f"{model_name}_total_parameters": total_params,
                f"{model_name}_trainable_parameters": trainable_params,
                f"{model_name}_model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
                "epoch": self.epoch,
                "global_step": self.step,
                "model_type": model_name
            }
            
            wandb.log(network_info, step=self.step)
            print(f"🧠 [W&B] Logged {model_name} info: {total_params:,} parameters at Epoch {self.epoch}")
            
        except Exception as e:
            print(f"⚠️ [W&B] Failed to log network info: {e}")
    
    def finish(self):
        """Finish the W&B run"""
        if self.initialized:
            try:
                # Save final state
                self._save_epoch_data()
                wandb.run.summary["final_epoch"] = self.epoch
                wandb.run.summary["final_step"] = self.step
                wandb.finish()
                print(f"✅ [W&B] Session finished at Epoch {self.epoch}, Step {self.step}")
            except Exception as e:
                print(f"⚠️ [W&B] Error finishing session: {e}")


# Test the logger if run directly
if __name__ == "__main__":
    logger = WandBAGILogger()
    logger.start_new_epoch()
    logger.increment_step()
    logger.log_learning_metrics({"test_metric": 42})
    logger.finish()
