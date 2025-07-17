#!/usr/bin/env python3
"""
Weights & Biases Analytics Logger for TRUE AGI
Replaces TensorBoard with W&B for Python 3.13 compatibility
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
                print(f"� [W&B] Created new continuous learning session")
                
        except Exception as e:
            # Fallback to new session
            print(f"⚠️ [W&B] Resume failed, creating new session: {e}")
            self._create_new_session()
    
    def _create_new_session(self):
        """Create a completely new W&B session"""
        wandb.init(
            project=self.project_name,
            name=f"AGI_Session_{self.session_id}",
            tags=["TRUE-AGI", "Continuous-Learning", "Neural-Networks"],
            config={
                "session_id": self.session_id,
                "framework": "PyTorch",
                "system": "TRUE AGI Continuous Learning", 
                "start_time": time.time()
            }
        )
        self.epoch = 1
    
    def log_learning_metrics(self, metrics: Dict[str, Any]):
        """Log learning progress metrics"""
        if not self.initialized:
            return
            
        try:
            # Add step to metrics
            metrics["step"] = self.step
            metrics["timestamp"] = time.time()
            
            wandb.log(metrics)
            print(f"📊 [W&B] Logged learning metrics: {list(metrics.keys())}")
        except Exception as e:
            print(f"⚠️ [W&B] Failed to log metrics: {e}")
    
    def log_gpu_performance(self):
        """Log GPU utilization and performance"""
        if not self.initialized:
            return
            
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                gpu_metrics = {
                    "gpu_utilization": gpu.load * 100,
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_total": gpu.memoryTotal,
                    "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "gpu_temperature": gpu.temperature,
                    "step": self.step
                }
                wandb.log(gpu_metrics)
        except Exception as e:
            print(f"⚠️ [W&B] Failed to log GPU metrics: {e}")
    
    def log_neural_network_info(self, model_name: str, model: torch.nn.Module):
        """Log neural network architecture and parameters"""
        if not self.initialized:
            return
            
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            network_info = {
                f"{model_name}_total_parameters": total_params,
                f"{model_name}_trainable_parameters": trainable_params,
                f"{model_name}_model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
                "step": self.step
            }
            
            wandb.log(network_info)
            print(f"🧠 [W&B] Logged {model_name} info: {total_params:,} parameters")
        except Exception as e:
            print(f"⚠️ [W&B] Failed to log network info: {e}")
    
    def log_learning_episode(self, episode_data: Dict[str, Any]):
        """Log a complete learning episode"""
        if not self.initialized:
            return
            
        try:
            episode_data["step"] = self.step
            episode_data["episode_timestamp"] = time.time()
            
            wandb.log(episode_data)
            print(f"🎯 [W&B] Logged learning episode")
        except Exception as e:
            print(f"⚠️ [W&B] Failed to log episode: {e}")
    
    def log_model_save(self, model_path: str, model_name: str, model_size_mb: float):
        """Log model saving events"""
        if not self.initialized:
            return
            
        try:
            save_info = {
                f"{model_name}_saved": True,
                f"{model_name}_size_mb": model_size_mb,
                f"{model_name}_save_path": str(model_path),
                "model_save_timestamp": time.time(),
                "step": self.step
            }
            
            wandb.log(save_info)
            print(f"💾 [W&B] Logged model save: {model_name}")
        except Exception as e:
            print(f"⚠️ [W&B] Failed to log model save: {e}")
    
    def log_learning_event(self, event_type: str, environment_state: Dict, action: Dict, outcome: str):
        """Log learning events for tracking AGI progress"""
        if not self.initialized:
            return
            
        try:
            event_data = {
                "event_type": event_type,
                "outcome": outcome,
                "timestamp": time.time(),
                "step": self.step
            }
            
            # Add environment state if available
            if environment_state:
                event_data.update({f"env_{k}": v for k, v in environment_state.items() if isinstance(v, (int, float, str, bool))})
            
            # Add action if available  
            if action:
                event_data.update({f"action_{k}": v for k, v in action.items() if isinstance(v, (int, float, str, bool))})
            
            wandb.log(event_data)
        except Exception as e:
            print(f"⚠️ [W&B] Failed to log learning event: {e}")
    
    def increment_step(self):
        """Increment the global step counter"""
        self.step += 1
    
    def finish(self):
        """Finish the W&B run"""
        if self.initialized:
            try:
                wandb.finish()
                print("✅ [W&B] Analytics session finished")
            except Exception as e:
                print(f"⚠️ [W&B] Failed to finish session: {e}")


# TensorBoard compatibility wrapper
class TensorBoardAGILogger:
    """Compatibility wrapper that maps TensorBoard calls to W&B"""
    
    def __init__(self, session_id: str = None):
        self.wandb_logger = WandBAGILogger(session_id=session_id)
    
    def log_scalar(self, tag: str, value: float, step: int = None):
        """Map TensorBoard scalar logging to W&B"""
        if self.wandb_logger.initialized:
            metrics = {tag: value}
            if step is not None:
                metrics["step"] = step
            self.wandb_logger.log_learning_metrics(metrics)
    
    def log_learning_metrics(self, metrics: Dict[str, Any]):
        """Direct pass-through to W&B"""
        return self.wandb_logger.log_learning_metrics(metrics)
    
    def log_gpu_performance(self):
        """Direct pass-through to W&B"""
        return self.wandb_logger.log_gpu_performance()
    
    def log_neural_network_info(self, model_name: str, model: torch.nn.Module):
        """Direct pass-through to W&B"""
        return self.wandb_logger.log_neural_network_info(model_name, model)
    
    def increment_step(self):
        """Direct pass-through to W&B"""
        return self.wandb_logger.increment_step()
    
    def finish(self):
        """Direct pass-through to W&B"""
        return self.wandb_logger.finish()
