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
    """Weights & Biases logger for TRUE AGI experiments"""
    
    def __init__(self, project_name: str = "TRUE-AGI-System", session_id: str = None):
        self.project_name = project_name
        self.session_id = session_id or f"agi_session_{int(time.time())}"
        self.step = 0
        self.initialized = False
        
        # Initialize W&B
        try:
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
            self.initialized = True
            print(f"✅ [W&B] Analytics initialized - Project: {self.project_name}")
            print(f"🌐 [W&B] Dashboard: https://wandb.ai/your-username/{self.project_name}")
        except Exception as e:
            print(f"⚠️ [W&B] Failed to initialize: {e}")
            self.initialized = False
    
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
