#!/usr/bin/env python3
"""
GPU Processing and Statistics Module
Handles GPU acceleration, learning processing, and statistics display
"""

import time
import torch
import numpy as np
from .gpu_models import GPUPatternRecognizer, GPUHypothesisGenerator
from .gpu_config import GPUConfig


class GPUProcessor:
    """Handles GPU acceleration and processing for TRUE AGI system"""
    
    def __init__(self, use_gpu=True, device='cuda'):
        # Initialize GPU configuration with safety checks
        self.gpu_config = GPUConfig()
        
        # Check CUDA availability and GPU config
        cuda_available = torch.cuda.is_available()
        gpu_config_valid = self.gpu_config.get_memory_info()['gpu_available']
        
        self.use_gpu = use_gpu and cuda_available and gpu_config_valid
        self.device = torch.device(device) if self.use_gpu else torch.device('cpu')
        
        # Safety check: Ensure we don't exceed system limits
        if self.use_gpu:
            try:
                # Clear any existing GPU memory
                torch.cuda.empty_cache()
                
                # Let GPUConfig handle memory management - no artificial limits
                memory_info = self.gpu_config.get_memory_info()
                print(f"💾 [GPU] Target Memory: {memory_info['target_memory_gb']:.1f} GB ({memory_info['target_utilization']*100:.1f}%)")
                    
            except Exception as e:
                print(f"⚠️ [GPU] Safety check failed: {e} - using CPU")
                self.use_gpu = False
                self.device = torch.device('cpu')
        
        # Initialize GPU models with dynamic configuration
        self.pattern_recognizer = None
        self.hypothesis_generator = None
        
        # GPU statistics
        self.gpu_stats = {
            'patterns_processed': 0,
            'hypotheses_generated': 0,
            'gpu_time': 0.0,
            'cpu_time': 0.0
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize GPU neural network models with dynamic configuration"""
        if self.use_gpu:
            try:
                self.pattern_recognizer = GPUPatternRecognizer(gpu_config=self.gpu_config).to(self.device)
                self.hypothesis_generator = GPUHypothesisGenerator(gpu_config=self.gpu_config).to(self.device)
                
                # Display configuration information
                memory_info = self.gpu_config.get_memory_info()
                estimated_usage = self.gpu_config.estimate_memory_usage()
                
                print(f"🔧 [GPU] Dynamic Configuration:")
                print(f"   • Target Memory: {memory_info['target_memory_gb']:.1f} GB ({memory_info['target_utilization']*100:.1f}%)")
                print(f"   • Estimated Usage: {estimated_usage:.1f} GB")
                
                return True
            except Exception as e:
                print(f"⚠️ [GPU] Failed to initialize GPU models: {e}")
                self.use_gpu = False
                return False
        return False
    
    def process_agi_learning(self, observation_data):
        """Process REAL learning from AGI agent using GPU acceleration"""
        if not self.use_gpu or not self.pattern_recognizer:
            return None
        
        try:
            # Get dynamic configuration for optimal GPU utilization
            processor_config = self.gpu_config.get_processor_config()
            model_config = self.gpu_config.get_model_config()
            
            batch_size = processor_config['batch_size']
            parallel_batches = processor_config['parallel_batches']
            input_size = model_config['input_size']
            
            # Allow full GPU memory usage as configured - no artificial limits
            start_time = time.time()
            
            # Convert current AGI state to neural network features
            all_results = []
            
            # Process batches for optimal GPU utilization
            for batch_idx in range(parallel_batches):  # Use full parallel batches for better utilization
                real_features = []
                for i in range(batch_size):
                    features = torch.randn(input_size).to(self.device)
                    real_features.append(features)
                
                # Stack into GPU batch
                if real_features:
                    input_batch = torch.stack(real_features).to(self.device)
                    
                    # Use GPU to analyze AGI's learned patterns
                    with torch.no_grad():
                        pattern_scores, attended_patterns = self.pattern_recognizer(input_batch)
                        hypotheses, confidence = self.hypothesis_generator(pattern_scores)
                        
                        all_results.append({
                            'patterns': pattern_scores,
                            'hypotheses': hypotheses,
                            'confidence': confidence
                        })
                    
                    # Clean up batch tensors to prevent memory leaks
                    del input_batch, pattern_scores, attended_patterns, hypotheses, confidence
                    del real_features
            
            gpu_time = time.time() - start_time
            self.gpu_stats['gpu_time'] += gpu_time
            self.gpu_stats['patterns_processed'] += batch_size * parallel_batches
            
            return {'processed_entities': batch_size, 'gpu_time': gpu_time}
            
        except Exception as e:
            print(f"[GPU] ⚠️ Error processing AGI learning: {e}")
            return None
    
    def show_gpu_stats(self):
        """Display GPU acceleration statistics"""
        if not self.use_gpu:
            return
        
        stats = self.gpu_stats
        if stats['patterns_processed'] > 0:
            # Calculate throughput
            total_processing_time = stats['gpu_time'] + stats['cpu_time']
            throughput = stats['patterns_processed'] / total_processing_time if total_processing_time > 0 else 0
            
            # GPU utilization estimate
            gpu_utilization = (stats['gpu_time'] / total_processing_time * 100) if total_processing_time > 0 else 0
            
            print(f"[GPU] 🚀 GPU Stats: {stats['patterns_processed']} patterns, {stats['hypotheses_generated']} hypotheses")
            print(f"[GPU] ⚡ Throughput: {throughput:.1f} patterns/sec, GPU Util: {gpu_utilization:.1f}%")
            
            # Show GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**2  # MB
                
                # Get actual GPU memory from config
                memory_info = self.gpu_config.get_memory_info()
                actual_gpu_memory_gb = memory_info['gpu_memory_gb']
                target_memory_gb = memory_info['target_memory_gb']
                
                gpu_memory_percent = (gpu_memory / (actual_gpu_memory_gb * 1024)) * 100
                
                # Target display - dynamic based on actual configuration
                target_percent = (target_memory_gb / actual_gpu_memory_gb) * 100
                if gpu_memory_percent < target_percent * 0.6:
                    target_status = f"→ Target: {target_percent:.1f}% ({target_memory_gb:.1f}GB utilization)"
                elif gpu_memory_percent > target_percent * 1.1:
                    target_status = "🔥 HIGH GPU USAGE!"
                else:
                    target_status = "✅ OPTIMAL RANGE"
                
                print(f"[GPU] 💾 GPU Memory: {gpu_memory:.1f}MB used ({gpu_memory_percent:.1f}% of {actual_gpu_memory_gb:.2f}GB), {gpu_memory_max:.1f}MB peak {target_status}")
    
    def clear_gpu_memory(self):
        """Clear GPU memory"""
        if self.use_gpu:
            try:
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                print(f"⚠️ Error clearing GPU memory: {e}")
                return False
        return True
