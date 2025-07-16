import torch
import psutil
import logging
from typing import Dict, Any

class GPUConfig:
    """Dynamic GPU configuration based on available hardware"""
    
    def __init__(self, target_utilization: float = 0.75):
        """
        Initialize GPU configuration with dynamic sizing
        
        Args:
            target_utilization: Target GPU memory utilization (0.75 = 75%)
        """
        self.target_utilization = target_utilization
        self.logger = logging.getLogger(__name__)
        self.config = self._detect_optimal_config()
        
    def _detect_optimal_config(self) -> Dict[str, Any]:
        """Detect optimal GPU configuration based on available hardware"""
        config = {
            'device': 'cpu',
            'gpu_available': False,
            'gpu_memory_gb': 0,
            'target_memory_gb': 0,
            'input_size': 1024,
            'hidden_size': 512,
            'num_patterns': 256,
            'batch_size': 16,
            'parallel_batches': 2
        }
        
        if torch.cuda.is_available():
            try:
                # Get GPU information safely
                gpu_properties = torch.cuda.get_device_properties(0)
                total_memory = gpu_properties.total_memory
                total_memory_gb = total_memory / (1024**3)
                
                # Calculate dynamic target memory usage based on GPU size
                if total_memory_gb >= 8.0:  # For 8GB+ GPUs - use 75% (6GB)
                    target_memory_gb = total_memory_gb * 0.75
                elif total_memory_gb >= 6.0:  # For 6GB GPUs - use 70% 
                    target_memory_gb = total_memory_gb * 0.70
                elif total_memory_gb >= 4.0:  # For 4GB GPUs - use 65%
                    target_memory_gb = total_memory_gb * 0.65
                else:  # For smaller GPUs - use 60%
                    target_memory_gb = total_memory_gb * 0.60
                
                # Update config with GPU information
                config.update({
                    'device': 'cuda',
                    'gpu_available': True,
                    'gpu_memory_gb': total_memory_gb,
                    'target_memory_gb': target_memory_gb,
                    'gpu_name': gpu_properties.name
                })
                
                # Dynamic scaling based on target memory
                memory_scale = target_memory_gb / 2.0  # Scale factor based on 2GB baseline
                
                # Base sizes that scale with available memory
                base_input = 8192
                base_hidden = 4096
                base_patterns = 2048
                base_batch = 32
                base_parallel = 4
                
                # Scale up based on available memory
                config.update({
                    'input_size': int(base_input * memory_scale),
                    'hidden_size': int(base_hidden * memory_scale),
                    'num_patterns': int(base_patterns * memory_scale),
                    'batch_size': int(base_batch * min(memory_scale, 4.0)),  # Cap batch scaling
                    'parallel_batches': int(base_parallel * min(memory_scale, 4.0))  # Cap parallel scaling
                })
                
                # Ensure num_patterns is divisible by common head counts (2, 4, 8, 16, 32)
                for divisor in [32, 16, 8, 4, 2]:
                    if config['num_patterns'] >= divisor:
                        config['num_patterns'] = (config['num_patterns'] // divisor) * divisor
                        break
                
                # Ensure minimum viable sizes
                config['input_size'] = max(config['input_size'], 2048)
                config['hidden_size'] = max(config['hidden_size'], 1024)
                config['num_patterns'] = max(config['num_patterns'], 512)
                config['batch_size'] = max(config['batch_size'], 16)
                config['parallel_batches'] = max(config['parallel_batches'], 2)
                
                # Test GPU memory allocation before proceeding
                if not self._test_gpu_allocation(config):
                    self.logger.warning("GPU memory test failed - falling back to CPU")
                    config['device'] = 'cpu'
                    config['gpu_available'] = False
                    return config
                
                self.logger.info(f"✅ GPU Configuration - SAFE MODE:")
                self.logger.info(f"  Device: {config['gpu_name']}")
                self.logger.info(f"  Total Memory: {total_memory_gb:.1f} GB")
                self.logger.info(f"  Target Memory: {target_memory_gb:.1f} GB")
                self.logger.info(f"  Input Size: {config['input_size']:,}")
                self.logger.info(f"  Hidden Size: {config['hidden_size']:,}")
                self.logger.info(f"  Patterns: {config['num_patterns']:,}")
                self.logger.info(f"  Batch Size: {config['batch_size']}")
                self.logger.info(f"  Parallel Batches: {config['parallel_batches']}")
                
            except Exception as e:
                self.logger.error(f"GPU detection failed: {e}")
                config['device'] = 'cpu'
                config['gpu_available'] = False
                
        else:
            self.logger.warning("CUDA not available - using CPU configuration")
            
        return config
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration for neural network models"""
        return {
            'input_size': self.config['input_size'],
            'hidden_size': self.config['hidden_size'],
            'num_patterns': self.config['num_patterns'],
            'device': self.config['device']
        }
    
    def get_processor_config(self) -> Dict[str, Any]:
        """Get configuration for GPU processor"""
        return {
            'batch_size': self.config['batch_size'],
            'parallel_batches': self.config['parallel_batches'],
            'device': self.config['device']
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        return {
            'gpu_available': self.config['gpu_available'],
            'gpu_memory_gb': self.config['gpu_memory_gb'],
            'target_memory_gb': self.config['target_memory_gb'],
            'target_utilization': self.target_utilization
        }
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in GB based on current configuration"""
        if not self.config['gpu_available']:
            return 0.0
        
        # Rough estimation based on model parameters
        # Each parameter is typically 4 bytes (float32)
        model_params = (
            self.config['input_size'] * self.config['hidden_size'] * 2 +  # Two layers
            self.config['hidden_size'] * self.config['num_patterns'] +
            self.config['num_patterns'] * self.config['input_size']
        )
        
        # Account for batch processing and overhead
        batch_overhead = self.config['batch_size'] * self.config['parallel_batches']
        total_params = model_params * 2  # Two models (PatternRecognizer + HypothesisGenerator)
        
        # Convert to GB (4 bytes per param + overhead)
        estimated_gb = (total_params * 4 + batch_overhead * 1000) / (1024**3)
        
        return estimated_gb
    
    def _test_gpu_allocation(self, config: Dict[str, Any]) -> bool:
        """Test GPU memory allocation to prevent system crashes"""
        try:
            # Test allocation with small tensors first
            device = torch.device('cuda')
            
            # Test basic allocation
            test_tensor = torch.randn(1000, 1000).to(device)
            torch.cuda.empty_cache()
            del test_tensor
            
            # Test model-sized allocation
            input_size = config['input_size']
            hidden_size = config['hidden_size']
            batch_size = config['batch_size']
            
            # Create test tensors similar to what the model will use
            test_input = torch.randn(batch_size, input_size).to(device)
            test_weight = torch.randn(hidden_size, input_size).to(device)
            test_output = torch.matmul(test_input, test_weight.t())
            
            # Check GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)   # GB
                
                self.logger.info(f"🧪 GPU Memory Test:")
                self.logger.info(f"   Allocated: {memory_allocated:.2f} GB")
                self.logger.info(f"   Reserved: {memory_reserved:.2f} GB")
                
                # Clean up test tensors
                del test_input, test_weight, test_output
                torch.cuda.empty_cache()
                
                # Check if memory usage is reasonable
                if memory_reserved > config['target_memory_gb']:
                    self.logger.warning(f"GPU memory test exceeded target: {memory_reserved:.2f} GB > {config['target_memory_gb']:.2f} GB")
                    return False
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"GPU memory test failed: {e}")
            return False
