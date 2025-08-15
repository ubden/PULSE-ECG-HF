"""
PULSE-7B Handler Utilities
Ubden® Team - Performance monitoring and helper functions
"""

import time
import torch
import psutil
import logging
import os
import json
import requests
from typing import Dict, Any, Optional
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Performance monitoring utilities for PULSE-7B handler"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'image_url_requests': 0,
            'base64_requests': 0,
            'text_only_requests': 0,
            'total_generation_time': 0.0,
            'total_image_processing_time': 0.0
        }
    
    def log_request(self, request_type: str, success: bool, 
                   generation_time: float = 0.0, 
                   image_processing_time: float = 0.0):
        """Log request metrics"""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        if request_type == 'image_url':
            self.metrics['image_url_requests'] += 1
        elif request_type == 'base64':
            self.metrics['base64_requests'] += 1
        else:
            self.metrics['text_only_requests'] += 1
        
        self.metrics['total_generation_time'] += generation_time
        self.metrics['total_image_processing_time'] += image_processing_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        total_requests = self.metrics['total_requests']
        if total_requests == 0:
            return self.metrics
        
        success_rate = (self.metrics['successful_requests'] / total_requests) * 100
        avg_generation_time = self.metrics['total_generation_time'] / total_requests
        avg_image_processing_time = self.metrics['total_image_processing_time'] / max(
            self.metrics['image_url_requests'] + self.metrics['base64_requests'], 1
        )
        
        return {
            **self.metrics,
            'success_rate_percent': round(success_rate, 2),
            'avg_generation_time_seconds': round(avg_generation_time, 3),
            'avg_image_processing_time_seconds': round(avg_image_processing_time, 3)
        }
    
    def reset_stats(self):
        """Reset all metrics"""
        for key in self.metrics:
            self.metrics[key] = 0 if isinstance(self.metrics[key], int) else 0.0

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.3f}s")
            return result, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed in {execution_time:.3f}s: {e}")
            raise e
    return wrapper

def get_system_info() -> Dict[str, Any]:
    """Get current system resource information"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    system_info = {
        'cpu_usage_percent': cpu_percent,
        'memory_total_gb': round(memory.total / (1024**3), 2),
        'memory_used_gb': round(memory.used / (1024**3), 2),
        'memory_available_gb': round(memory.available / (1024**3), 2),
        'memory_usage_percent': memory.percent
    }
    
    # Add GPU info if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_stats()
        system_info.update({
            'gpu_available': True,
            'gpu_memory_allocated_gb': round(
                torch.cuda.memory_allocated() / (1024**3), 2
            ),
            'gpu_memory_reserved_gb': round(
                torch.cuda.memory_reserved() / (1024**3), 2
            ),
            'gpu_device_name': torch.cuda.get_device_name(0)
        })
    else:
        system_info['gpu_available'] = False
    
    return system_info

def validate_image_input(image_input: str) -> Dict[str, Any]:
    """Validate image input and return metadata"""
    if not image_input or not isinstance(image_input, str):
        return {'valid': False, 'type': None, 'error': 'Invalid input type'}
    
    # Check if URL
    if image_input.startswith(('http://', 'https://')):
        return {
            'valid': True,
            'type': 'url',
            'length': len(image_input),
            'domain': image_input.split('/')[2] if '/' in image_input else 'unknown'
        }
    
    # Check if base64
    elif image_input.startswith('data:image/') or len(image_input) > 100:
        is_data_url = image_input.startswith('data:')
        base64_data = image_input
        
        if is_data_url:
            if 'base64,' in image_input:
                base64_data = image_input.split('base64,')[1]
            else:
                return {'valid': False, 'type': 'base64', 'error': 'Invalid data URL format'}
        
        # Estimate decoded size
        estimated_size = len(base64_data) * 3 // 4
        
        return {
            'valid': True,
            'type': 'base64',
            'is_data_url': is_data_url,
            'base64_length': len(base64_data),
            'estimated_size_bytes': estimated_size,
            'estimated_size_kb': round(estimated_size / 1024, 2)
        }
    
    return {'valid': False, 'type': None, 'error': 'Unrecognized format'}

def sanitize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize and validate generation parameters"""
    sanitized = {}
    
    # Max new tokens
    max_new_tokens = parameters.get('max_new_tokens', 512)
    sanitized['max_new_tokens'] = max(1, min(max_new_tokens, 2048))
    
    # Temperature
    temperature = parameters.get('temperature', 0.2)
    sanitized['temperature'] = max(0.01, min(temperature, 2.0))
    
    # Top-p
    top_p = parameters.get('top_p', 0.9)
    sanitized['top_p'] = max(0.01, min(top_p, 1.0))
    
    # Repetition penalty
    repetition_penalty = parameters.get('repetition_penalty', 1.05)
    sanitized['repetition_penalty'] = max(1.0, min(repetition_penalty, 2.0))
    
    # Stop sequences
    stop = parameters.get('stop', ['</s>'])
    if isinstance(stop, str):
        stop = [stop]
    sanitized['stop'] = stop[:5]  # Limit to 5 stop sequences
    
    # Return full text
    sanitized['return_full_text'] = bool(parameters.get('return_full_text', False))
    
    # Do sample
    sanitized['do_sample'] = bool(parameters.get('do_sample', sanitized['temperature'] > 0.01))
    
    return sanitized

def create_health_check() -> Dict[str, Any]:
    """Create a health check response"""
    try:
        system_info = get_system_info()
        
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'system': system_info,
            'model': 'PULSE-7B',
            'handler_version': '2.0.0',
            'features': [
                'image_url_support',
                'base64_image_support',
                'stop_sequences',
                'parameter_validation',
                'performance_monitoring'
            ]
        }
        
        # Check if system is under stress
        if system_info['memory_usage_percent'] > 90:
            health_status['warnings'] = ['High memory usage']
        
        if system_info['cpu_usage_percent'] > 90:
            health_status.setdefault('warnings', []).append('High CPU usage')
        
        return health_status
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'timestamp': time.time(),
            'error': str(e)
        }

class DeepSeekClient:
    """DeepSeek API client for Turkish commentary"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('deep_key') or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        
    def is_available(self) -> bool:
        """Check if DeepSeek API is available"""
        return bool(self.api_key)
    
    def get_turkish_commentary(self, english_analysis: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Get Turkish commentary for English medical analysis
        
        Args:
            english_analysis: English medical analysis text
            timeout: Request timeout in seconds
            
        Returns:
            Dict with success status and commentary
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "DeepSeek API key not configured",
                "comment_text": ""
            }
        
        try:
            # Prepare the prompt for Turkish medical commentary
            prompt = f"""Bu bir EKG sonucu klinik incelemesi. Aşağıdaki İngilizce medikal analizi Türkçe olarak yorumla ve hasta için anlaşılır bir dilde açıkla:

"{english_analysis}"

Lütfen:
1. Medikal terimleri Türkçe karşılıklarıyla açıkla
2. Hastanın anlayabileceği basit bir dille yaz
3. Gerekirse aciliyet durumu hakkında bilgi ver
4. Kısa ve net ol

Türkçe Yorum:"""

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "Sen deneyimli bir kardiyolog doktorsun. EKG sonuçlarını Türkçe olarak hastalar için anlaşılır şekilde açıklıyorsun."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 500,
                "stream": False
            }
            
            logger.info("🔄 DeepSeek API'ye Türkçe yorum için istek gönderiliyor...")
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                comment_text = result['choices'][0]['message']['content'].strip()
                
                # Clean up the response - remove "Türkçe Yorum:" prefix if present
                if comment_text.startswith("Türkçe Yorum:"):
                    comment_text = comment_text[13:].strip()
                
                logger.info("✅ DeepSeek'ten Türkçe yorum başarıyla alındı")
                
                return {
                    "success": True,
                    "comment_text": comment_text,
                    "model": "deepseek-chat",
                    "tokens_used": result.get('usage', {}).get('total_tokens', 0)
                }
            else:
                return {
                    "success": False,
                    "error": "DeepSeek API'den geçersiz yanıt",
                    "comment_text": ""
                }
                
        except requests.exceptions.Timeout:
            logger.error("❌ DeepSeek API timeout")
            return {
                "success": False,
                "error": "DeepSeek API timeout - istek zaman aşımına uğradı",
                "comment_text": ""
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ DeepSeek API request error: {e}")
            return {
                "success": False,
                "error": f"DeepSeek API bağlantı hatası: {str(e)}",
                "comment_text": ""
            }
            
        except Exception as e:
            logger.error(f"❌ DeepSeek API unexpected error: {e}")
            return {
                "success": False,
                "error": f"DeepSeek API beklenmeyen hata: {str(e)}",
                "comment_text": ""
            }

# Global instances
performance_monitor = PerformanceMonitor()
deepseek_client = DeepSeekClient()
