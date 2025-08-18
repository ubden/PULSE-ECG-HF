"""
PULSE-7B Enhanced Handler
Ubden® Team - Edited by https://github.com/ck-cankurt
Support: Text, Image URLs, and Base64 encoded images
"""

import torch
from typing import Dict, List, Any
import base64
from io import BytesIO
from PIL import Image
import requests
import time

# Import utilities if available
try:
    from utils import (
        performance_monitor, 
        validate_image_input, 
        sanitize_parameters, 
        get_system_info,
        create_health_check,
        deepseek_client
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    deepseek_client = None
    print("⚠️ Utils module not found - performance monitoring and DeepSeek integration disabled")


class EndpointHandler:
    def __init__(self, path=""):
        """
        Hey there! Let's get this PULSE-7B model up and running.
        We'll load it from the HuggingFace hub directly, so no worries about local files.
        
        Args:
            path: Model directory path (we actually ignore this and load from HF hub)
        """
        print("🚀 Starting up PULSE-7B handler...")
        print("📝 Enhanced by Ubden® Team - github.com/ck-cankurt")
        
        # Let's see what hardware we're working with
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Running on: {self.device}")
        
        try:
            # For LLaVA models, we need to load them properly with vision capabilities
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            
            print("📦 Loading LLaVA model with vision capabilities...")
            self.processor = AutoProcessor.from_pretrained("PULSE-ECG/PULSE-7B", trust_remote_code=True)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                "PULSE-ECG/PULSE-7B",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.model.eval()
            self.use_pipeline = False
            print("✅ LLaVA model loaded successfully with vision support!")
            
        except Exception as e:
            print(f"⚠️ LLaVA loading failed: {e}")
            print("🔄 Falling back to pipeline approach...")
            
            try:
                # Fallback - using pipeline but aware it won't handle images properly
                from transformers import pipeline
                
                print("📦 Fetching model from HuggingFace Hub...")
                self.pipe = pipeline(
                    "text-generation",
                    model="PULSE-ECG/PULSE-7B",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device=0 if torch.cuda.is_available() else -1,
                    trust_remote_code=True,
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                        "use_safetensors": True
                    }
                )
                self.use_pipeline = True
                self.processor = None
                print("✅ Model loaded via pipeline (text-only mode)!")
                
            except Exception as e2:
                print(f"😓 Pipeline also failed: {e2}")
                # If all else fails, we'll handle it gracefully
                self.pipe = None
                self.model = None
                self.processor = None
                self.use_pipeline = None

    def process_image_input(self, image_input):
        """
        Handle both URL and base64 image inputs like a champ!
        
        Args:
            image_input: Can be a URL string or base64 encoded image
            
        Returns:
            PIL Image object or None if something goes wrong
        """
        if not image_input or not isinstance(image_input, str):
            print("❌ Invalid image input provided")
            return None
            
        try:
            # Check if it's a URL (starts with http/https)
            if image_input.startswith(('http://', 'https://')):
                print(f"🌐 Fetching image from URL: {image_input[:50]}...")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(image_input, timeout=15, headers=headers)
                response.raise_for_status()
                
                # Verify it's actually an image
                if not response.headers.get('content-type', '').startswith('image/'):
                    print(f"⚠️ URL doesn't seem to point to an image: {response.headers.get('content-type')}")
                
                image = Image.open(BytesIO(response.content)).convert('RGB')
                print(f"✅ Image downloaded successfully! Size: {image.size}")
                return image
            
            # Handle base64 images
            else:
                print("🔍 Processing base64 image...")
                base64_data = image_input
                
                # Remove data URL prefix if it exists (data:image/jpeg;base64,...)
                if image_input.startswith('data:'):
                    if 'base64,' in image_input:
                        base64_data = image_input.split('base64,')[1]
                    else:
                        print("❌ Invalid data URL format - missing base64 encoding")
                        return None
                
                # Clean up any whitespace
                base64_data = base64_data.strip().replace('\n', '').replace('\r', '').replace(' ', '')
                
                # Validate base64 format
                try:
                    # Add padding if necessary
                    missing_padding = len(base64_data) % 4
                    if missing_padding:
                        base64_data += '=' * (4 - missing_padding)
                    
                    image_data = base64.b64decode(base64_data, validate=True)
                except Exception as decode_error:
                    print(f"❌ Invalid base64 encoding: {decode_error}")
                    return None
                
                # Verify it's a valid image
                if len(image_data) < 100:  # Too small to be a real image
                    print("❌ Decoded data too small to be a valid image")
                    return None
                
                image = Image.open(BytesIO(image_data)).convert('RGB')
                print(f"✅ Base64 image decoded successfully! Size: {image.size}")
                return image
                
        except requests.exceptions.Timeout:
            print("❌ Request timeout - image URL took too long to respond")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error while fetching image: {e}")
            return None
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None
        
        return None

    def add_turkish_commentary(self, response: Dict[str, Any], enable_commentary: bool, timeout: int = 30) -> Dict[str, Any]:
        """Add Turkish commentary to the response using DeepSeek API"""
        if not enable_commentary:
            return response
            
        if not UTILS_AVAILABLE or not deepseek_client:
            print("⚠️ DeepSeek client not available - skipping Turkish commentary")
            response["commentary_status"] = "unavailable"
            return response
            
        if not deepseek_client.is_available():
            print("⚠️ DeepSeek API key not configured - skipping Turkish commentary")
            response["commentary_status"] = "api_key_missing"
            return response
            
        generated_text = response.get("generated_text", "")
        if not generated_text:
            print("⚠️ No generated text to comment on")
            response["commentary_status"] = "no_text"
            return response
            
        print("🔄 DeepSeek ile Türkçe yorum ekleniyor...")
        commentary_result = deepseek_client.get_turkish_commentary(generated_text, timeout)
        
        if commentary_result["success"]:
            response["comment_text"] = commentary_result["comment_text"]
            response["commentary_model"] = commentary_result.get("model", "deepseek-chat")
            response["commentary_tokens"] = commentary_result.get("tokens_used", 0)
            response["commentary_status"] = "success"
            print("✅ Türkçe yorum başarıyla eklendi")
        else:
            response["comment_text"] = ""
            response["commentary_error"] = commentary_result["error"]
            response["commentary_status"] = "failed"
            print(f"❌ Türkçe yorum eklenemedi: {commentary_result['error']}")
            
        return response

    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        if UTILS_AVAILABLE:
            return create_health_check()
        else:
            return {
                'status': 'healthy',
                'model': 'PULSE-7B',
                'timestamp': time.time(),
                'handler_version': '2.0.0'
            }

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main processing function - where the magic happens!
        
        Args:
            data: Input data with 'inputs' and optional 'parameters'
        
        Returns:
            List with the generated response
        """
        # Quick check - is our model ready?
        if self.use_pipeline is None:
            return [{
                "generated_text": "Oops! Model couldn't load properly. Please check the deployment settings.",
                "error": "Model initialization failed",
                "handler": "Ubden® Team Enhanced Handler"
            }]
        
        # Performance monitoring
        start_time = time.time()
        request_type = "text_only"
        success = False
        image_processing_time = 0.0
        
        try:
            # Parse the inputs - flexible format support
            inputs = data.get("inputs", "")
            text = ""
            image = None
            
            if isinstance(inputs, dict):
                # Dictionary input - check for text and image
                # Support multiple text field names: query, text, prompt
                text = inputs.get("query", inputs.get("text", inputs.get("prompt", "")))
                
                # Check for image in various formats
                image_input = inputs.get("image", inputs.get("image_url", inputs.get("image_base64", None)))
                if image_input:
                    # Determine request type and validate input
                    if UTILS_AVAILABLE:
                        validation = validate_image_input(image_input)
                        request_type = validation.get('type', 'unknown')
                        if request_type == 'url':
                            request_type = 'image_url'
                    else:
                        request_type = 'image_url' if image_input.startswith(('http://', 'https://')) else 'base64'
                    
                    # Process image with timing
                    image_start = time.time()
                    image = self.process_image_input(image_input)
                    image_processing_time = time.time() - image_start
                    
                    if image:
                        print(f"✅ Image processed successfully: {image.size[0]}x{image.size[1]} pixels")
                        # Store the image for later use with LLaVA model
                        # Don't modify the text prompt - let LLaVA handle the image-text combination
                        print(f"🖼️ Image will be passed to model: {image.size} pixels")
            else:
                # Simple string input
                text = str(inputs)
            
            if not text:
                return [{"generated_text": "Hey, I need some text to work with! Please provide an input."}]
            
            # Get generation parameters with sensible defaults
            parameters = data.get("parameters", {})
            
            # Check if Turkish commentary is requested
            enable_turkish_commentary = parameters.get("enable_turkish_commentary", False)  # Default false
            deepseek_timeout = parameters.get("deepseek_timeout", 30)
            
            # Use utils for parameter sanitization if available
            if UTILS_AVAILABLE:
                sanitized_params = sanitize_parameters(parameters)
                max_new_tokens = sanitized_params["max_new_tokens"]
                temperature = sanitized_params["temperature"]
                top_p = sanitized_params["top_p"]
                repetition_penalty = sanitized_params["repetition_penalty"]
                stop_sequences = sanitized_params["stop"]
                return_full_text = sanitized_params["return_full_text"]
                do_sample = sanitized_params["do_sample"]
            else:
                max_new_tokens = min(parameters.get("max_new_tokens", 512), 2048)
                temperature = max(0.01, min(parameters.get("temperature", 0.2), 2.0))
                top_p = max(0.01, min(parameters.get("top_p", 0.9), 1.0))
                do_sample = parameters.get("do_sample", temperature > 0.01)
                repetition_penalty = max(1.0, min(parameters.get("repetition_penalty", 1.05), 2.0))
                stop_sequences = parameters.get("stop", ["</s>"])
                return_full_text = parameters.get("return_full_text", False)
            
            print(f"🎛️ Generation params: max_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}")
            
            # Using pipeline? Let's go!
            if self.use_pipeline:
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": do_sample,
                    "repetition_penalty": repetition_penalty,
                    "return_full_text": return_full_text
                }
                
                # Add stop sequences if supported
                if stop_sequences and stop_sequences != ["</s>"]:
                    generation_kwargs["stop_sequence"] = stop_sequences[0]  # Most pipelines support single stop
                
                result = self.pipe(text, **generation_kwargs)
                
                # Pipeline returns a list, let's handle it properly
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    # Clean up any stop sequences that might remain
                    for stop_seq in stop_sequences:
                        if generated_text.endswith(stop_seq):
                            generated_text = generated_text[:-len(stop_seq)].rstrip()
                    
                    success = True
                    result = {
                        "generated_text": generated_text,
                        "model": "PULSE-7B",
                        "processing_method": "pipeline"
                    }
                    
                    # Add Turkish commentary if requested
                    result = self.add_turkish_commentary(result, enable_turkish_commentary, deepseek_timeout)
                    
                    # Log performance metrics
                    if UTILS_AVAILABLE:
                        generation_time = time.time() - start_time
                        performance_monitor.log_request(
                            request_type, success, generation_time, image_processing_time
                        )
                    
                    return [result]
                else:
                    success = True
                    result_dict = {
                        "generated_text": str(result),
                        "model": "PULSE-7B", 
                        "processing_method": "pipeline"
                    }
                    
                    # Add Turkish commentary if requested
                    result_dict = self.add_turkish_commentary(result_dict, enable_turkish_commentary, deepseek_timeout)
                    
                    # Log performance metrics
                    if UTILS_AVAILABLE:
                        generation_time = time.time() - start_time
                        performance_monitor.log_request(
                            request_type, success, generation_time, image_processing_time
                        )
                    
                    return [result_dict]
            
            # Manual generation mode (LLaVA with vision support)
            else:
                if hasattr(self, 'processor') and self.processor is not None:
                    # LLaVA model with vision support
                    print("🔥 Using LLaVA model with vision capabilities")
                    
                    if image is not None:
                        # Process both image and text with LLaVA processor
                        inputs = self.processor(text, image, return_tensors="pt")
                        print(f"🖼️ LLaVA processing image + text: '{text[:50]}...'")
                    else:
                        # Text-only processing
                        inputs = self.processor(text, return_tensors="pt")
                        print(f"📝 LLaVA processing text-only: '{text[:50]}...'")
                    
                    # Move inputs to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Generate response
                    with torch.no_grad():
                        generation_kwargs = {
                            **inputs,
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                            "do_sample": do_sample,
                            "repetition_penalty": repetition_penalty,
                        }
                        
                        outputs = self.model.generate(**generation_kwargs)
                    
                    # Decode the response
                    generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                    
                    # Clean up the generated text (remove input prompt)
                    if text in generated_text:
                        generated_text = generated_text.replace(text, "").strip()
                    
                    # Clean up any remaining stop sequences
                    for stop_seq in stop_sequences:
                        if generated_text.endswith(stop_seq):
                            generated_text = generated_text[:-len(stop_seq)].rstrip()
                    
                    print(f"✅ LLaVA generated response: {len(generated_text)} characters")
                    
                else:
                    # Fallback to text-only processing (shouldn't happen with proper LLaVA loading)
                    print("⚠️ No processor available, falling back to text-only mode")
                    
                    # This is the old tokenizer-based approach (without image support)
                    if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                        return [{
                            "generated_text": "",
                            "error": "No tokenizer available for text processing",
                            "model": "PULSE-7B",
                            "processing_method": "manual"
                        }]
                    
                    encoded = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=4096
                    )
                    
                    input_ids = encoded["input_ids"].to(self.device)
                    attention_mask = encoded.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample,
                            repetition_penalty=repetition_penalty,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    generated_ids = outputs[0][input_ids.shape[-1]:]
                    generated_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                
                success = True
                result = {
                    "generated_text": generated_text.strip(),
                    "model": "PULSE-7B",
                    "processing_method": "llava_vision" if (hasattr(self, 'processor') and self.processor is not None) else "manual"
                }
                
                # Add Turkish commentary if requested
                result = self.add_turkish_commentary(result, enable_turkish_commentary, deepseek_timeout)
                
                # Log performance metrics
                if UTILS_AVAILABLE:
                    generation_time = time.time() - start_time
                    performance_monitor.log_request(
                        request_type, success, generation_time, image_processing_time
                    )
                
                return [result]
            
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Log failed request
            if UTILS_AVAILABLE:
                generation_time = time.time() - start_time
                performance_monitor.log_request(
                    request_type, success, generation_time, image_processing_time
                )
            
            return [{
                "generated_text": "",
                "error": error_msg,
                "model": "PULSE-7B",
                "handler": "Ubden® Team Enhanced Handler",
                "success": False
            }]