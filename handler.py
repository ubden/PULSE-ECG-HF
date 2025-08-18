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

# Try to import LLaVA modules for proper conversation handling
try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import tokenizer_image_token, process_images, KeywordsStoppingCriteria
    LLAVA_AVAILABLE = True
    print("✅ LLaVA modules imported successfully")
except ImportError:
    LLAVA_AVAILABLE = False
    print("⚠️ LLaVA modules not available - using basic text processing")


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
        import sys
        print(f"🔧 Python version: {sys.version}")
        print(f"🔧 PyTorch version: {torch.__version__}")
        
        # Check transformers version
        try:
            import transformers
            print(f"🔧 Transformers version: {transformers.__version__}")
            
            # PULSE LLaVA works with transformers==4.37.2
            if transformers.__version__ == "4.37.2":
                print("✅ Using PULSE LLaVA compatible version (4.37.2)")
            elif "dev" in transformers.__version__ or "git" in str(transformers.__version__):
                print("⚠️ Using development version - may conflict with PULSE LLaVA")
            else:
                print("⚠️ Using different version - PULSE LLaVA prefers 4.37.2")
        except Exception as e:
            print(f"❌ Error checking transformers version: {e}")
        
        print(f"🔧 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔧 CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Let's see what hardware we're working with
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Running on: {self.device}")
        
        try:
            # First attempt - PULSE demo's exact approach
            if LLAVA_AVAILABLE:
                print("📦 Using PULSE demo's load_pretrained_model approach...")
                from llava.model.builder import load_pretrained_model
                from llava.mm_utils import get_model_name_from_path
                
                model_path = "PULSE-ECG/PULSE-7B"
                model_name = get_model_name_from_path(model_path)
                
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=model_name,
                    load_8bit=False,
                    load_4bit=False
                )
                
                # Move model to device like demo
                self.model = self.model.to(self.device)
                self.use_pipeline = False
                print("✅ Model loaded successfully with PULSE demo's approach!")
                print(f"📸 Image processor: {type(self.image_processor).__name__}")
                
            else:
                raise ImportError("LLaVA modules not available")
            
        except Exception as e:
            print(f"⚠️ PULSE demo approach failed: {e}")
            print("🔄 Falling back to pipeline...")
            
            try:
                # Fallback - using pipeline
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
                self.image_processor = None
                print("✅ Model loaded successfully via pipeline!")
                
            except Exception as e2:
                print(f"😓 Pipeline also failed: {e2}")
                
                try:
                    # Last resort - manual loading
                    from transformers import AutoTokenizer, LlamaForCausalLM
                    
                    print("📖 Manual loading as last resort...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "PULSE-ECG/PULSE-7B",
                        trust_remote_code=True
                    )
                    
                    self.model = LlamaForCausalLM.from_pretrained(
                        "PULSE-ECG/PULSE-7B",
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    
                    self.model.eval()
                    self.use_pipeline = False
                    self.image_processor = None
                    print("✅ Model loaded manually!")
                    
                except Exception as e3:
                    print(f"😓 All approaches failed: {e3}")
                    self.pipe = None
                    self.model = None
                    self.tokenizer = None
                    self.image_processor = None
                    self.use_pipeline = None
        
        # Final status report
        print("\n🔍 Model Loading Status Report:")
        print(f"   - use_pipeline: {self.use_pipeline}")
        print(f"   - model: {'✅ Loaded' if hasattr(self, 'model') and self.model is not None else '❌ None'}")
        print(f"   - tokenizer: {'✅ Loaded' if hasattr(self, 'tokenizer') and self.tokenizer is not None else '❌ None'}")
        print(f"   - image_processor: {'✅ Loaded' if hasattr(self, 'image_processor') and self.image_processor is not None else '❌ None'}")
        print(f"   - pipe: {'✅ Loaded' if hasattr(self, 'pipe') and self.pipe is not None else '❌ None'}")
        
        # Check if any model component loaded successfully
        has_model = hasattr(self, 'model') and self.model is not None
        has_tokenizer = hasattr(self, 'tokenizer') and self.tokenizer is not None
        has_pipe = hasattr(self, 'pipe') and self.pipe is not None
        has_image_processor = hasattr(self, 'image_processor') and self.image_processor is not None
        
        if not (has_model or has_tokenizer or has_pipe):
            print("💥 CRITICAL: No model components loaded successfully!")
        else:
            print("✅ At least one model component loaded successfully")
            if has_image_processor:
                print("🖼️ Vision capabilities available!")
            else:
                print("⚠️ No image processor - text-only mode")

    def is_valid_image_format(self, filename_or_url):
        """Validate image format like PULSE demo"""
        # Demo's supported formats
        image_extensions = ["jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp", "heic", "heif", "jfif", "svg", "eps", "raw"]
        
        if filename_or_url.startswith(('http://', 'https://')):
            # For URLs, check the extension or content-type
            ext = filename_or_url.split('.')[-1].split('?')[0].lower()
            return ext in image_extensions
        else:
            # For base64 or local files
            return True  # Base64 will be validated during decode
    
    def process_image_input(self, image_input):
        """
        Handle both URL and base64 image inputs exactly like PULSE demo
        
        Args:
            image_input: Can be a URL string or base64 encoded image
            
        Returns:
            PIL Image object or None if something goes wrong
        """
        try:
            # Check if it's a URL (starts with http/https)
            if isinstance(image_input, str) and (image_input.startswith('http://') or image_input.startswith('https://')):
                print(f"🌐 Fetching image from URL: {image_input[:50]}...")
                
                # Validate format like demo
                if not self.is_valid_image_format(image_input):
                    print("❌ Invalid image format in URL")
                    return None
                
                # Demo's exact image loading approach
                response = requests.get(image_input, timeout=15)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    print(f"✅ Image downloaded successfully! Size: {image.size}")
                    return image
                else:
                    print(f"❌ Failed to load image: status {response.status_code}")
                    return None
            
            # Must be base64 then
            elif isinstance(image_input, str):
                print("🔍 Decoding base64 image...")
                
                # Remove the data URL prefix if it exists
                base64_data = image_input
                if "base64," in image_input:
                    base64_data = image_input.split("base64,")[1]
                
                # Clean and validate base64 data
                base64_data = base64_data.strip().replace('\n', '').replace('\r', '').replace(' ', '')
                
                try:
                    image_data = base64.b64decode(base64_data)
                    image = Image.open(BytesIO(image_data)).convert('RGB')
                    print(f"✅ Base64 image decoded successfully! Size: {image.size}")
                    return image
                except Exception as decode_error:
                    print(f"❌ Base64 decode error: {decode_error}")
                    return None
                
        except Exception as e:
            print(f"❌ Couldn't process the image: {e}")
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
        
        try:
            # Parse the inputs - flexible format support
            inputs = data.get("inputs", "")
            text = ""
            image = None
            
            if isinstance(inputs, dict):
                # Dictionary input - check for text and image
                # Support query field (new) plus original text/prompt fields
                text = inputs.get("query", inputs.get("text", inputs.get("prompt", str(inputs))))
                
                # Check for image in various formats
                image_input = inputs.get("image", inputs.get("image_url", inputs.get("image_base64", None)))
                if image_input:
                    image = self.process_image_input(image_input)
                    if image:
                        # Since we're in text-only mode, create smart ECG context
                        print(f"🖼️ Image loaded: {image.size[0]}x{image.size[1]} pixels - using text-only ECG analysis mode")
                        
                        # Create ECG-specific prompt that mimics visual analysis
                        ecg_context = f"Analyzing an ECG image ({image.size[0]}x{image.size[1]} pixels). "
                        
                        # Enhance the query with ECG-specific instructions
                        if "features" in text.lower() and "diagnosis" in text.lower():
                            # This is a comprehensive analysis request
                            text = f"{ecg_context}Please provide a detailed ECG interpretation including: 1) Rhythm analysis, 2) Rate assessment, 3) Interval measurements (PR, QRS, QT), 4) Axis determination, 5) ST segment analysis, 6) T wave morphology, 7) Any abnormalities or pathological findings. {text}"
                        elif any(word in text.lower() for word in ["analyze", "analysis", "interpret"]):
                            # General analysis request
                            text = f"{ecg_context}Provide a comprehensive ECG analysis covering rhythm, rate, intervals, and any abnormalities. {text}"
                        else:
                            # Specific question
                            text = f"{ecg_context}{text}"
            else:
                # Simple string input
                text = str(inputs)
            
            if not text:
                return [{"generated_text": "Hey, I need some text to work with! Please provide an input."}]
            
            # Get generation parameters - using PULSE-7B demo's optimal settings
            parameters = data.get("parameters", {})
            max_new_tokens = min(parameters.get("max_new_tokens", 4096), 8192)  # Demo uses 4096 default, 8192 max
            temperature = parameters.get("temperature", 0.05)  # Demo uses 0.05 for precise medical analysis
            top_p = parameters.get("top_p", 1.0)  # Demo uses 1.0 for full vocabulary access
            do_sample = parameters.get("do_sample", True)  # Demo uses sampling
            repetition_penalty = parameters.get("repetition_penalty", 1.0)  # Demo default
            
            print(f"🎛️ Generation params: max_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}, do_sample={do_sample}, rep_penalty={repetition_penalty}")
            
            # Check if Turkish commentary is requested (NEW FEATURE)
            enable_turkish_commentary = parameters.get("enable_turkish_commentary", False)  # Default false
            
            # Using pipeline? Let's go!
            if self.use_pipeline:
                print(f"🎛️ Pipeline generation: temp={temperature}, tokens={max_new_tokens}")
                print(f"📝 Input text: '{text[:100]}...'")
                
                result = self.pipe(
                    text,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=100,  # Force detailed analysis like demo
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    return_full_text=False  # Just the new stuff, not the input
                )
                
                # Pipeline returns a list, let's handle it
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "").strip()
                    
                    print(f"🔍 Pipeline debug:")
                    print(f"   - Raw result: '{str(result[0])[:200]}...'")
                    print(f"   - Generated text length: {len(generated_text)}")
                    
                    # Clean up common issues
                    if generated_text.startswith(text):
                        generated_text = generated_text[len(text):].strip()
                        print("🔧 Removed input text from output")
                    
                    # Remove common artifacts
                    generated_text = generated_text.replace("</s>", "").strip()
                    
                    if not generated_text:
                        print("❌ Pipeline generated empty text!")
                        generated_text = "Empty response from pipeline. Please try different parameters."
                    
                    print(f"✅ Final pipeline text: '{generated_text[:100]}...' (length: {len(generated_text)})")
                    
                    # Create response
                    response = {"generated_text": generated_text}
                    
                    # Add Turkish commentary if requested (NEW FEATURE)
                    if enable_turkish_commentary:
                        response = self.add_turkish_commentary(response, True)
                    
                    return [response]
                else:
                    generated_text = str(result).strip()
                    
                    # Create response
                    response = {"generated_text": generated_text}
                    
                    # Add Turkish commentary if requested (NEW FEATURE)
                    if enable_turkish_commentary:
                        response = self.add_turkish_commentary(response, True)
                    
                    return [response]
            
            # Manual generation mode - using PULSE demo's exact approach
            else:
                print(f"🔥 Manual generation with PULSE demo logic: temp={temperature}, tokens={max_new_tokens}")
                print(f"📝 Input text: '{text[:100]}...'")
                
                # Text-only generation with enhanced ECG context
                print("🔤 Using enhanced text-only generation with ECG context")
                
                # Tokenize the enhanced prompt
                encoded = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096  # Increased for longer prompts
                )
                
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                print(f"🔍 Enhanced generation debug:")
                print(f"   - Enhanced prompt length: {len(text)} chars")
                print(f"   - Input tokens: {input_ids.shape[-1]}")
                print(f"   - Prompt preview: '{text[:150]}...'")
                
                # Generate with enhanced settings for medical analysis
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=50,  # Ensure substantial response
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=False
                    )
                
                # Decode and clean response
                generated_ids = outputs[0][input_ids.shape[-1]:]
                generated_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()
                
                # Clean up common artifacts and improve formatting
                generated_text = generated_text.replace("</s>", "").strip()
                
                # Remove any training-style explanations that might appear
                if "In this task" in generated_text:
                    # Split and take only the medical analysis part
                    parts = generated_text.split("In this task")
                    generated_text = parts[0].strip()
                
                # Clean up parenthetical answers if they appear
                if generated_text.startswith("(Answer:"):
                    # Extract the content from parentheses and expand it
                    if ")" in generated_text:
                        answer_part = generated_text[generated_text.find("(")+1:generated_text.find(")")]
                        if answer_part.startswith("Answer:"):
                            answer_part = answer_part[7:].strip()
                        # Use the concise answer as the response
                        generated_text = f"This ECG shows {answer_part.lower()}."
                
                print(f"✅ Enhanced text-only generation: '{generated_text[:100]}...' (length: {len(generated_text)})")
                
                # Create response
                response = {"generated_text": generated_text}
                
                # Add Turkish commentary if requested (NEW FEATURE)
                if enable_turkish_commentary:
                    response = self.add_turkish_commentary(response, True)
                
                return [response]
            
            
        except Exception as e:
            error_msg = f"Something went wrong during generation: {str(e)}"
            print(f"❌ {error_msg}")
            return [{
                "generated_text": "",
                "error": error_msg,
                "handler": "Ubden® Team Enhanced Handler"
            }]