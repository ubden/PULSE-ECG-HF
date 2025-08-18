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
            # First attempt - using pipeline (easiest and most stable way)
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
            print("✅ Model loaded successfully via pipeline!")
            
        except Exception as e:
            print(f"⚠️ Pipeline didn't work out: {e}")
            print("🔄 Let me try a different approach...")
            
            try:
                # Plan B - load model and tokenizer separately
                from transformers import AutoTokenizer, LlamaForCausalLM
                
                # Get the tokenizer ready
                print("📖 Setting up tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "PULSE-ECG/PULSE-7B",
                    trust_remote_code=True
                )
                
                # Load the model as Llama (it works, trust me!)
                print("🧠 Loading the model as Llama...")
                self.model = LlamaForCausalLM.from_pretrained(
                    "PULSE-ECG/PULSE-7B",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                # Quick fix for padding token if it's missing
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                self.model.eval()
                self.use_pipeline = False
                print("✅ Model loaded successfully via direct loading!")
                
            except Exception as e2:
                print(f"😓 That didn't work either: {e2}")
                # If all else fails, we'll handle it gracefully
                self.pipe = None
                self.model = None
                self.tokenizer = None
                self.use_pipeline = None
        else:
            self.use_pipeline = True
        
        # Final status report
        print("\n🔍 Model Loading Status Report:")
        print(f"   - use_pipeline: {self.use_pipeline}")
        print(f"   - model: {'✅ Loaded' if hasattr(self, 'model') and self.model is not None else '❌ None'}")
        print(f"   - tokenizer: {'✅ Loaded' if hasattr(self, 'tokenizer') and self.tokenizer is not None else '❌ None'}")
        print(f"   - pipe: {'✅ Loaded' if hasattr(self, 'pipe') and self.pipe is not None else '❌ None'}")
        
        # Check if any model component loaded successfully
        has_model = hasattr(self, 'model') and self.model is not None
        has_tokenizer = hasattr(self, 'tokenizer') and self.tokenizer is not None
        has_pipe = hasattr(self, 'pipe') and self.pipe is not None
        
        if not (has_model or has_tokenizer or has_pipe):
            print("💥 CRITICAL: No model components loaded successfully!")
        else:
            print("✅ At least one model component loaded successfully")

    def process_image_input(self, image_input):
        """
        Handle both URL and base64 image inputs like a champ!
        
        Args:
            image_input: Can be a URL string or base64 encoded image
            
        Returns:
            PIL Image object or None if something goes wrong
        """
        try:
            # Check if it's a URL (starts with http/https)
            if isinstance(image_input, str) and (image_input.startswith('http://') or image_input.startswith('https://')):
                print(f"🌐 Fetching image from URL: {image_input[:50]}...")
                response = requests.get(image_input, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
                print("✅ Image downloaded successfully!")
                return image
            
            # Must be base64 then
            elif isinstance(image_input, str):
                print("🔍 Decoding base64 image...")
                # Remove the data URL prefix if it exists
                if "base64," in image_input:
                    image_input = image_input.split("base64,")[1]
                
                image_data = base64.b64decode(image_input)
                image = Image.open(BytesIO(image_data)).convert('RGB')
                print("✅ Image decoded successfully!")
                return image
                
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
                        # Keep original text for demo's LLaVA processing
                        # Demo will handle image token insertion properly
                        print(f"🖼️ Image loaded: {image.size[0]}x{image.size[1]} pixels - will use demo's LLaVA format")
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
                
                if LLAVA_AVAILABLE and image is not None:
                    print("🖼️ Using PULSE demo's LLaVA conversation format")
                    
                    # Use demo's conversation template
                    conv_mode = "llava_v1"  # Demo uses llava_v1 for PULSE
                    conv = conv_templates[conv_mode].copy()
                    
                    # Process image like demo
                    image_tensor = process_images([image], self.tokenizer, self.model.config)[0]
                    image_tensor = image_tensor.half().to(self.model.device)
                    
                    # Create conversation like demo
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + text
                    conv.append_message(conv.roles[0], inp)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    
                    # Tokenize with image token like demo
                    input_ids = tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    ).unsqueeze(0).to(self.model.device)
                    
                    # Set up stopping criteria like demo
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                    
                    # Generate like demo
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids,
                            images=image_tensor.unsqueeze(0),  # Add batch dimension
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=max_new_tokens,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria]
                        )
                    
                    # Decode response like demo
                    generated_text = self.tokenizer.decode(
                        outputs[0, input_ids.shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    print(f"✅ PULSE demo style generation: '{generated_text[:100]}...' (length: {len(generated_text)})")
                    
                else:
                    print("🔤 Using basic tokenizer generation (no LLaVA)")
                    
                    # Basic tokenizer approach for text-only or when LLaVA not available
                    encoded = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048
                    )
                    
                    input_ids = encoded["input_ids"].to(self.device)
                    attention_mask = encoded.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=100,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample,
                            repetition_penalty=repetition_penalty,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            early_stopping=False
                        )
                    
                    generated_ids = outputs[0][input_ids.shape[-1]:]
                    generated_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    ).strip()
                
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