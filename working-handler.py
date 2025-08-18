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
                text = inputs.get("text", inputs.get("prompt", str(inputs)))
                
                # Check for image in various formats
                image_input = inputs.get("image", inputs.get("image_url", inputs.get("image_base64", None)))
                if image_input:
                    image = self.process_image_input(image_input)
                    if image:
                        # For now, we'll add a note about the image since we're text-only
                        text = f"[Image provided - {image.size[0]}x{image.size[1]} pixels] {text}"
            else:
                # Simple string input
                text = str(inputs)
            
            if not text:
                return [{"generated_text": "Hey, I need some text to work with! Please provide an input."}]
            
            # Get generation parameters with sensible defaults
            parameters = data.get("parameters", {})
            max_new_tokens = min(parameters.get("max_new_tokens", 256), 1024)
            temperature = parameters.get("temperature", 0.7)
            top_p = parameters.get("top_p", 0.95)
            do_sample = parameters.get("do_sample", True)
            repetition_penalty = parameters.get("repetition_penalty", 1.0)
            
            # Using pipeline? Let's go!
            if self.use_pipeline:
                result = self.pipe(
                    text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    return_full_text=False  # Just the new stuff, not the input
                )
                
                # Pipeline returns a list, let's handle it
                if isinstance(result, list) and len(result) > 0:
                    return [{"generated_text": result[0].get("generated_text", "")}]
                else:
                    return [{"generated_text": str(result)}]
            
            # Manual generation mode
            else:
                # Tokenize the input
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
                
                # Generate the response
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode only the new tokens (not the input)
                generated_ids = outputs[0][input_ids.shape[-1]:]
                generated_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                return [{"generated_text": generated_text}]
            
        except Exception as e:
            error_msg = f"Something went wrong during generation: {str(e)}"
            print(f"❌ {error_msg}")
            return [{
                "generated_text": "",
                "error": error_msg,
                "handler": "Ubden® Team Enhanced Handler"
            }]