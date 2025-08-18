---
license: apache-2.0
datasets:
- PULSE-ECG/ECGInstruct
- PULSE-ECG/ECGBench
- PULSE-ECG/ecg-log
language:
- tr
- en
tags:
- aimedlab
- pulse
- ubden
- ck_cankurt
---
# PULSE-7B Hugging Face Inference Endpoint

[![Model](https://img.shields.io/badge/Model-PULSE--7B-blue)](https://huggingface.co/PULSE-ECG/PULSE-7B)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/downloads/)

This repository provides a custom handler for deploying the **PULSE-7B** ECG analysis model as a Hugging Face Inference Endpoint. PULSE-7B is a specialized large language model designed for ECG interpretation and cardiac health analysis. 

**🚀 Enhanced with DeepSeek Integration**: This handler automatically translates PULSE-7B's English medical analysis into patient-friendly Turkish commentary using DeepSeek AI, providing bilingual ECG interpretation for Turkish healthcare professionals and patients.

## 🚀 Quick Start

### Prerequisites
- Hugging Face account with PRO subscription (for GPU endpoints)
- Hugging Face API token
- Basic knowledge of REST APIs

## 📦 Repository Structure

```
pulse-hf/
├── handler.py              # Custom inference handler with DeepSeek integration
├── utils.py                # Performance monitoring and DeepSeek client
├── requirements.txt        # Python dependencies
├── generation_config.json  # Model generation configuration
├── test_requests.json      # Example request templates
├── deployment_guide.md     # Detailed deployment guide
└── README.md              # This file
```

## 🛠️ Deployment Instructions

### Step 1: Fork or Clone This Repository

1. Go to [https://huggingface.co/ubden/aimedlab-pulse-hf](https://huggingface.co/ubden/aimedlab-pulse-hf)
2. Click "Clone repository" or create your own repository
3. Upload the `handler.py` and `requirements.txt` files

### Step 2: Create Inference Endpoint

1. Navigate to [Hugging Face Inference Endpoints](https://ui.endpoints.huggingface.co/)
2. Click **"New endpoint"**
3. Configure your endpoint:
   - **Model repository**: `ubden/aimedlab-pulse-hf`
   - **Endpoint name**: Choose a unique name
   - **Instance type**: 
     - Minimum: `GPU · medium · 1x NVIDIA A10G · 16GB`
     - Recommended: `GPU · large · 1x NVIDIA A100 · 80GB`
   - **Task**: `Custom`
   - **Container type**: `Default`
   - **Revision**: `main`

4. Click **"Create Endpoint"**
5. Wait for the status to change from `Building` → `Initializing` → `Running`

### Step 3: Configure DeepSeek API Key (Optional)

To enable Turkish commentary feature:

1. Go to your endpoint's **"Environment"** tab
2. In **"Secret Env"** section, add:
   - **Key**: `deep_key`
   - **Value**: Your DeepSeek API key
3. Click **"Update Endpoint"**

**Note**: Without this configuration, the endpoint will work but without Turkish commentary.

### Step 4: Get Your Endpoint URL

Once running, you'll receive an endpoint URL like:
```
https://YOUR-ENDPOINT-NAME.endpoints.huggingface.cloud
```

## 💻 Usage Examples

### cURL

#### Image URL Request (DeepSeek Türkçe Yorum Aktif)
```bash
curl -X POST "https://YOUR-ENDPOINT.endpoints.huggingface.cloud" \
  -H "Authorization: Bearer YOUR_HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "query": "What are the main features and diagnosis in this ECG image? Provide a concise, clinical answer.",
      "image": "https://i.imgur.com/7uuejqO.jpeg"
    },
    "parameters": {
      "max_new_tokens": 512,
      "temperature": 0.2,
      "top_p": 0.9,
      "repetition_penalty": 1.05,
      "enable_turkish_commentary": true,
      "deepseek_timeout": 30
    }
  }'
```

#### Base64 Image Request (DeepSeek Türkçe Yorum Aktif)
```bash
curl -X POST "https://YOUR-ENDPOINT.endpoints.huggingface.cloud" \
  -H "Authorization: Bearer YOUR_HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "query": "What are the main features and diagnosis in this ECG image? Provide a concise, clinical answer.",
      "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
    },
    "parameters": {
      "max_new_tokens": 512,
      "temperature": 0.2,
      "top_p": 0.9,
      "repetition_penalty": 1.05,
      "enable_turkish_commentary": true,
      "deepseek_timeout": 30
    }
  }'
```

#### Image Request (DeepSeek Türkçe Yorum Deaktif)
```bash
curl -X POST "https://YOUR-ENDPOINT.endpoints.huggingface.cloud" \
  -H "Authorization: Bearer YOUR_HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "query": "Analyze this ECG image briefly.",
      "image": "https://i.imgur.com/7uuejqO.jpeg"
    },
    "parameters": {
      "temperature": 0.2,
      "enable_turkish_commentary": false
    }
  }'
```

#### Text-only Request (DeepSeek Türkçe Yorum Aktif)
```bash
curl -X POST "https://YOUR-ENDPOINT.endpoints.huggingface.cloud" \
  -H "Authorization: Bearer YOUR_HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "query": "What are the key features of atrial fibrillation on an ECG?"
    },
    "parameters": {
      "max_new_tokens": 256,
      "temperature": 0.7,
      "top_p": 0.95,
      "enable_turkish_commentary": true
    }
  }'
```

### Python

```python
import requests
import json
import base64
from PIL import Image
from io import BytesIO

class PULSEEndpoint:
    def __init__(self, endpoint_url, hf_token):
        self.endpoint_url = endpoint_url
        self.headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
    
    def analyze_text(self, text, max_new_tokens=256, temperature=0.7, enable_turkish_commentary=True):
        """
        Send text to PULSE-7B endpoint for analysis
        
        Args:
            text: Input text/question about ECG
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            enable_turkish_commentary: Enable DeepSeek Turkish commentary
        
        Returns:
            Generated response with optional Turkish commentary
        """
        payload = {
            "inputs": {
                "query": text
            },
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "do_sample": True,
                "enable_turkish_commentary": enable_turkish_commentary
            }
        }
        
        response = requests.post(
            self.endpoint_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result[0]
        else:
            raise Exception(f"Request failed: {response.status_code} - {response.text}")
    
    def analyze_image_url(self, image_url, query, max_new_tokens=512, temperature=0.2, enable_turkish_commentary=True):
        """
        Analyze ECG image from URL with DeepSeek Turkish commentary
        
        Args:
            image_url: URL of the ECG image
            query: Question about the ECG image
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_turkish_commentary: Enable DeepSeek Turkish commentary
        
        Returns:
            Generated response with optional Turkish commentary
        """
        payload = {
            "inputs": {
                "query": query,
                "image": image_url
            },
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "enable_turkish_commentary": enable_turkish_commentary,
                "deepseek_timeout": 30
            }
        }
        
        response = requests.post(
            self.endpoint_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result[0]
        else:
            raise Exception(f"Request failed: {response.status_code} - {response.text}")
    
    def analyze_image_base64(self, image_path, query, max_new_tokens=512, temperature=0.2, enable_turkish_commentary=True):
        """
        Analyze ECG image from local file with DeepSeek Turkish commentary
        
        Args:
            image_path: Path to local image file
            query: Question about the ECG image
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_turkish_commentary: Enable DeepSeek Turkish commentary
        
        Returns:
            Generated response with optional Turkish commentary
        """
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type = "image/jpeg" if image_path.lower().endswith('.jpg') else "image/png"
            base64_string = f"data:{mime_type};base64,{image_data}"
        
        payload = {
            "inputs": {
                "query": query,
                "image": base64_string
            },
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "enable_turkish_commentary": enable_turkish_commentary,
                "deepseek_timeout": 30
            }
        }
        
        response = requests.post(
            self.endpoint_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result[0]
        else:
            raise Exception(f"Request failed: {response.status_code} - {response.text}")

# Usage example
if __name__ == "__main__":
    # Initialize endpoint
    endpoint = PULSEEndpoint(
        endpoint_url="https://YOUR-ENDPOINT.endpoints.huggingface.cloud",
        hf_token="YOUR_HF_TOKEN"
    )
    
    # Example 1: Text analysis with Turkish commentary
    response = endpoint.analyze_text(
        "What are the characteristics of a normal sinus rhythm?",
        enable_turkish_commentary=True
    )
    print("English Response:", response["generated_text"])
    if "comment_text" in response:
        print("Turkish Commentary:", response["comment_text"])
    
    # Example 2: Image URL analysis with Turkish commentary
    response = endpoint.analyze_image_url(
        image_url="https://i.imgur.com/7uuejqO.jpeg",
        query="What are the main features and diagnosis in this ECG image?",
        enable_turkish_commentary=True
    )
    print("English Analysis:", response["generated_text"])
    if "comment_text" in response:
        print("Turkish Commentary:", response["comment_text"])
    
    # Example 3: Local image analysis with Turkish commentary
    response = endpoint.analyze_image_base64(
        image_path="./ecg_image.jpg",
        query="Analyze this ECG for any abnormalities",
        enable_turkish_commentary=True
    )
    print("English Analysis:", response["generated_text"])
    if "comment_text" in response:
        print("Turkish Commentary:", response["comment_text"])
    
    # Example 4: Analysis without Turkish commentary
    response = endpoint.analyze_image_url(
        image_url="https://i.imgur.com/7uuejqO.jpeg",
        query="Brief ECG analysis",
        enable_turkish_commentary=False
    )
    print("English Only Response:", response["generated_text"])
```

### JavaScript / Node.js

```javascript
// Using fetch (Node.js 18+ or browser)
class PULSEEndpoint {
    constructor(endpointUrl, hfToken) {
        this.endpointUrl = endpointUrl;
        this.headers = {
            'Authorization': `Bearer ${hfToken}`,
            'Content-Type': 'application/json'
        };
    }

    async analyzeText(text, parameters = {}) {
        const payload = {
            inputs: {
                query: text
            },
            parameters: {
                max_new_tokens: parameters.maxNewTokens || 256,
                temperature: parameters.temperature || 0.7,
                top_p: parameters.topP || 0.95,
                do_sample: parameters.doSample !== false,
                enable_turkish_commentary: parameters.enableTurkishCommentary !== false
            }
        };

        try {
            const response = await fetch(this.endpointUrl, {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            return result[0];
        } catch (error) {
            console.error('Error calling PULSE endpoint:', error);
            throw error;
        }
    }

    async analyzeImageUrl(imageUrl, query, parameters = {}) {
        const payload = {
            inputs: {
                query: query,
                image: imageUrl
            },
            parameters: {
                max_new_tokens: parameters.maxNewTokens || 512,
                temperature: parameters.temperature || 0.2,
                top_p: parameters.topP || 0.9,
                repetition_penalty: parameters.repetitionPenalty || 1.05,
                enable_turkish_commentary: parameters.enableTurkishCommentary !== false,
                deepseek_timeout: parameters.deepseekTimeout || 30
            }
        };

        try {
            const response = await fetch(this.endpointUrl, {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            return result[0];
        } catch (error) {
            console.error('Error calling PULSE endpoint:', error);
            throw error;
        }
    }

    async analyzeImageBase64(imageFile, query, parameters = {}) {
        // Convert image file to base64
        const base64String = await this.fileToBase64(imageFile);
        
        const payload = {
            inputs: {
                query: query,
                image: base64String
            },
            parameters: {
                max_new_tokens: parameters.maxNewTokens || 512,
                temperature: parameters.temperature || 0.2,
                top_p: parameters.topP || 0.9,
                repetition_penalty: parameters.repetitionPenalty || 1.05,
                enable_turkish_commentary: parameters.enableTurkishCommentary !== false,
                deepseek_timeout: parameters.deepseekTimeout || 30
            }
        };

        try {
            const response = await fetch(this.endpointUrl, {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            return result[0];
        } catch (error) {
            console.error('Error calling PULSE endpoint:', error);
            throw error;
        }
    }

    async fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }
}

// Usage example
async function main() {
    const pulse = new PULSEEndpoint(
        'https://YOUR-ENDPOINT.endpoints.huggingface.cloud',
        'YOUR_HF_TOKEN'
    );

    try {
        // Example 1: Text analysis with Turkish commentary
        const response1 = await pulse.analyzeText(
            'What are the ECG signs of myocardial infarction?',
            { enableTurkishCommentary: true }
        );
        console.log('English Response:', response1.generated_text);
        if (response1.comment_text) {
            console.log('Turkish Commentary:', response1.comment_text);
        }

        // Example 2: Image URL analysis with Turkish commentary
        const response2 = await pulse.analyzeImageUrl(
            'https://i.imgur.com/7uuejqO.jpeg',
            'What are the main features and diagnosis in this ECG image?',
            { 
                enableTurkishCommentary: true,
                maxNewTokens: 512,
                temperature: 0.2
            }
        );
        console.log('English Analysis:', response2.generated_text);
        if (response2.comment_text) {
            console.log('Turkish Commentary:', response2.comment_text);
        }

        // Example 3: Analysis without Turkish commentary
        const response3 = await pulse.analyzeImageUrl(
            'https://i.imgur.com/7uuejqO.jpeg',
            'Brief ECG analysis',
            { enableTurkishCommentary: false }
        );
        console.log('English Only Response:', response3.generated_text);
    } catch (error) {
        console.error('Error:', error);
    }
}

main();
```

### JavaScript (Browser)

```html
<!DOCTYPE html>
<html>
<head>
    <title>PULSE-7B ECG Analyzer</title>
</head>
<body>
    <h1>ECG Analysis with PULSE-7B</h1>
    <textarea id="input" rows="4" cols="50" 
              placeholder="Enter your ECG-related question..."></textarea>
    <br>
    <button onclick="analyzeECG()">Analyze</button>
    <div id="result"></div>

    <script>
    async function analyzeECG() {
        const input = document.getElementById('input').value;
        const resultDiv = document.getElementById('result');
        
        if (!input) {
            alert('Please enter a question');
            return;
        }
        
        resultDiv.innerHTML = 'Analyzing...';
        
        const endpoint = 'https://YOUR-ENDPOINT.endpoints.huggingface.cloud';
        const token = 'YOUR_HF_TOKEN'; // In production, use secure token management
        
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    inputs: input,
                    parameters: {
                        max_new_tokens: 256,
                        temperature: 0.7
                    }
                })
            });
            
            const result = await response.json();
            resultDiv.innerHTML = `<h3>Analysis Result:</h3><p>${result[0].generated_text}</p>`;
        } catch (error) {
            resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
        }
    }
    </script>
</body>
</html>
```

## 🎛️ Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_new_tokens` | int | 512 | Maximum number of tokens to generate (1-2048) |
| `temperature` | float | 0.2 | Controls randomness (0.01-2.0). Lower = more focused |
| `top_p` | float | 0.9 | Nucleus sampling threshold (0.01-1.0) |
| `top_k` | int | 50 | Top-k sampling parameter |
| `do_sample` | bool | true | Whether to use sampling or greedy decoding |
| `repetition_penalty` | float | 1.05 | Penalty for repeating tokens (1.0-2.0) |
| `enable_turkish_commentary` | bool | true | Enable/disable DeepSeek Turkish commentary |
| `deepseek_timeout` | int | 30 | DeepSeek API timeout in seconds (10-60) |
| `stop` | array | ["</s>"] | Stop sequences for generation |
| `return_full_text` | bool | false | Return full text including input |

## 📊 Response Format

### With DeepSeek Turkish Commentary (Default)

```json
[
  {
    "generated_text": "Answer: This ECG image shows a sinus rhythm with a normal heart rate, indicating a regular cardiac rhythm. The most striking feature is the presence of ST elevation in the inferior leads, which suggests acute myocardial infarction (MI) or acute coronary syndrome.",
    "model": "PULSE-7B",
    "processing_method": "pipeline",
    "comment_text": "Bu EKG sonucu alt duvar miyokard infarktüsü (kalp krizi) bulgularını göstermektedir. Alt derivasyonlarda ST yükselmesi görülmekte olup, bu acil müdahale gerektiren ciddi bir durumdur. Hastanın derhal kardiyoloji uzmanına başvurması ve acil tedavi alması gerekmektedir.",
    "commentary_model": "deepseek-chat",
    "commentary_tokens": 85,
    "commentary_status": "success"
  }
]
```

### Without DeepSeek Turkish Commentary

```json
[
  {
    "generated_text": "Answer: This ECG image shows a sinus rhythm with a normal heart rate, indicating a regular cardiac rhythm. The most striking feature is the presence of ST elevation in the inferior leads, which suggests acute myocardial infarction (MI) or acute coronary syndrome.",
    "model": "PULSE-7B",
    "processing_method": "pipeline"
  }
]
```

### Error Response

```json
[
  {
    "generated_text": "",
    "error": "Error message here",
    "model": "PULSE-7B",
    "handler": "Ubden® Team Enhanced Handler",
    "success": false
  }
]
```

### Commentary Status Values

- `"success"` - Turkish commentary successfully added
- `"failed"` - DeepSeek API error occurred
- `"unavailable"` - Utils module not available
- `"api_key_missing"` - DeepSeek API key not configured
- `"no_text"` - No text available for commentary

## 🏥 Medical Use Cases

### Example Prompts for ECG Analysis:

1. **Rhythm Analysis**
   ```
   "Analyze an ECG showing irregular rhythm with absent P waves and irregularly irregular R-R intervals"
   ```

2. **ST Segment Changes**
   ```
   "What are the ECG criteria for STEMI in different leads?"
   ```

3. **Conduction Abnormalities**
   ```
   "Describe the ECG findings in complete heart block"
   ```

4. **Electrolyte Imbalances**
   ```
   "What ECG changes are seen in hyperkalemia vs hypokalemia?"
   ```

5. **Differential Diagnosis**
   ```
   "List the differential diagnosis for T wave inversions in precordial leads"
   ```

## ⚡ Performance Tips

1. **Optimize Token Length**: Start with fewer tokens and increase as needed
2. **Temperature Settings**: 
   - Use 0.3-0.5 for factual medical information
   - Use 0.7-0.9 for more creative explanations
3. **Batch Processing**: Send multiple requests in parallel for better throughput
4. **Caching**: Implement client-side caching for repeated queries

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Endpoint returns 503 | Wait for endpoint to fully initialize (can take 5-10 minutes) |
| Timeout errors | Reduce `max_new_tokens` or increase client timeout |
| Out of memory | Use a larger GPU instance or reduce batch size |
| Slow response | Consider using temperature=0 for faster greedy decoding |

## 📈 Monitoring

Monitor your endpoint usage at:
```
https://ui.endpoints.huggingface.co/endpoints
```

Check:
- Request count
- Average latency
- Error rate
- GPU utilization

## 🔒 Security Best Practices

1. **Never expose your HF token in client-side code**
2. **Use environment variables for tokens**:
   ```python
   import os
   hf_token = os.getenv("HF_TOKEN")
   ```
3. **Implement rate limiting in production**
4. **Validate and sanitize all inputs**
5. **Use HTTPS only**

## 📝 License

This deployment wrapper is provided under Apache 2.0 License. The PULSE-7B model itself may have different licensing terms - please check [PULSE-ECG/PULSE-7B](https://huggingface.co/PULSE-ECG/PULSE-7B) for model-specific licensing.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
- 🧑🏿‍💻 https://github.com/ck-cankurt
- 🧑🏿‍💻 https://github.com/ubden

## 📧 Support

For issues related to:
- **Endpoint deployment**: Contact Hugging Face support
- **Model performance**: Refer to [PULSE-ECG repository](https://github.com/ubden/ECG-PULSE)
- **This handler**: Open an issue in this repository

## 🙏 Acknowledgments

- PULSE-7B model by [AIMedLab](https://github.com/ubden/ECG-PULSE)
- Hugging Face for the Inference Endpoints platform
- The open-source community

---

**Disclaimer**: This tool is for research and educational purposes. Always consult qualified healthcare professionals for medical decisions.
**Disclaimer**: This tool is for research and educational purposes. Always consult qualified healthcare professionals for medical decisions.