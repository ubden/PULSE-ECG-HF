# PULSE-7B Handler Deployment Guide

## 🚀 Deployment Rehberi

### Gereksinimler
- Python 3.8+
- CUDA 11.8+ (GPU kullanımı için)
- Minimum 16GB RAM (CPU), 8GB VRAM (GPU)

### Kurulum

1. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

2. **Flash Attention (isteğe bağlı, performans için):**
```bash
pip install flash-attn --no-build-isolation
```

### HuggingFace Inference Deployment

#### 1. Model Repository Yapısı
```
your-model-repo/
├── handler.py
├── config.json
├── generation_config.json
├── requirements.txt
├── model.safetensors.index.json
├── tokenizer_config.json
├── special_tokens_map.json
└── tokenizer.model
```

#### 2. Endpoint Oluşturma
```bash
# HuggingFace CLI ile deploy
huggingface-cli login
huggingface-cli repo create your-pulse-endpoint --type=space
```

#### 3. Test Requests

**Image URL ile test:**
```bash
curl -X POST "YOUR_ENDPOINT_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "query": "Analyze this ECG image",
      "image": "https://i.imgur.com/7uuejqO.jpeg"
    },
    "parameters": {
      "temperature": 0.2,
      "max_new_tokens": 512
    }
  }'
```

**Base64 ile test:**
```bash
curl -X POST "YOUR_ENDPOINT_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "query": "What do you see in this ECG?",
      "image": "data:image/jpeg;base64,/9j/4AAQ..."
    },
    "parameters": {
      "temperature": 0.2
    }
  }'
```

### Performans Optimizasyonları

#### GPU Memory Optimizasyonu
- `torch_dtype=torch.bfloat16` kullanın
- `low_cpu_mem_usage=True` ayarlayın
- `device_map="auto"` ile otomatik dağıtım

#### CPU Optimizasyonu
- `torch_dtype=torch.float32` kullanın
- Thread sayısını ayarlayın: `torch.set_num_threads(4)`

### Monitoring ve Debugging

#### Log Seviyeleri
```python
import logging
logging.basicConfig(level=logging.INFO)
```

#### Memory Usage
```python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
```

### Troubleshooting

#### Common Issues:

1. **CUDA Out of Memory**
   - Batch size'ı azaltın
   - `max_new_tokens` değerini düşürün
   - Gradient checkpointing kullanın

2. **Slow Image Processing**
   - Image timeout değerini artırın
   - Image resize threshold ayarlayın

3. **Model Loading Issues**
   - HuggingFace token'ını kontrol edin
   - Network bağlantısını doğrulayın
   - Cache dizinini temizleyin

### Security Best Practices

- Image URL'leri validate edin
- Base64 boyut limitlerini ayarlayın
- Rate limiting uygulayın
- Input sanitization yapın

### Monitoring Metrics

- Response time
- Memory usage
- Error rates
- Image processing success rate
- Token generation speed
