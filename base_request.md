https://www.base64-image.de/  # image to base64 for test

{
  "inputs": {
    "query": "What are the main features and diagnosis in this ECG image?",
    "image": "https://ruy.app/uploads/ekg3.jpg"
  },
  "parameters": {
    "temperature": 0.3,
    "max_new_tokens": 512,
    "top_p": 0.9,
    "repetition_penalty": 1.02
  }
}



  {
    "inputs": {
      "query": "What are the main features and diagnosis in this ECG image? Provide a concise, clinical answer.",
      "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2..."
    },
    "parameters": {
      "max_new_tokens": 512,
      "temperature": 0.2,
      "top_p": 0.9,
      "repetition_penalty": 1.05,
      "stop": ["</s>"],
      "return_full_text": false
    }
  }


  {
    "inputs": {
      "query": "What are the main features and diagnosis in this ECG image? Provide a concise, clinical answer.",
      "image": "https://i.imgur.com/7uuejqO.jpeg"
    },
    "parameters": {
      "temperature": 0.2,
      "enable_turkish_commentary": true,    
      "deepseek_timeout": 30               
    }
  }