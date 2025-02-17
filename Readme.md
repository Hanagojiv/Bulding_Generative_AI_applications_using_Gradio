# 📜 Image Captioning & NLP Tasks with Gradio

Welcome to this repository! 🚀 This project contains two Jupyter Notebooks:

1. **Image Captioning App** 📸📝
2. **NLP Tasks with a Simple Interface** 🧠💬

Both notebooks leverage **Gradio** to build interactive applications for AI-powered tasks. Below is a detailed guide on how to set up, run, and understand each of these notebooks. 💡

---

## 🔧 Installation & Setup

Before running the notebooks, ensure you have the required dependencies installed. You can install them using:

```bash
pip install gradio transformers torch torchvision pillow requests numpy
```

Additionally, if you are using **Hugging Face models**, you might need an API key. Create a `.env` file and add:

```plaintext
HF_API_KEY=your_huggingface_api_key_here
```

Load environment variables in Python using:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 📸 Image Captioning App

### 📝 Overview
This notebook builds an **Image Captioning Application** using the **BLIP model** from Hugging Face, allowing users to upload images and generate textual descriptions.

### 🚀 How It Works
1. **Convert Image to Base64** 📷 → 🔤
   ```python
   import base64
   def image_to_base64_str(pil_image):
       byte_arr = io.BytesIO()
       pil_image.save(byte_arr, format='PNG')
       byte_arr = byte_arr.getvalue()
       return str(base64.b64encode(byte_arr).decode('utf-8'))
   ```
2. **Send Image to Hugging Face Model** 🤖
   ```python
   import requests
   def get_caption(base64_image):
       headers = {"Authorization": f"Bearer {hf_api_key}"}
       data = {"inputs": base64_image}
       response = requests.post(HF_API_SUMMARY_BASE, headers=headers, json=data)
       return response.json()[0]['generated_text']
   ```

![Captioning](/Images/Img1.png)

1. **Build Gradio Interface** 🎨
   ```python
   import gradio as gr
   def captioner(image):
       base64_image = image_to_base64_str(image)
       return get_caption(base64_image)
   
   gr.Interface(fn=captioner, inputs=[gr.Image(type="pil")], outputs=[gr.Textbox()]).launch()
   ```

   ![Captioning2](/Images/Img2.png)
   ![Captioning3](/Images/Img3.png)

---

## 🧠 NLP Tasks with a Simple Interface

### 📝 Overview
This notebook provides an **interactive UI** for performing various NLP tasks like **text generation, summarization, sentiment analysis, and translation**.

### 🚀 How It Works
1. **Load Pre-trained Transformer Models** 🏗️
   ```python
   from transformers import pipeline
   text_generator = pipeline("text-generation", model="gpt2")
   summarizer = pipeline("summarization")
   sentiment_analysis = pipeline("sentiment-analysis")
   ```
2. **Define NLP Processing Functions** 🧩
   ```python
   def generate_text(prompt):
       return text_generator(prompt, max_length=100)[0]['generated_text']

   def summarize_text(text):
       return summarizer(text, max_length=50, min_length=20, do_sample=False)[0]['summary_text']

   def analyze_sentiment(text):
       return sentiment_analysis(text)[0]
   ```
3. **Build Gradio Interface** 🎨
   ```python
   gr.Interface(
       fn=generate_text,
       inputs=[gr.Textbox(lines=2, placeholder="Enter prompt")],
       outputs=[gr.Textbox()]
   ).launch()
   ```
    ![NER](/Images/NER.png)
---

## 🎯 Running the Notebooks
To start the applications:

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
2. **Run the Notebook Cells** ▶️
3. **Click the Gradio Link to Interact with the App** 🎉

---

## 🔗 Resources
- [Hugging Face Models](https://huggingface.co/models)
- [Gradio Documentation](https://gradio.app/)
- [Transformers Library](https://huggingface.co/transformers/)

---

## ❤️ Contributions & Support
Feel free to fork this repository, submit issues, or suggest improvements. Happy coding! 🎨🚀