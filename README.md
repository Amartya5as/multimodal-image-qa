🖼️ **Multimodal Image Question Answering App**
An AI-powered web application that allows users to upload an image and ask questions about it.
The system uses vision-language models to detect objects and generate natural language responses.

🚀 **Live Demo**


🧠 **How It Works**
**This application combines:**
1. CLIP for zero-shot image understanding
   Model: openai/clip-vit-base-patch32
   
2. FLAN-T5 for natural language question answering
   Model: google/flan-t5-small

3. Streamlit for web deployment

**The pipeline:**
1. User uploads an image
2. CLIP predicts relevant objects (zero-shot classification)
3. Detected tags are passed into a prompt
4. FLAN-T5 generates a contextual answer

🛠️ **Tech Stack**
1. Python
2. PyTorch
3. HuggingFace Transformers
4. Streamlit

📂 **Project Structure**
Multimodal_Image/
│
├── app.py
├── requirements.txt
└── README.md

▶️ **Run Locally**
pip install -r requirements.txt
streamlit run app.py

🌍 **Deployment**
Deployed using Streamlit Community Cloud.

📌 **Limitations**
1. Tag-based detection (not full scene understanding)
2. Performance depends on candidate labels
3. Lightweight models used for efficiency

🔮 **Future Improvements**
1. Replace tag-based approach with image captioning model (e.g., BLIP)
2. Improve UI/UX
3. Add confidence visualization
4. Deploy scalable version
