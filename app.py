import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    return processor, model

processor, model = load_model()

st.title("🖼️ AI Image Captioning & Q&A App")
st.write("Upload an image and ask a question about it.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)

    st.write("### 📝 Image Caption:")
    st.write(caption)

    question = st.text_input("Ask a question about the image:")

    if question:
        prompt = f"Question: {question} Answer based on this description: {caption}"

        inputs = processor(image, prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_length=100)

        answer = processor.decode(output[0], skip_special_tokens=True)

        st.write("### 🤖 Answer:")
        st.write(answer)
