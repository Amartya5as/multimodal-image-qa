import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

# --------------------
# Load Models
# --------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)

    return clip_model, clip_processor, tokenizer, t5_model

clip_model, clip_processor, tokenizer, t5_model = load_models()

# --------------------
# Tag Generator
# --------------------

def generate_tags(image):
    candidate_labels = [
        "a photo of a dog",
        "a photo of a cat",
        "a photo of a person",
        "a photo of a car",
        "an indoor scene",
        "an outdoor scene"
    ]

    inputs = clip_processor(
        text=candidate_labels,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    probs = probs[0]

    tags = []
    for i, prob in enumerate(probs):
        if prob.item() > 0.20:
            tags.append(candidate_labels[i])

    return tags

# --------------------
# Prompt Builder
# --------------------

def build_prompt(tags, question):
    tag_text = ", ".join(tags)

    prompt = f"""
You are an AI assistant answering questions about an image.

The image contains: {tag_text}.

Question: {question}

Answer clearly:
"""
    return prompt

# --------------------
# Generate Response
# --------------------

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = t5_model.generate(**inputs, max_length=100)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --------------------
# Streamlit UI
# --------------------

st.title("🖼️ AI Image Q&A App")
st.write("Upload an image and ask questions about it.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    tags = generate_tags(image)
    st.write("Detected tags:", tags)

    question = st.text_input("Ask a question about the image:")

    if question:
        prompt = build_prompt(tags, question)
        answer = generate_response(prompt)
        st.write("### Answer:")
        st.write(answer)