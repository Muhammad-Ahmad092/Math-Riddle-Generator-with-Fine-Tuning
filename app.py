import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_name, adapter_path):
    """Loads the base GPT-2 model and applies the LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")
    
    # Load LoRA Adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

# Define paths
BASE_MODEL = "gpt2"  # Base GPT-2 model
ADAPTER_PATH = "model"  # Change if adapter path is different

# Load Model with Adapter
model, tokenizer = load_model(BASE_MODEL, ADAPTER_PATH)

# Streamlit UI
st.title("🧩 Math Riddle Generator")
st.write("Generate creative math riddles using an AI model!")

# User Input
prompt = st.text_input("Enter a starting prompt for the riddle:", "Math Riddle: ")

generate_button = st.button("Generate Riddle")

if generate_button:
    if prompt.strip():
        with st.spinner("Generating riddle..."):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, top_p=0.9, do_sample=True)
            riddle = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Generated Riddle:")
        st.write(riddle)
    else:
        st.warning("Please enter a valid prompt!")
