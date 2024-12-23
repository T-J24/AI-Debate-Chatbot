import os
from dotenv import load_dotenv, find_dotenv
from text_generation import Client
import gradio as gr
from flask import Flask, render_template

# Load environment variables
load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']

# Hugging Face client setup
client = Client(os.environ['HF_API_FALCON_BASE'], headers={"Authorization": f"Bearer {hf_api_key}"}, timeout=120)

app = Flask(__name__)

# Function to generate response from the AI
def generate(input, slider):
    output = client.generate(input, max_new_tokens=slider).generated_text
    return output

# Gradio interface
demo = gr.Interface(
    fn=generate, 
    inputs=[gr.Textbox(label="Prompt"), gr.Slider(label="Max new tokens", value=20, maximum=1024, minimum=1)], 
    outputs=[gr.Textbox(label="Completion")]
)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    demo.launch(share=True)
