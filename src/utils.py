import os
import base64
from PIL import Image
from transformers import AutoTokenizer
import streamlit as st


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_data(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2", low_cpu_mem_usage=True)


@st.cache_data(show_spinner=False)
def get_asset(file_path: str, is_image: bool = False) -> str:
    """
    Load a byte file (image or not) and return its base64 encoded string
    If is_image = True, it returns the PIL Image object
    """
    with open(file_path, "rb") as f:
        if is_image:
            img = Image.open(f)
            img.load()  # Ensures the image data is fully loaded
            return img
        else:
            return base64.b64encode(f.read()).decode("utf-8")


@st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(os.path.join(ROOT_DIR, "src", "style.css"), "r") as f:
        return f"<style>{f.read()}</style>"


def get_chat_message(contents: str = "", align: str = "left") -> str:
    div_class, color, asset = ("AI-line", "rgb(240, 242, 246)", "AI_icon.png") if align != "right" else ("human-line", "rgb(91, 133, 69)", "user_icon.png")
    asset_path = os.path.join(ROOT_DIR, "src", "assets", asset)
    src = f"data:image/gif;base64,{get_asset(asset_path)}"
    icon_code = f"<img class='chat-icon' src='{src}' width=32 height=32 alt='avatar'>"
    return f"""<div class="{div_class}">{icon_code}<div class="chat-bubble" style="background: {color};">&#8203;{contents}</div></div>"""