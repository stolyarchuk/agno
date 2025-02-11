import os
from pathlib import Path

import streamlit as st
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.utils.log import logger
from image_extraction import image_processing_agent
from utils import (
    about_widget,
    add_message,
    rename_session_widget,
    session_selector_widget,
)

# Streamlit App Configuration
st.set_page_config(
    page_title="VisioAI Chat",
    page_icon="📷",
    layout="wide",
)

# **App Header**
st.markdown("<h1 class='main-title'>VisioAI 🖼️</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Your AI-powered smart image analysis agent</p>",
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.markdown("#### 🖼️ Image Processing Queries")

    # Model Selection
    model_choice = st.selectbox(
        "🔍 Select Model Provider", ["OpenAI", "Gemini"], index=0
    )

    # API Key Input
    api_key = st.text_input(f"🔑 Enter {model_choice} API Key", type="password")

    # Mode Selection
    mode = st.radio("⚙️ Extraction Mode", ["Auto", "Manual", "Hybrid"], index=0)

    # Manual/Hybrid Mode Instructions
    instruction = None
    if mode in ["Manual", "Hybrid"]:
        instruction = st.text_area(
            "📝 Enter Extraction Instructions", placeholder="Extract number plates..."
        )

    # Session Management
    user_id = st.text_input("👤 User ID (For session tracking)", value="user_123")
    session_id = st.text_input(
        "📂 Session ID (To keep chat history)", value="session_1"
    )

# **Store selections in session_state**
st.session_state["model_choice"] = model_choice
st.session_state["api_key"] = api_key
st.session_state["user_id"] = user_id

# **Ensure the Model is Initialized Properly**
if model_choice == "OpenAI":
    model = OpenAIChat(id="gpt-4o", api_key=api_key)
else:
    model = Gemini(id="gemini-2.0-flash", api_key=api_key)

# **Session Management**
if (
    "image_agent" not in st.session_state
    or st.session_state.get("current_model") != model_choice
):
    logger.info("---*--- Creating new Image Agent ---*---")
    st.session_state["image_agent"] = image_processing_agent(
        model=model, user_id=user_id
    )
    st.session_state["current_model"] = model_choice

image_agent = st.session_state["image_agent"]

# **Session Selector**
session_selector_widget(image_agent)

# **Image Upload**
uploaded_file = st.file_uploader("📤 Upload an Image 📷", type=["png", "jpg", "jpeg"])
image_path = None
if uploaded_file:
    temp_dir = Path("temp_images")
    temp_dir.mkdir(exist_ok=True)
    image_path = temp_dir / uploaded_file.name
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

# **Chat Input**
if prompt := st.chat_input("💬 Ask about the image or anything else..."):
    add_message("user", prompt)

# **Process User Query**
if uploaded_file:
    extracted_data = image_agent.run(image_path, mode, instruction)
    st.json(extracted_data.content)
    add_message("assistant", extracted_data.content)

# **Rename Sessions**
rename_session_widget(image_agent)

# **About Section**
about_widget()
