import os
from pathlib import Path

import streamlit as st
from agno.utils.pprint import pprint_run_response
from image_workflow import ImageProcessingWorkflow  # Import the Agno workflow

# Streamlit App Configuration
st.set_page_config(
    page_title="VisioAI",
    page_icon="📷",
    layout="wide",
)

# Main Content
st.title("VisioAI 🖼️")
st.markdown(":orange_heart: **Built using [Agno](https://github.com/agno-agi/agno)**")

# Sidebar Section
with st.sidebar:
    st.markdown("🚀 AI-powered image understanding at your fingertips!")
    # Model Selection
    model_provider = st.selectbox(
        "🔍 Select Model Provider", ["OpenAI", "Gemini"], index=0
    )

    # API Key Input
    api_key = st.text_input(f"🔑 Enter {model_provider} API Key", type="password")

    # Mode Selection (Auto, Manual, Hybrid)
    mode = st.radio("⚙️ Select Extraction Mode", ["Auto", "Manual", "Hybrid"], index=0)

    # Manual/Hybrid Mode: Get User Instructions
    instruction = None
    if mode in ["Manual", "Hybrid"]:
        instruction = st.text_area(
            "📝 Enter Instructions (For Manual/Hybrid Mode)",
            placeholder="Example: Extract number plates from all vehicles.",
        )

    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown(
        """
        - **Upload an image** <br>
        - **Select a model & enter API Key** <br>
        - **Choose extraction mode** <br>
        - **Click Extract to process the image** <br>
        - **Get structured JSON output!**
        """,
        unsafe_allow_html=True,
    )

# Main Content
st.markdown(
    "Extract **valuable insights** and **structured information** from images using VisioAI. 🚀"
)

# Image Upload
uploaded_file = st.file_uploader("📤 Upload an Image 📷", type=["png", "jpg", "jpeg"])

# Submit Button
if st.button("🚀 Extract Information"):
    if not api_key:
        st.error("⚠️ Please enter an API key.")
    elif not uploaded_file:
        st.error("⚠️ Please upload an image.")
    else:
        # Save uploaded file to a temporary location
        temp_dir = Path("temp_images")
        temp_dir.mkdir(exist_ok=True)
        image_path = temp_dir / uploaded_file.name
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run Workflow
        try:
            st.info("⏳ Processing Image... It might take 1-2 minutes")
            workflow = ImageProcessingWorkflow()
            responses = workflow.run(
                image_path=image_path,
                mode=mode,
                instruction=instruction,
                model_choice=model_provider,
                api_key=api_key,
            )

            # Display Output
            st.success("✅ Image analysis completed!")
            st.subheader("📊 Extracted Data:")
            st.json(responses.content)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Footer Section
st.markdown("---")
st.markdown(
    """
    🌟 **Features:**
    - 📸 Extract **objects, text, and insights** from any image.
    - 🧠 **Supports multiple models provider** (OpenAI, Gemini).
    - ⚙️ Works in **Auto, Manual, and Hybrid** modes.
    - 📥 **Download structured JSON** for further use.
    """,
    unsafe_allow_html=True,
)
st.markdown("---")
st.markdown(":orange_heart: **Thank you for using Visio AI!**")
