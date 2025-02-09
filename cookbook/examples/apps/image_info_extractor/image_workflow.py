import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.utils.log import logger
from agno.utils.pprint import pprint_run_response
from agno.workflow import RunResponse, Workflow
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Toggle for debugging mode
DEMO_MODE = True
today = datetime.now().strftime("%Y-%m-%d")


class ImageAnalysisResult(BaseModel):
    """Stores structured data from the Identify Agent"""

    objects_detected: Dict[str, Dict] = Field(
        ..., description="Categorized detected objects"
    )
    text_presence: Dict = Field(..., description="Categorized detected text sources")
    extraction_plan: List[str] = Field(
        ..., description="List of specific elements to extract"
    )


class ExtractedImageData(BaseModel):
    """Stores structured data from the Extractor Agent"""

    extracted_data: Dict = Field(..., description="Final structured extraction output")


class ImageProcessingWorkflow(Workflow):
    """
    AI-Powered Image Processing Workflow:
    --------------------------------------------------------
    1. Analyzes an image and determines what information can be extracted.
    2. Extracts structured details based on the analysis or user instructions.
    3. Returns structured JSON for further use.
    """

    def analyze_image(self, image_path: Path, model) -> ImageAnalysisResult:
        """Analyze the image to determine extractable details."""
        logger.info("Running Identify Agent for image analysis...")

        # Identify Agent: Understands what can be extracted
        identifier_agent = Agent(
            model=model,
            description="An AI agent that analyzes an image and determines what details can be extracted.",
            response_model=ImageAnalysisResult,
        )

        prompt = """
        Analyze the provided image and determine **what can be extracted**.

        For examples, extracted plan for these type of use-cases can be:

        **For Charts & Graphs:**
           - Detect the **type of chart** (bar chart, pie chart, scatter plot, etc.).
           - Extract **X-axis and Y-axis labels**, data points if visible.
           - Provide a **summary of insights** from the chart.

        **For Traffic or Object Images (e.g., Cars, People, Roads):**
           - Count number of cars, people, bicycles, etc. in the image.
           - Identify vehicle colors and number plates.
           - Provide a scene understanding.

        **For Documents & Handwritten Text:**
           - Determine whether the text is printed or handwritten.
           - Extract headings, paragraphs, tables, bullet points, signatures.

        **For Shop, Mall, or Place Images:**
           - Identify place type (e.g., restaurant, mall, historical monument).
           - Detect shop name, location, and operating hours.

        **General Image Insights:**
           - Identify main subjects (e.g., nature, urban setting, people).
           - Provide structured extraction plan.

        Do not extract the data, only create an extraction plan.

        #### **Example JSON Output**
        {
            "objects_detected": {
                "vehicles": {
                    "count": 2,
                    "details_to_extract": "Extract number plate"
                },
                "road_signs": {
                    "count": 1,
                    "details_to_extract": "Extract text from sign"
                },
                "billboards": {
                    "count": 1,
                    "details_to_extract": "Extract advertisement text"
                },
                "people": {
                    "count": 1,
                    "details_to_extract": "Mention presence only (Do not extract identity)"
                }
            },
            "text_presence": {
                "detected": true,
                "text_sources": ["Signboard", "Billboard", "Vehicle Number Plate"]
            },
            "extraction_plan": [
                "Extract number plate from all vehicles",
                "Extract all readable text from signboards",
                "Extract advertisement text from billboards"
            ]
        }
        """

        response = identifier_agent.run(
            prompt, images=[Image(filepath=image_path)]
        ).content

        if isinstance(response, str):
            response = json.loads(response)

        logger.info(f"Response type: {type(response)}\nResponse: {response}")
        response = ImageAnalysisResult.model_validate(response)

        return response

    def extract_data(
        self, image_path: Path, image_extraction_plan, model
    ) -> ExtractedImageData:
        """Extract structured details based on Identify Agent’s plan or manual instructions."""
        logger.info("Running Extractor Agent for data extraction...")

        # Extractor Agent: Extracts details based on Identify Agent’s plan or user input
        extractor_agent = Agent(
            model=model,
            description=(
                "An AI agent that extracts structured information from images based on the extraction "
                "plan provided by the Identify Agent or user instructions."
            ),
            response_model=ExtractedImageData,
        )

        prompt = f"""
        ### Task: Extract Data Based on Plan

        **Image Extraction Plan Instructions:** {image_extraction_plan}

        Using this plan, extract structured data from the image.
        Return results in structured JSON format.

        **Example Output:**
        {{
        "extracted_data": {{
            "vehicles": {{
            "count": 3,
                        "colors": ["Red", "Blue", "Black"],
                        "number_plates": ["AB1234", "XY5678", "CD9999"]
                    }},
                    "signboards": {{
            "text": ["STOP", "Speed Limit 60 km/h"]
                    }}
            "image_analysis": "the image is of a busy street picture near traffic signal with cars, some warnings are also mentioned there in the road. The cars are waiting at a red light.",
            "significance": "this road is an iconic place where
            }}
        }}

        Analysis of each image is must and significance field should be returned if you know anything special about the place or objects in the image.
        """

        response = extractor_agent.run(
            prompt, images=[Image(filepath=image_path)]
        ).content

        if isinstance(response, str):
            response = json.loads(response)

        logger.info(f"Response type: {type(response)}\nResponse: {response}")
        response = ExtractedImageData.model_validate(response)

        return response

    def run(
        self,
        image_path,
        mode: str,
        model_choice: str,
        api_key: str,
        instruction: Optional[str] = None,
    ) -> RunResponse:
        """Main workflow to analyze and extract structured data from an image"""

        # Select AI Model
        if model_choice == "OpenAI":
            model = OpenAIChat(id="gpt-4o", api_key=api_key)
        else:
            model = Gemini(id="gemini-2.0-flash", api_key=api_key)

        if mode == "Auto":
            # Auto Mode: Extract everything possible
            analysis_result = self.analyze_image(image_path, model)
            extracted_data = self.extract_data(image_path, analysis_result, model)

        elif mode == "Manual":
            if not instruction:
                raise ValueError("Manual mode requires a user-provided instruction.")

            extracted_data = self.extract_data(image_path, instruction, model)

        elif mode == "Hybrid":
            if not instruction:
                raise ValueError("Hybrid mode requires additional user instructions.")

            # Hybrid Mode: Identify + User-Specified Data
            analysis_result = self.analyze_image(image_path, model)
            combined_instruction = (
                str(analysis_result)
                + "\n\nRequest from user to extract information:"
                + str(instruction)
            )
            logger.info(f"Combined_instruction: {combined_instruction}")
            extracted_data = self.extract_data(image_path, combined_instruction, model)

        else:
            raise ValueError(f"Invalid mode selected: {mode}")

        return RunResponse(content=extracted_data.model_dump())
