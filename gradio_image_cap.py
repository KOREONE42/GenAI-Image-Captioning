import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load models globally to avoid reloading on each function call
# Using lazy initialization with a singleton pattern
class ModelSingleton:
    """Singleton class to manage model loading and access."""
    _processor = None
    _model = None
    
    @classmethod
    def get_processor(cls):
        """Get or initialize the processor."""
        if cls._processor is None:
            cls._processor = AutoProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir="./model_cache"  # Add caching for faster subsequent loads
            )
        return cls._processor
    
    @classmethod
    def get_model(cls):
        """Get or initialize the model."""
        if cls._model is None:
            cls._model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir="./model_cache"  # Add caching for faster subsequent loads
            )
        return cls._model

def caption_image(input_image: np.ndarray) -> str:
    """
    Generate a caption for the provided image.
    
    Args:
        input_image (np.ndarray): Input image as a numpy array
        
    Returns:
        str: Generated caption for the image
        
    Raises:
        ValueError: If input_image is None or invalid
    """
    # Input validation
    if input_image is None:
        return "Error: No image provided"
    
    try:
        # Convert numpy array to PIL Image and ensure RGB format
        raw_image = Image.fromarray(input_image.astype('uint8')).convert('RGB')
        
        # Get processor and model instances
        processor = ModelSingleton.get_processor()
        model = ModelSingleton.get_model()
        
        # Process image with error handling
        inputs = processor(raw_image, return_tensors="pt")
        
        # Generate caption with optimized parameters
        out = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,  # Add beam search for better quality
            early_stopping=True  # Stop when all beams reach EOS
        )
        
        # Decode the output
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        return caption.capitalize()  # Capitalize first letter for better readability
        
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Create Gradio interface with improved configuration
interface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(
        type="numpy",
        label="Upload an Image"
    ),
    outputs=gr.Textbox(
        label="Generated Caption",
        placeholder="Caption will appear here"
    ),
    title="Image Captioning Tool",
    description="Upload an image to generate a descriptive caption using BLIP model from Salesforce.",
    theme=gr.themes.Soft(),  # Add a modern theme
    examples=[  # Add example images
        ["examples/dog.jpg"],
        ["examples/landscape.jpg"]
    ],
    allow_flagging="never"  # Disable flagging for simplicity
)

def main():
    """Launch the Gradio interface."""
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False  # Set to True for public URL
    )

if __name__ == "__main__":
    main()
