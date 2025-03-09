import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

def load_image_captioning_model():
    """
    Load and initialize the BLIP image captioning model and processor.
    
    Returns:
        tuple: (processor, model) - The loaded processor and model objects
    """
    # Load pretrained processor and model from Hugging Face
    # Using BLIP (Bootstrapping Language-Image Pre-training) model
    # Source: https://huggingface.co/Salesforce/blip-image-captioning-base
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def process_image(img_path):
    """
    Load and preprocess an image for captioning.
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        PIL.Image: Processed RGB image
    """
    try:
        # Open and convert image to RGB format
        image = Image.open(img_path).convert('RGB')
        return image
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found")
        return None
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def generate_caption(processor, model, image, prompt="the image of"):
    """
    Generate a caption for the given image using the BLIP model.
    
    Args:
        processor: Pretrained BLIP processor
        model: Pretrained BLIP model
        image: PIL Image object in RGB format
        prompt (str): Initial text prompt for caption generation
        
    Returns:
        str: Generated caption or None if generation fails
    """
    if image is None:
        return None
        
    try:
        # Prepare inputs for the model
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        
        # Generate caption with a maximum length of 50 tokens
        outputs = model.generate(**inputs, max_length=50)
        
        # Decode the generated tokens into readable text
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error generating caption: {str(e)}")
        return None

def main():
    """Main function to run the image captioning process."""
    # Define image path
    img_path = "king.jpg"
    
    # Load model and processor
    processor, model = load_image_captioning_model()
    
    # Process the image
    image = process_image(img_path)
    
    if image:
        # Generate and print caption
        caption = generate_caption(processor, model, image)
        if caption:
            print(f"Generated caption: {caption}")

if __name__ == "__main__":
    main()
