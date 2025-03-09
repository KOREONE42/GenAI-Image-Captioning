import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Initialize global variables once to avoid repeated loading
PROCESSOR = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def fetch_webpage(url: str) -> BeautifulSoup:
    """
    Fetch and parse a webpage using requests and BeautifulSoup.
    
    Args:
        url (str): The URL of the webpage to fetch
        
    Returns:
        BeautifulSoup: Parsed HTML content
        
    Raises:
        requests.RequestException: If the webpage cannot be fetched
    """
    response = requests.get(url, timeout=10)  # Added timeout for better error handling
    response.raise_for_status()  # Raise an exception for bad status codes
    return BeautifulSoup(response.text, 'html.parser')

def process_image(img_url: str) -> tuple:
    """
    Download and process an image, returning the prepared image or None if failed.
    
    Args:
        img_url (str): URL of the image to process
        
    Returns:
        tuple: (PIL.Image or None, str or None) - Processed image and error message if any
    """
    try:
        response = requests.get(img_url, timeout=5)
        response.raise_for_status()
        
        # Convert image data to PIL Image and ensure RGB format
        raw_image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Skip small images (less than 20x20 pixels)
        if raw_image.size[0] * raw_image.size[1] < 400:
            return None, "Image too small"
            
        return raw_image, None
    except (requests.RequestException, Image.UnidentifiedImageError) as e:
        return None, str(e)

def generate_caption(image: Image.Image) -> str:
    """
    Generate a caption for the given image using the BLIP model.
    
    Args:
        image (PIL.Image): Image to generate caption for
        
    Returns:
        str: Generated caption
    """
    inputs = PROCESSOR(image, return_tensors="pt")
    out = MODEL.generate(**inputs, max_new_tokens=50)
    return PROCESSOR.decode(out[0], skip_special_tokens=True)

def main():
    """Main function to scrape images from Wikipedia and generate captions."""
    url = "https://en.wikipedia.org/wiki/IBM"
    
    try:
        # Fetch and parse webpage
        soup = fetch_webpage(url)
        img_elements = soup.find_all('img')
        
        # Use context manager for file handling
        with open("captions.txt", "w", encoding='utf-8') as caption_file:
            for img_element in img_elements:
                img_url = img_element.get('src')
                
                # Skip invalid or unwanted image URLs
                if not img_url or 'svg' in img_url.lower() or '1x1' in img_url:
                    continue
                    
                # Normalize URL
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif not img_url.startswith(('http://', 'https://')):
                    continue
                    
                # Process image and generate caption
                image, error = process_image(img_url)
                if image:
                    try:
                        caption = generate_caption(image)
                        caption_file.write(f"{img_url}: {caption}\n")
                    except Exception as e:
                        print(f"Error generating caption for {img_url}: {e}")
                elif error:
                    print(f"Error processing image {img_url}: {error}")
                    
    except requests.RequestException as e:
        print(f"Error fetching webpage: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
