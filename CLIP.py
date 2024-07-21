from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests

clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

def clip_ft(prompt_list, image):
    global clip_model, processor
    inputs = processor(prompt_list, image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim = 1)
    return probs

def main():
    url = "17_30_0.png_1.png"
    image = Image.open(url)
    prompt_list = ["a photo of a dog", "a photo of a cat"] 
    print(clip_ft(prompt_list, image))

if __name__ == '__main__':
    main()