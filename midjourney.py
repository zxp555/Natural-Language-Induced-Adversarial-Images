import requests
import time
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from crop_image import *
import torch.nn as nn
from torchvision.models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet101(pretrained=True)
# model.eval()
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model = resnet101(pretrained=True)
model.eval()


# model.fc = nn.Sequential(
#     nn.Linear(2048, 512),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(512, 10)
# )
# model = vit_l_16()
# num_classes = 7
# model.heads = nn.Linear(1024, num_classes)
# model.load_state_dict(torch.load('model/resnet101_animal10_clean.pth'))
# num_classes = 7  
# model.heads = nn.Linear(1024, num_classes)
# model.load_state_dict(torch.load('model/resnet101_animal10_clean.pth'))
# model.eval()  

def send(prompt):
    url = 'http://localhost:8080/mj/submit/imagine'
    data = {
        "base64": "",
        "notifyHook": "",
        "prompt": prompt,
        "state": ""
    }

    try:
        response = requests.post(url, json=data)

        if response.status_code == 200:
            print("Request successful!")
            print("Response:")
            print(response.json())  # If the response is in JSON format
            return response.json()['result']
        else:
            print(f"Request failed with status code: {response.status_code}")
            print("Response:")
            print(response.text)
            return ""

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

def get(task_id):
    # Construct the complete URL for the API call
    url = f"http://localhost:8080/mj/task/{task_id}/fetch"
    try:
        # Sending GET request to the API
        response = requests.get(url)
        # print(response.json())
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # The response content will be in JSON format if the API returns JSON data
            json_response = response.json()
            # Process the JSON response as needed
            if json_response['status'] == 'FAILURE':
                return "failure"
            elif json_response['status'] == 'SUCCESS':
                return json_response['imageUrl'] 
            else :
                return ""
        else:
            print(f"Failed to call API. Status code: {response.status_code}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while calling the API: {e}")

def download_image(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for any errors in the response

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            f.write(response.content)
            
        print(f"Image downloaded successfully and saved at: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while downloading the image: {e}")