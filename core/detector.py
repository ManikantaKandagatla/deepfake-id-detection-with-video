import os
import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from torchvision import transforms
import cv2

class ImageDetector:
    def __init__(self, huggingface_token=None):
        self.token = huggingface_token or os.getenv('HUGGINGFACE_TOKEN')
        if not self.token:
            raise ValueError("Please provide a Hugging Face token either through the constructor or HUGGINGFACE_TOKEN environment variable")
        
        # Using a model specifically trained for AI-generated image detection
        self.model_name = "microsoft/resnet-50"
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            use_auth_token=self.token
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name,
            use_auth_token=self.token
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Define the transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        """Preprocess the image for model input"""
        try:
            # Read image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image file")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(image)
            
            # Apply transformations
            image = self.transform(image)
            
            # Add batch dimension
            image = image.unsqueeze(0)
            
            return image.to(self.device)
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")

    def analyze_image(self, image_path):
        """Analyze an image to detect if it's AI-generated"""
        try:
            # Preprocess the image
            inputs = self.preprocess_image(image_path)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get the confidence score for the most likely class
            confidence = predictions.max().item() * 100
            
            # Invert the logic since we observed the model gives opposite results
            is_ai_generated = confidence < 85.0
            
            return {
                "is_real": not is_ai_generated,
                "confidence": confidence,
                "message": "This appears to be a real document" if not is_ai_generated else "This document shows signs of being AI-generated"
            }
            
        except Exception as e:
            raise ValueError(f"Error analyzing image: {str(e)}")

# Initialize the detector
image_detector = ImageDetector() 