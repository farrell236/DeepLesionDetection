import os
import json
import torch
import torchvision

from PIL import Image
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer

from model import CLIPModel


# Load the JSON file for multiple-choice evaluation
file_path = 'data/test_files/lesion_qa.json'
with open(file_path, 'r') as f:
    test_data = json.load(f)

# Device setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize the CLIP Model (same as when it was fine-tuned)
base_model = torchvision.models.resnet152(pretrained=True)
modules = list(base_model.children())[:-2]
base_model = nn.Sequential(*modules, nn.AdaptiveAvgPool2d((1, 1)))
clip_model = CLIPModel(base_model)
clip_model.to(device)

# Load the fine-tuned weights
clip_model.load_state_dict(torch.load('checkpoints/clip/clip_63.pth', map_location=device))

# Set model to evaluation mode
clip_model.eval()

# Preprocessing pipeline for images
preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
])

# Text preprocessing using tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

# Evaluation for multiple-choice question answering
correct_predictions = 0
total_predictions = 0
results = []

for item in tqdm(test_data):
    if item['q_type'] == 'img2txt':
        lesion_idx = item['lesion_idx']
        image_path = f"data/test_files/img2txtnobox/lesion_{lesion_idx}.png"
        # Image loading and preprocessing
        image = Image.open(image_path).convert('RGB')  # Convert to RGB if it’s grayscale
        image = preprocess(image).unsqueeze(0).to(device)  # Preprocess image and move to device

        #image = preprocess(Image.open(image_path)).convert('RGB').unsqueeze(0).to(device)  # Preprocess image

        question = item['question']
        choices = item['choices']  # Dict of choices
        answer = item['correct_choice']    # Correct answer
        labels = list(choices.values())  # Textual answer choices
        label_keys = list(choices.keys())  # Corresponding keys (A, B, C, etc.)

        logits = []
        for label in labels:
            # Concatenate question with each choice
            text_input = f"Nodule {label} in size"
            tokenized_input = tokenizer(text_input, return_tensors="pt", padding="max_length", max_length=256).to(device)

            # Forward pass through the model
            with torch.no_grad():
                image_embeddings, text_embeddings, temperature = clip_model(image, tokenized_input)
                logit = (text_embeddings @ image_embeddings.T) / temperature
                logits.append(logit.item())  # Append the logit score for each choice

        # Find the index of the highest logit score
        logits = torch.tensor(logits)
        predicted_idx = torch.argmax(logits).item()
        predicted_label = label_keys[predicted_idx]  # Predicted choice key (A, B, C, etc.)

        # Check if the prediction is correct
        if predicted_label == answer:
            correct_predictions += 1

        total_predictions += 1
        results.append({
            'lesion_idx': lesion_idx,
            'question': question,
            'predicted': predicted_label,
            'correct': answer,
            'choices': choices
        })

# Calculate accuracy
accuracy = 100 * correct_predictions / total_predictions
print(f"Evaluation Accuracy: {accuracy:.2f}%")

# Optionally, save the results for further analysis
# with open('multiple_choice_evaluation_results.json', 'w') as outfile:
#     json.dump(results, outfile, indent=4)