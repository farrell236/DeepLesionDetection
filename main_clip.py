import torch
import torchvision

from torch import nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForMaskedLM


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TextEncoder(nn.Module):
    def __init__(self, embed_dim, proj_dim):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            output_hidden_states=True)
        self.projection = nn.Linear(embed_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, texts):
        x = self.model(input_ids=texts['input_ids'], attention_mask=texts['attention_mask'])['hidden_states'][-1]
        x = x[:, 0, :]  # B, T[cls], E
        x = self.projection(x)
        x = self.layer_norm(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, base_model, embed_dim, proj_dim):
        super().__init__()

        self.model = nn.Sequential(
            base_model,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def forward(self, x):
        return self.model(x)


class CLIPModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.image_encoder = ImageEncoder(base_model=base_model, embed_dim=1280, proj_dim=256)
        self.text_encoder = TextEncoder(embed_dim=768, proj_dim=256)

        self.temperature = nn.Parameter(torch.ones([])*np.log(1/0.07))

    def forward(self, images, texts):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)

        return image_features, text_features, self.temperature.exp()


# base_model = torchvision.models.resnet152(pretrained=True)
# modules = list(base_model.children())[:-2]
# base_model = nn.Sequential(*modules)
base_model = torchvision.models.mobilenet_v2(pretrained=True).features
clip_model = CLIPModel(base_model)
clip_model.to(device)


a=1

from deeplesion import DeepLesion, get_transform


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    text_data = [item[1]['caption'] for item in batch]

    # Collate images normally (stacking)
    images_collated = torch.stack(images, dim=0)

    # Collate text data ['input_ids', 'token_type_ids', 'attention_mask']
    text_collated = {}
    keys = text_data[0].keys()
    for key in keys:
        text_collated[key] = torch.cat([d[key] for d in text_data], dim=0)

    return images_collated, text_collated


IMAGES_PATH = '/data/houbb/data/DeepLesion/Images_png'
ANNOTATIONS_TRAIN = 'data/cococaption_train_deeplesion.json'
ANNOTATIONS_VAL = 'data/cococaption_val_deeplesion.json'


# define training and validation data loaders
dataset = DeepLesion(root=IMAGES_PATH, annFile=ANNOTATIONS_TRAIN, transforms=get_transform())
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=1,
    collate_fn=custom_collate_fn
)
dataset_val = DeepLesion(root=IMAGES_PATH, annFile=ANNOTATIONS_VAL, transforms=get_transform())
data_loader_test = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=8,
    shuffle=False,
    collate_fn=custom_collate_fn
)


# optimizer and loss function
optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(100):  # Adjust the number of epochs as needed
    clip_model.train()
    total_loss = 0
    for images, texts in tqdm(data_loader):
        optimizer.zero_grad()
        images = images.to(device)
        texts = {k: v.to(device) for k, v in texts.items()}
        image_features, text_features, logit_scale = clip_model(images, texts)
        logits = logit_scale * image_features @ text_features.t()
        labels = torch.arange(len(images)).to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")
    # torch.save(clip_model.state_dict(), f'checkpoints_clip/clip_{epoch}.pth')

    # Evaluation
    clip_model.eval()
    eval_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, texts in tqdm(data_loader_test):
            images = images.to(device)
            texts = {k: v.to(device) for k, v in texts.items()}
            image_features, text_features, logit_scale = clip_model(images, texts)
            logits = logit_scale * image_features @ text_features.t()
            labels = torch.arange(len(images)).to(device)
            loss = criterion(logits, labels)
            eval_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Evaluation Loss: {eval_loss / len(data_loader_test):.4f}, Accuracy: {accuracy:.2f}%")
