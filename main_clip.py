import torch
import torchvision

from torch import nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForMaskedLM


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            output_hidden_states=True)

    def forward(self, texts):
        x = self.model(input_ids=texts['input_ids'], attention_mask=texts['attention_mask'])['hidden_states'][-1]
        x = x[:, 0, :]  # B, T[cls], E
        return x


class ImageEncoder(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.model = nn.Sequential(
            base_model,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.model(x)


class CLIPModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.image_encoder = ImageEncoder(base_model)
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(
            embedding_dim=2048,
            projection_dim=256,
            dropout=0.1
        )
        self.text_projection = ProjectionHead(
            embedding_dim=768,
            projection_dim=256,
            dropout=0.1
        )

        self.temperature = 1.  # nn.Parameter(torch.ones([])*np.log(1/0.07))

    def forward(self, images, texts):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return image_embeddings, text_embeddings, self.temperature


a=1




base_model = torchvision.models.resnet152(pretrained=True)
modules = list(base_model.children())[:-2]
base_model = nn.Sequential(*modules, nn.AdaptiveAvgPool2d((1, 1)))
# base_model = torchvision.models.mobilenet_v2(pretrained=True).features
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


TOKENIZER = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'  # JointLearning_v2
IMAGES_PATH = '/data/houbb/data/DeepLesion/Images_png'
ANNOTATIONS_TRAIN = 'data/cococaption_train_deeplesion.json'
ANNOTATIONS_VAL = 'data/cococaption_val_deeplesion.json'


# define training and validation data loaders
dataset = DeepLesion(root=IMAGES_PATH, annFile=ANNOTATIONS_TRAIN, transforms=get_transform(), tokenizer=TOKENIZER)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=1,
    collate_fn=custom_collate_fn
)
dataset_val = DeepLesion(root=IMAGES_PATH, annFile=ANNOTATIONS_VAL, transforms=get_transform(), tokenizer=TOKENIZER)
data_loader_test = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=8,
    shuffle=False,
    collate_fn=custom_collate_fn
)


# optimizer and loss function
optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-5)
log_softmax = nn.LogSoftmax(dim=-1)

def compute_losses(image_embeddings, text_embeddings, temperature):
    logits = (text_embeddings @ image_embeddings.T) / temperature
    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax(
        (images_similarity + texts_similarity) / 2 * temperature, dim=-1
    )
    images_loss = (-targets.T * log_softmax(logits.T)).sum(1)
    texts_loss = (-targets * log_softmax(logits)).sum(1)
    return (images_loss + texts_loss) / 2.0


for epoch in range(100):  # Adjust the number of epochs as needed
    clip_model.train()
    total_loss = 0
    t = tqdm(data_loader)
    for images, texts in tqdm(data_loader):
        optimizer.zero_grad()
        images = images.to(device)
        texts = {k: v.to(device) for k, v in texts.items()}
        image_embeddings, text_embeddings, temperature = clip_model(images, texts)
        loss = compute_losses(image_embeddings, text_embeddings, temperature).mean()
        loss.backward()
        optimizer.step()
        t.set_description(f"Train Loss = {loss.item():.2f}")
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")
    # torch.save(clip_model.state_dict(), f'checkpoints_clip/clip_{epoch}.pth')

    # Evaluation
    clip_model.eval()
    eval_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        t = tqdm(data_loader)
        for images, texts in t:
            images = images.to(device)
            texts = {k: v.to(device) for k, v in texts.items()}
            image_embeddings, text_embeddings, temperature = clip_model(images, texts)
            loss = compute_losses(image_embeddings, text_embeddings, temperature).mean()
            t.set_description(f"Eval Loss = {loss.item():.2f}")
            eval_loss += loss.item()

            # _, predicted = torch.max(logits, 1)
            # correct += (predicted == labels).sum().item()
            # total += labels.size(0)

        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Evaluation Loss: {eval_loss / len(data_loader_test):.4f}, Accuracy: {accuracy:.2f}%")
