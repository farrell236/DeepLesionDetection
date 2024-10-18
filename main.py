import os

import torch
import utils

from engine import train_one_epoch, evaluate
from deeplesion import DeepLesion, get_transform
# from model import CustomFasterRCNN
from model import JointLearning_v2

########################################################################################################################
# Weights and Biases Logging
########################################################################################################################

# import wandb

# Ensure the W&B API key is set
# if not os.getenv('WANDB_API_KEY'):
#     raise ValueError("WANDB_API_KEY environment variable not set.")

config = {
    # 'backbone': 'fasterrcnn_resnet50_fpn',
    'num_epochs': 10,
    'batch_size': 32,
    'num_workers': 4,
    'shuffle': True,
    'optimizer': 'Adam',
    'learning_rate': 0.0001,
    # 'momentum': 0.9,
    # 'weight_decay': 0.0001,
    'scheduler': 'StepLR',
    'step_size': 3,
    'gamma': 0.1,
    'model_save_folder': 'jointmodel',
    'log_interval': 10,
}

# wandb.login()
# wandb.init(project='JointLearning', entity="farrell236", config=config, name=f'fasterrcnn')


########################################################################################################################
# Train on GPU or CPU
########################################################################################################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


########################################################################################################################
# DeepLesion Dataloader
########################################################################################################################

# JSON with captions
# TOKENIZER = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'  # JointLearning_v1
TOKENIZER = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'  # JointLearning_v2
IMAGES_PATH = '/data/houbb/data/DeepLesion/Images_png'
ANNOTATIONS_TRAIN = 'data/cococaption_train_deeplesion.json'
ANNOTATIONS_VAL = 'data/cococaption_val_deeplesion.json'

# JSON without captions (https://github.com/urmagicsmine/MVP-Net/blob/master/data/DeepLesion/annotation)
# TOKENIZER = None
# IMAGES_PATH = '/data/houbb/data/DeepLesion/Images_png'
# ANNOTATIONS_TRAIN = '/data/houbb/data/DeepLesion/annotation/deeplesion_train.json'
# ANNOTATIONS_VAL = '/data/houbb/data/DeepLesion/annotation/deeplesion_val.json'

dataset = DeepLesion(root=IMAGES_PATH, annFile=ANNOTATIONS_TRAIN, transforms=get_transform(), tokenizer=TOKENIZER)
dataset_val = DeepLesion(root=IMAGES_PATH, annFile=ANNOTATIONS_VAL, transforms=get_transform(), tokenizer=TOKENIZER)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config['batch_size'],
    shuffle=config['shuffle'],
    num_workers=config['num_workers'],
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)


########################################################################################################################
# Model
########################################################################################################################

# model = CustomFasterRCNN(num_classes=2)
model = JointLearning_v2(num_classes=2)
model.to(device)


########################################################################################################################
# Optimizer and Logging
########################################################################################################################

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(
    params,
    lr=config['learning_rate'],
    # momentum=config['momentum'],
    # weight_decay=config['weight_decay']
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config['step_size'],
    gamma=config['gamma']
)


########################################################################################################################
# Training Loop
########################################################################################################################

metrics_names = [
    'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large'
]

# Start training
for epoch in range(config['num_epochs']):
    # train for one epoch, printing every 10 iterations
    log_train = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    log_train = {key: meter.value for key, meter in log_train.meters.items()}
    # save model
    # torch.save(model.state_dict(), f'{config["model_save_folder"]}/jointmodel_clip_{epoch}.pth')
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    log_eval = evaluate(model, data_loader_test, device=device)
    bbox_stats = log_eval.coco_eval['bbox'].stats
    bbox_stats = {f'bbox {name}': bbox_stats[idx] for idx, name in enumerate(metrics_names)}
    # wandb.log({**log_train, **bbox_stats})
