"""Define your architecture here."""
import torch
from models import SimpleNet
import torch.nn as nn
from torchvision.models import googlenet, resnet50, resnet18
# from trainer import write_output
import torch.optim as optim
from trainer import LoggingParameters, Trainer
from utils import load_dataset, get_nof_params

def get_nof_params(model: nn.Module) -> int:
    """Return the number of trainable model parameters.

    Args:
        model: nn.Module.

    Returns:
        The number of model parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def bonus_train():
    train_dataset = load_dataset('fakes_dataset', 'train')
    validation_dataset = load_dataset('fakes_dataset', 'val')

    model = build_resnet_backbone()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trainer = Trainer(model=model, optimizer=optimizer,
                      criterion=nn.CrossEntropyLoss(), batch_size=16,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      test_dataset=validation_dataset)

    optimizer_params = optimizer.param_groups[0].copy()
    # remove the parameter values from the optimizer parameters for a cleaner
    # log
    del optimizer_params['params']

    # Training Logging Parameters
    logging_parameters = LoggingParameters(model_name='bonus_model',
                                           dataset_name='fakes_dataset',
                                           optimizer_name='SGD',
                                           optimizer_params=optimizer_params,)

    trainer.run(epochs=10, logging_parameters=logging_parameters)

def build_resnet_backbone():
    # initialize your model:
    model = googlenet(pretrained=True)
    print(f'googlenet params before change: {get_nof_params(model)}')
    model.fc = nn.Linear(model.fc.in_features, 5)
    print(f'googlenet params after change: {get_nof_params(model)}')

    return model

def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = build_resnet_backbone()

    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model


if __name__ == "__main__":
    bonus_train()

# python train_main.py -d fakes_dataset -m bonus_model --lr 0.01 -b 16 -e 3 -o SGD
# python plot_accuracy_and_loss.py -m bonus_model -j out/fakes_dataset_bonus_model_SGD.json -d fakes_dataset
# python numerical_analysis.py -m bonus_model -cpp checkpoints/bonus_model.pt -d fakes_dataset

