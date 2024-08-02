import argparse
import optuna
import albumentations as A
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
import pandas as pd
from dataset import ISIC_Dataset
from util import *
from torch.utils.data import DataLoader
from classifier import EfficientNetBinaryClassifierLightning
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.loggers import TensorBoardLogger
import torch.nn as nn
import pytorch_lightning as pl


BATCH_SIZE = 32
 
df_main = pd.read_csv('train-metadata.csv')
df_add = pd.read_csv('metadata.csv')
df_add = df_add[~df_add['benign_malignant'].isna()]
df_add['benign_malignant'].value_counts()
df_add = df_add[df_add['benign_malignant']!= 'indeterminate']
df_add['target'] = df_add['benign_malignant'].apply(func = lambda x : 1 if 'malignant' in x else 0)
common_cols = ['isic_id', 'target']
data = pd.concat([df_add[common_cols], df_main[common_cols]], axis = 'rows')
data_mini = pd.concat((data[data['target'] == 1], data[data['target'] == 0].iloc[:1000])).reset_index().sample(frac = 1)
#data_test = pd.read_csv('test-metadata.csv')


class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def objective(trial: optuna.trial.Trial) -> float:
    # We optimize the number of layers, hidden units in each layer and dropouts.
    
    lr = trial.suggest_float("lr", 1e-5, 1e-1)
    optim_choice = trial.suggest_categorical("optim_choice", ['Adam', 'SGD'])
    
    data_path = './train-image/image/'
    transform = A.Compose([
        A.Resize(100, 100),                   
        A.Rotate(limit=30, p=0.5)             
    ])
    
    batch_size = BATCH_SIZE
    
    # Initialize the EfficientNet model (Assuming using EfficientNet-B0 for example)
    efficientnet = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
    num_features = efficientnet.classifier[1].in_features  # Assuming '_fc' is the final fully connected layer
    train_df, val_df = get_train_test_dfs(dataframe=data_mini, test_size = 0.3)
            
    # Replace the final layer with an identity layer
    efficientnet.classifier = nn.Identity()
    train_dataset = ISIC_Dataset(data_path, train_df, transform=transform)
    val_dataset = ISIC_Dataset(data_path, val_df, transform=transform)
    train_sampler = get_sampler(train_df)
    val_sampler = get_sampler(val_df)
    # Initialize DataModule and Model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = train_sampler, pin_memory = True, num_workers = 2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler = val_sampler, pin_memory = True, num_workers = 2)
    
    #data_module = ISICDataModule(data_path, dataframe, batch_size=32, transform=transform)
    model = EfficientNetBinaryClassifierLightning(efficientnet, val_loader, num_features, optim_choice, lr)
    logger = TensorBoardLogger("tb_logs", name="my_model")
    # Initialize Trainer
    trainer = pl.Trainer(logger = logger, max_epochs = 1, accelerator = 'auto', callbacks=[OptunaPruning(trial, monitor="val_acc")])
    
    # Train and validate the model
    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["val_acc"].item()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIC competition experiments.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=3, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))