from fb_net import FBNet
from ff_net import FFNet

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

from tqdm import tqdm

import pandas as pd

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def get_model(ff_net_bool, n_layers, n_input, n_output):
    if ff_net_bool:
        model = FFNet(n_layers=n_layers, n_input=n_input, n_output=n_output)
    else:
        model = FBNet(n_layers=n_layers, n_input=n_input, n_output=n_output)
    return model

def train_and_get_eval(model, epochs=1):
    train_loader, test_loader = MNIST_loaders()
    model = model.to("cuda")
    
    # import IPython;IPython.embed();exit(1)
    with torch.autocast("cuda", dtype=torch.float32):
        for epoch in range(epochs):
            for x, y in train_loader:
                x.to("cuda")
                y.to("cuda")
                loss = model.train(x, y, 4)
    
#eval on test
    with torch.autocast("cuda", dtype=torch.float32):
        for x, y in test_loader:
            x.to("cuda")
            y.to("cuda")
            test_loss = model.eval(x, y)

    if type(loss) == torch.Tensor:
        loss = loss.mean().item()
    if type(test_loss) == torch.Tensor:
        test_loss = test_loss.mean().item()

    return loss, test_loss


if __name__ == "__main__":
    ff_train_losses, ff_test_losses = [], []
    fb_train_losses, fb_test_losses = [], []

    for _ in tqdm(range(10)):
        n_layers = 2
        n_input = 784
        n_output = 10

        # print("FBNet")
        model = get_model(False, n_layers, n_input, n_output)
        fb_train_loss, fb_test_loss = train_and_get_eval(model)
        ff_train_losses.append(fb_train_loss)
        ff_test_losses.append(fb_test_loss)

        # print("FFNet")
        model = get_model(True, n_layers, n_input, n_output)
        ff_train_loss, ff_test_loss = train_and_get_eval(model)
        fb_train_losses.append(ff_train_loss)
        fb_test_losses.append(ff_test_loss)
    
    # to pandas csv
    df = pd.DataFrame({"ff_train_losses": ff_train_losses, "ff_test_losses": ff_test_losses, "fb_train_losses": fb_train_losses, "fb_test_losses": fb_test_losses})
    df.to_csv("losses.csv")


