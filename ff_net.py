import torch
import numpy as np
from tqdm import tqdm 


def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y.long()] = x.max()
    return x_

class FFNet(torch.nn.Module):
    def __init__(self, n_input, n_output, n_layers=2):
        super().__init__()
        self.layers = []
        dims = [784, 100, 10]
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).cuda()]


    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x, y, epochs=None):
        x_pos = overlay_y_on_x(x, y)
        
        rnd = torch.randperm(x.size(0))
        x_neg = overlay_y_on_x(x, y[rnd])
        h_pos, h_neg = x_pos, x_neg
        losses = []
        for i, layer in enumerate(self.layers):
            # print('training layer', i, '...')
            h_pos, h_neg, loss = layer.train(h_pos, h_neg)
            losses += [loss.item()]

        return losses
    
    def eval(self, x, y_true):
        for layer in self.layers:
            x = layer(x)
        y_pred = x
        y_pred, y_true = y_pred.to("cuda"), y_true.to("cuda")
        # log loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(y_pred, y_true)
        return loss


class Layer(torch.nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 100

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(
            x_direction.cuda(), self.weight.T
            ) +
            self.bias.unsqueeze(0))


    def train(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach(), loss
