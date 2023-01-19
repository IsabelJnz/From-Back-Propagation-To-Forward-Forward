import torch


class FBNet(torch.nn.Module):
    """
    a fully connected neural network trained with forward-backward with a n_layers hidden layer"""
    def __init__(self, n_layers=2, n_input=10, n_output=4, lr=0.001):
        super().__init__()
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_input, n_input).to("cuda") for i in range(n_layers)])
        self.out = torch.nn.Linear(n_input, n_output)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = x.to("cuda")
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.relu(x)
        x = self.out(x)
        return x
    
    def train(self, x, y, epochs):
        for _ in range(epochs):
            self.optimizer.zero_grad()
            output = self.forward(x)
            y = y.to("cuda")
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

        return loss.item()
    
    def eval(self, x, y):
        output = self.forward(x)
        y = y.to("cuda")
        loss = self.criterion(output, y)
        return loss.item()