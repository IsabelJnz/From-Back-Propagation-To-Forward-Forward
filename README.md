# From-Back-Propagation-To-Forward-Forward
This paper presents an empirical investigation comparing Back Propagation and Forward-Forward, two algorithms for optimizing neural networks, using the MNIST data set. The study compared the algorithms based on their accuracy and found that the Forward-Forward algorithm outperforms Back Propagation. The results suggest a paradigm shift from Back Propagation to Forward-Forward in the optimization of neural networks, offering new insights into the field. The study highlights the superiority of the Forward-Forward algorithm and its potential to be adopted as the preferred optimization method.

![status](https://img.shields.io/badge/Status-investigation-green)


1. Import the required libraries: **`torch`**, **`numpy`**, **`pandas`**, **`time`**, and **`tqdm`**.

2. Load the **`MNIST`** dataset from **`torchvision.datasets`**.

3. Preprocess the data using the **`Compose`**, **`ToTensor`**, **`Normalize`**, and **`Lambda`** classes from **`torchvision.transforms`**.

4. Load and batch the data using the **`DataLoader`** class from **`torch.utils.data`**.

5. Define the **`device`** variable to specify whether the computations will use the GPU or the CPU.

6. Define the **`overlay_y_on_x`** function to create an overlay of the input tensor **`x`** and the values specified by tensor **`y`**.

7. Define the **`Net`** class to encapsulate the functionality of a neural network with an array of integers **`dims`** as input.

8. Define the **`predict`** method of the **`Net`** class to predict the label with the maximum goodness for each input data.

9. Define the **`train`** method of the **`Net`** class to train the network on positive and negative examples.

10. Define the **`Layer`** class to create a custom layer that can learn a non-linear feature representation of the input data by inheriting the properties of the **`torch.nn.Linear`** layer and also adding additional functionality such as training the layer on positive and negative samples, and calculating the activation of the input data.

11. Define the **`FBNet`** class which extends the **`torch.nn.Module`** class from the PyTorch library, initializes various attributes of the network such as the number of hidden layers, the input and output sizes, the learning rate, and the loss function, optimizer, etc. and applies a forward pass through the network using the **`torch.nn.functional.relu`** activation function.

Ergebnis
