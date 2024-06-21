# "How machines learn" for Java developers 
## AKA implementation from scratch of automatic differentiation and gradient descent in Java to understand how it works

This repo is the code I've written to understand how Tensorflow/PyTorch models can "learn" using automatic differentiation and gradient descent.

I'm considering writing an article on Medium.com or some other platform like [mokabyte.it](https://www.mokabyte.it/autore/cristiano-costantini/) to talk about it.

## "Tensor" Branch
While the main branch focuses on SGD with scalar values, I'm exploring how to change the code to support N Dimensional arrays in the [this](https://github.com/cristcost/java-gradient-descent/tree/tensor) branch.

### Current status:
* implemented support for Tensor operations and implemented MNIST dataset (handwritten digits) training loop in pure Java.
* ongoing refactoring and re-modularization of the code.

**Current Levelized Build**:
* Level 1: api, math, file 
* Level 2: dataset, optimizers, tensor, tensormath
* Level 3: builder
* Level 4: core
* Level 5: debug
* Level 6: examples
* Orphan: extras


*Note:* Model learning perform and learn at equivalent rate than the following PyTorch code. Interestingly, a key factor that boosted learning performance was to use a Kaiming Uniform Initialization for the model parameters (until that, this Java model was converging 10 time slower than PyTorch one).


```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

class HandwrittenDigitParserModel(nn.Module):
    def __init__(self):
        super(HandwrittenDigitParserModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
samples_count = list(train_dataset.targets.size())[0]
tot_rounds_per_epoch = int(samples_count/64)

handwrittenDigitParserModel = HandwrittenDigitParserModel()

optimizer = optim.SGD(handwrittenDigitParserModel.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9, eps=1e-08)

# Training loop
num_epochs = 25
start_time = time.time()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_samples = 0
          
    print("")
    print("### Epoch ", epoch, " ####")
    for round, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        probabilities = handwrittenDigitParserModel(inputs)

        true_labels_one_hot = F.one_hot(labels, num_classes=10).float()
        
        # loss = F.mse_loss(true_labels_one_hot,probabilities)
        loss = F.cross_entropy(probabilities, true_labels_one_hot)


        loss.backward()
        optimizer.step()

        predicted = torch.argmax(probabilities, dim=1)
        correct = (predicted == labels).sum().item()

        epoch_correct += correct
        epoch_samples += labels.size(0)
        epoch_loss += loss.item()

        if round == 10 or round == 50 or round == 100 or round == 500 or round == tot_rounds_per_epoch-1:
            print("   === Epoch ", epoch, " round", round, "===")
            print(f"         Execution time: {int(time.time() - start_time)} seconds")
            print("             Loss value: ", loss.item())
            print("    Correct predictions: ", correct, " out of ", labels.size(0))
    print("         Epoch Loss value: ", epoch_loss/tot_rounds_per_epoch)
    print("Epoch Correct predictions: ", epoch_correct, " out of ", epoch_samples)
    if(epoch_correct / epoch_samples > 0.90):
        break

print("Finished Training")
```

**Next**: 
* Code in this branch to be refactored and styled for presentation.
* Add additional unit tests on the operations to compare outout with Tensorflow and PyTorch