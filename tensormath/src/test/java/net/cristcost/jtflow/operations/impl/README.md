# About the tests

Tests are generated from PyTorch with the following code snippets:

```
def data(tensor):
    if type(tensor) == torch.Tensor:
        return ", ".join(map(str, [x for x in tensor.flatten().detach().numpy()]))
    else:
        return ", ".join(map(str, [x for x in tensor]))


def shape(l):
    return ", ".join(map(str, [x for x in l]))


def nll(predictions, oneHotEncodedLabels):
    if type(predictions) != torch.Tensor:
        predictions = torch.tensor(predictions)
    if type(oneHotEncodedLabels) != torch.Tensor:
        oneHotEncodedLabels = torch.tensor(oneHotEncodedLabels)
    predictions = torch.clamp(predictions, min=1e-15)
    assert (
        max(oneHotEncodedLabels) == 1.0 and sum(oneHotEncodedLabels) == 1.0
    ), f"second operand looks not 1 hot encoded: {oneHotEncodedLabels}"
    target = oneHotEncodedLabels
    return -torch.sum(target * torch.log(predictions))


def printNllTest(predictions, oneHotEncodedLabels):
    result = nll(predictions, oneHotEncodedLabels)
    print(f"assertEquals({result.item()}, NegativeLogLikelihoodLoss.nll(data({data(predictions)}), data({data(oneHotEncodedLabels)})), DELTA);")




def printNllChainTest(predictions, oneHotEncodedLabels, gradient=1.0):
    if type(predictions) != torch.Tensor:
        predictions = torch.tensor(predictions, requires_grad=True)
    else:
        predictions = predictions.clone().detach().requires_grad_(True)

    oneHotEncodedLabels = torch.tensor(oneHotEncodedLabels, requires_grad=True)

    result = nll(predictions, oneHotEncodedLabels)
    result.backward(torch.tensor(gradient))

    print(
        f"assertArrayEquals(data({data(predictions.grad)}), NegativeLogLikelihoodLoss.predictionsGradient({gradient}, data({data(predictions)}), data({data(oneHotEncodedLabels)})), DELTA);"
    )



def printMseTest(a, b):
    if type(a) != torch.Tensor:
        a = torch.tensor(a)

    if type(b) != torch.Tensor:
        b = torch.tensor(b)

    result = F.mse_loss(a, b)
    print(f"assertEquals({result.item()}, MeanSquareError.mse(data({data(a)}), data({data(b)})), DELTA);")


def printMseChainTest(a, b, gradient=1.0):
    if type(a) != torch.Tensor:
        a = torch.tensor(a, requires_grad=True)
    else:
        a = a.clone().detach().requires_grad_(True)

    if type(b) != torch.Tensor:
        b = torch.tensor(b, requires_grad=True)
    else:
        b = b.clone().detach().requires_grad_(True)

    result = F.mse_loss(a, b)
    result.backward(torch.tensor(gradient))

    print(
        f"assertArrayEquals(data({data(a.grad)}), MeanSquareError.operandGradient({gradient}, data({data(a)}), data({data(b)})), DELTA);"
    )
    print(
        f"assertArrayEquals(data({data(b.grad)}), MeanSquareError.operandGradient({gradient}, data({data(b)}), data({data(a)})), DELTA);"
    )

def printCrossentropyTest(a, b):
    if type(a) != torch.Tensor:
        a = torch.tensor(a)

    if type(b) != torch.Tensor:
        b = torch.tensor(b)

    result = F.cross_entropy(a, b)
    print(f"assertEquals({result.item()}, CategoricalCrossentropy.cce(data({data(a)}), data({data(b)})), DELTA);")


def printCrossentropyChainTest(predictions, oneHotEncodedLabels, gradient=1.0):
    if type(predictions) != torch.Tensor:
        predictions = torch.tensor(predictions, requires_grad=True)
    else:
        predictions = predictions.clone().detach().requires_grad_(True)

    oneHotEncodedLabels = torch.tensor(oneHotEncodedLabels, requires_grad=True)

    result = F.cross_entropy(predictions, oneHotEncodedLabels)
    result.backward(torch.tensor(gradient))

    print(
        f"assertArrayEquals(data({data(predictions.grad)}), CategoricalCrossentropy.predictionsGradient({gradient}, data({data(predictions)}), data({data(oneHotEncodedLabels)})), DELTA);"
    )
```

Sample usage:

```
printNllTest([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
printNllChainTest([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], gradient=5.0)

printMseTest([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
printMseChainTest([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], gradient=5.0)

printCrossentropyTest([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
printCrossentropyChainTest([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], gradient=5.0)

```




