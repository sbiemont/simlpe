# simlpe

`simlpe` stands for `SI`mple `M`ulti `L`ayer `PE`rceptron

It a very simple implementation freely inspired from [this post](https://apprendre-le-deep-learning.com/coder-reseau-de-neurones-from-scratch/).

## Examples

Check the provided [examples](https://github.com/sbiemont/simlpe/blob/master/example/) of the `xor` function or the `mnist` database of handwritten digits

## Create a network

A network car be instanciated using `NewNetwork` and set the following parameters

* learning rate
* number of input neurons

Then, add layers using `AddLayer`.
A new layer is created using :

* a builder (like a `LinearBuilder` to initialize a linear layer)
* the number of neurons of this new layer
* the activation function (`Sigmoid`, `Htan`, `ReLU`...)

```go
// Build a basic network
net := mlp.NewNetwork(0.3, 2)                       // input layer (2 neurons)
net.AddLayer(mlp.LinearBuilder{}, 3, mlp.Sigmoid{}) // hidden layer (3 neurons)
net.AddLayer(mlp.LinearBuilder{}, 1, mlp.Sigmoid{}) // output layer (1 neuron)
```

## Train the network

### Set input, output reference data

In order to train the `xor` function, set the following data:

* Input data shall be a matrix of size `k` (number of features) x `n` (number of input neurons).
* Output data shall be a matrix of size `k` x `m` (number of output neurons)

```go
xData := [][]float64{
  {0, 0},
  {1, 0},
  {0, 1},
  {1, 1},
}
yData := [][]float64{{0}, {1}, {1}, {0}}
```

### Early stop processing

Fill optional condition of stop the training.
By default, only one epoch will be processed.
If one or more criteria are filled, the first to be true will stop the training.

```go
net.Stop.OnEpoch(10000)             // max epoch = 10 000
net.Stop.OnDuration(time.Second)    // max duration = 1 second
net.Stop.OnMeanSquaredError(0.0001) // min mean squared error is 1e-4
```

### Launch the training

Launch training using the `Train` function.
It returns the ending condition reached or an error if any.

```go
term, err := net.Train(ctx, xData, yData)
```

## Predict or check the network

Use function `Predict` data for each input neurons to produce data for each output neurons.

```go
net.Predict([]float64{0, 0}) // should be ~0
net.Predict([]float64{1, 0}) // should be ~1
net.Predict([]float64{0, 1}) // should be ~1
net.Predict([]float64{1, 1}) // should be ~0
```

## Import / export a network

Use the json marshaler to read or write a network.

```go
// Export the network
js, err := net.MarshalJSON()
// if err != nil ...

// Import the same network
net2 := mlp.Network{}
err2 := net2.UnmarshalJSON(js)
// if err2 != nil ...
```
