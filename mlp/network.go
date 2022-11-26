package mlp

// goemelpe

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

type Network struct {
	layers       []Layer     // List of layers
	inputs       [][]float64 // Memo input
	learningRate float64     // Learning rate
	neurons      []int       // Number of neurons at each layer

	Stop Termination // Ending conditions
}

// NewNetwork builds an empty network with a given number of inputs
func NewNetwork(learningRate float64, in int) Network {
	return Network{
		learningRate: learningRate,
		neurons:      []int{in},
	}
}

// in computes the number of inputs
func (net Network) in() int {
	return net.neurons[0]
}

// out computes the number of outputs
func (net Network) out() int {
	return net.neurons[len(net.neurons)-1]
}

// AddLayer pushes a new layer to the network
func (net *Network) AddLayer(bld LayerBuilder, neurons int, act Activator) {
	// Save neurons nb
	lastOut := net.out()
	net.neurons = append(net.neurons, neurons)

	// Add layers
	net.layers = append(
		net.layers,
		bld.New(lastOut, neurons),
		newActivatorLayer(act),
	)
}

// feedForward the input into the whole network
func (net *Network) feedForward(x []float64) []float64 {
	// Clear list of inputs
	net.inputs = make([][]float64, len(net.layers))

	// Set inputs for each layers
	io := x
	for i, layer := range net.layers {
		net.inputs[i] = io
		io = layer.FeedForward(io)
	}

	// Return last computed output
	return io
}

// backPropagation computes gradients for layers (backward)
func (net Network) backPropagation(yGrad []float64) {
	grad := yGrad
	for i := len(net.layers) - 1; i >= 0; i-- {
		grad = net.layers[i].BackPropagation(net.inputs[i], grad)
	}
}

// update all layers
func (net Network) update() {
	for _, layer := range net.layers {
		layer.Update(net.learningRate)
	}
}

// check that input and output matches
func (net Network) check(in, out []float64) error {
	switch {
	case len(net.layers) == 0:
		return fmt.Errorf("at least one layer expected")
	case len(in) != net.in():
		return fmt.Errorf("input data (%d) does not match input neurons (%d)", len(in), net.in())
	case len(out) != net.out():
		return fmt.Errorf("output data (%d) does not match output neurons (%d)", len(out), net.out())
	default:
		return nil
	}
}

// train the network on one epoch
// return the last computed output and the last expected output
func (net Network) trainOneEpoch(
	ctx context.Context, xData, yData [][]float64,
) ([]float64, []float64, error) {
	if len(xData) != len(yData) {
		return nil, nil, fmt.Errorf("input / output should have the same length")
	}

	var y []float64 // last y
	for i, xi := range xData {
		// Listen to context
		// select {
		// case <-ctx.Done():
		// 	return nil, nil, ctx.Err()
		// }

		yi := yData[i]
		if err := net.check(xi, yi); err != nil {
			return nil, nil, err
		}

		y = net.feedForward(xi)     // Compute y
		yGrad := newVector(len(yi)) // Gradient = y - target
		yGrad.iter(func(j int) {
			yGrad[j] = y[j] - yi[j]
		})

		net.backPropagation(yGrad) // Compute gradients
		net.update()               // Udpate weights
	}

	return y, yData[len(yData)-1], nil
}

func (net Network) Train(ctx context.Context, xData, yData [][]float64) (Termination, error) {
	start := time.Now()
	for epoch := 0; ; epoch++ { // epoch, no ending condition
		yLast, yDataLast, err := net.trainOneEpoch(ctx, xData, yData)
		if err != nil {
			return Termination{}, err
		}

		// Only compute what is checked
		current := Termination{}
		if net.Stop.epoch != nil {
			current.epoch = &epoch
		}
		if net.Stop.duration != nil {
			duration := time.Since(start)
			current.duration = &duration
		}
		if net.Stop.meanSquaredError != nil {
			mse := meanSquaredError(yLast, yDataLast)
			current.meanSquaredError = &mse
		}
		if net.Stop.hasReached(current) {
			mse := meanSquaredError(yLast, yDataLast)
			current.meanSquaredError = &mse
			return current, nil
		}
	}
}

// Predict takes a vector of inputs and computes a vector of outputs
func (net Network) Predict(x []float64) []float64 {
	return net.feedForward(x)
}

// MarshalJSON exports the whole network in a json format
func (net Network) MarshalJSON() ([]byte, error) {
	layers := make([]map[string]Layer, len(net.layers))
	for i, layer := range net.layers {
		layers[i] = map[string]Layer{
			layer.Type(): layer,
		}
	}

	type marshal struct {
		Rate    float64            `json:"learning-rate"`
		Neurons []int              `json:"neurons"`
		Layers  []map[string]Layer `json:"layers"`
	}
	return json.Marshal(marshal{
		Rate:    net.learningRate,
		Neurons: net.neurons,
		Layers:  layers,
	})
}

// UnmarshalJSON fills the network with a json content
func (net *Network) UnmarshalJSON(data []byte) error {
	// First unmarshal
	type unmarshal struct {
		Rate    float64                      `json:"learning-rate"`
		Neurons []int                        `json:"neurons"`
		Layers  []map[string]json.RawMessage `json:"layers"`
	}
	unm := unmarshal{}
	err := json.Unmarshal(data, &unm)
	if err != nil {
		return err
	}

	// Set known data
	net.learningRate = unm.Rate
	net.neurons = unm.Neurons
	net.layers = make([]Layer, len(unm.Layers))

	// Unmarshal layers
	for i, item := range unm.Layers {
		if len(item) != 1 {
			return fmt.Errorf("expected only one tag in layer")
		}

		var layer Layer
		for typ, data := range item { // only one item processed
			switch typ {
			case "linear":
				linear := Linear{}
				err = linear.UnmarshalJSON(data)
				layer = linear
			case "activator":
				activ := activatorLayer{}
				err = activ.UnmarshalJSON(data)
				layer = activ
			default:
				err = fmt.Errorf("unknown layer type %q", typ)
			}
			if err != nil {
				return err
			}
		}
		net.layers[i] = layer
	}

	return nil
}
