package mlp

import (
	"math"
)

type Activator interface {
	Activ(float64) float64
	Deriv(float64) float64
	String() string
}

// Sigmoid function
type Sigmoid struct{}

// Activ sigmoid = 1/(1+exp(-x))
func (s Sigmoid) Activ(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Deriv sigmoid = sig(x)*(1-sig(x))
func (s Sigmoid) Deriv(x float64) float64 {
	sig := s.Activ(x)
	return sig * (1 - sig)
}

// String converts to constants
func (s Sigmoid) String() string {
	return "sigmoid"
}

// Htan hyperbolic tangent
type Htan struct{}

// Active hyperbolic tan = (exp(2x)-1) / (exp(2x)+1)
func (h Htan) Activ(x float64) float64 {
	exp := math.Exp(2 * x)
	return (exp - 1) / (exp + 1)
}

// Deriv hyperbolic tan = 1 - htan(x)²
func (h Htan) Deriv(x float64) float64 {
	tan := h.Activ(x)
	return 1 - tan*tan
}

// String converts to constants
func (h Htan) String() string {
	return "htan"
}

// ReLU stands for "Rectified Linear Unit" (non linear activation function)
type ReLU struct{}

// Activ ReLU relu = max(0,x)
func (r ReLU) Activ(x float64) float64 {
	return math.Max(0.0, x)
}

// Deriv ReLU relu = 1 if x>0 ; 0 otherwise
func (r ReLU) Deriv(x float64) float64 {
	if x > 0 {
		return 1.0
	}
	return 0.0
}

// String converts to constants
func (r ReLU) String() string {
	return "relu"
}

