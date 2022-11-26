package mlp

import "math/rand"

// meanSquaredError or mse
// mse = ∑ (x-y)² / n
func meanSquaredError(v1, v2 vector) float64 {
	var sum float64
	v1.iter(func(i int) {
		vi := v1[i] - v2[i]
		sum += vi * vi
	})
	return sum / float64(len(v1))
}

// random float using standard deviation and mean
func normRandom(stdDev, mean float64) float64 {
	return rand.NormFloat64()*stdDev + mean
}
