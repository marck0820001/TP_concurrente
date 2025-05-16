package regression

import (
	"math"
	"sync"
)

// LogisticRegressionConcurrent is a concurrent implementation of logistic regression
// using goroutines and mutex for parallel gradient computation
type LogisticRegressionConcurrent struct {
	Weights      []float64
	Bias         float64
	LearningRate float64
	Epochs       int
}

// NewLogisticRegressionConcurrent initializes the concurrent logistic regression model
func NewLogisticRegressionConcurrent(learningRate float64, epochs int) *LogisticRegressionConcurrent {
	return &LogisticRegressionConcurrent{
		LearningRate: learningRate,
		Epochs:       epochs,
	}
}

// TrainConcurrently performs gradient descent in parallel using goroutines
func (lr *LogisticRegressionConcurrent) TrainConcurrently(X [][]float64, Y []float64, numWorkers int) {
	n := len(X)
	d := len(X[0])
	lr.Weights = make([]float64, d)

	for epoch := 0; epoch < lr.Epochs; epoch++ {
		gradW := make([]float64, d)
		gradB := 0.0

		var wg sync.WaitGroup
		mutex := &sync.Mutex{}

		chunkSize := (n + numWorkers - 1) / numWorkers

		for w := 0; w < numWorkers; w++ {
			start := w * chunkSize
			end := start + chunkSize
			if end > n {
				end = n
			}

			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				localGradW := make([]float64, d)
				localGradB := 0.0

				for i := start; i < end; i++ {
					z := lr.dotProduct(lr.Weights, X[i]) + lr.Bias
					yPred := sigmoid(z)
					err := yPred - Y[i]

					for j := 0; j < d; j++ {
						localGradW[j] += err * X[i][j]
					}
					localGradB += err
				}

				mutex.Lock()
				for j := 0; j < d; j++ {
					gradW[j] += localGradW[j]
				}
				gradB += localGradB
				mutex.Unlock()
			}(start, end)
		}
		wg.Wait()

		for j := 0; j < d; j++ {
			lr.Weights[j] -= lr.LearningRate * gradW[j] / float64(n)
		}
		lr.Bias -= lr.LearningRate * gradB / float64(n)
	}
}

// Predict returns predictions (0 or 1) for multiple samples
func (lr *LogisticRegressionConcurrent) Predict(X [][]float64) []float64 {
	predictions := make([]float64, len(X))
	for i := range X {
		predictions[i] = lr.PredictOne(X[i])
	}
	return predictions
}

// PredictOne returns the prediction for a single sample
func (lr *LogisticRegressionConcurrent) PredictOne(X []float64) float64 {
	y := sigmoid(lr.dotProduct(lr.Weights, X) + lr.Bias)
	if y >= 0.5 {
		return 1.0
	}
	return 0.0
}

// dotProduct computes the dot product between two vectors
func (lr *LogisticRegressionConcurrent) dotProduct(v1, v2 []float64) float64 {
	sum := 0.0
	for i := range v1 {
		sum += v1[i] * v2[i]
	}
	return sum
}

// sigmoid computes the logistic activation
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
