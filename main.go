package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"TP/models/regression"
)

// loadDataset reads a CSV file and splits it into features (X) and targets (Y)
func loadDataset(path string) ([][]float64, []float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	headers, err := reader.Read()
	if err != nil {
		return nil, nil, err
	}

	// Mapa de nombres de columnas a índices
	colIndex := make(map[string]int)
	for i, name := range headers {
		colIndex[name] = i
	}

	requiredCols := []string{
		"amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud",
	}

	for _, col := range requiredCols {
		if _, ok := colIndex[col]; !ok {
			return nil, nil, fmt.Errorf("missing required column: %s", col)
		}
	}

	var X [][]float64
	var Y []float64

	for {
		record, err := reader.Read()
		if err != nil {
			break
		}

		var features []float64
		for _, col := range requiredCols[:5] { // solo las features
			val, err := strconv.ParseFloat(record[colIndex[col]], 64)
			if err != nil {
				val = 0.0 // por si hay valores vacíos o inválidos
			}
			features = append(features, val)
		}

		target, err := strconv.ParseFloat(record[colIndex["isFraud"]], 64)
		if err != nil {
			return nil, nil, err
		}

		X = append(X, features)
		Y = append(Y, target)
	}

	fmt.Printf("Fraud dataset loaded: %d samples, %d features\n", len(X), len(X[0]))
	return X, Y, nil
}

// calculateAccuracy computes the accuracy of predictions
func calculateAccuracy(preds, actuals []float64) float64 {
	correct := 0
	for i := 0; i < len(preds); i++ {
		if preds[i] == actuals[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(preds))
}

func main() {
	// Cargar el dataset
	X, Y, err := loadDataset("Cifer-Fraud-Detection-Dataset-AF-part-1-14.csv")
	if err != nil {
		log.Fatalf("Error loading dataset: %v", err)
	}

	// Dividir en entrenamiento y prueba (80% entrenamiento, 20% prueba)
	splitIndex := int(0.8 * float64(len(X)))
	XTrain, YTrain := X[:splitIndex], Y[:splitIndex]
	XTest, YTest := X[splitIndex:], Y[splitIndex:]

	// Crear y entrenar el modelo de regresión logística concurrente
	model := regression.NewLogisticRegressionConcurrent(0.01, 500)
	start := time.Now()
	model.TrainConcurrently(XTrain, YTrain, 8)
	elapsed := time.Since(start)

	// Realizar predicciones y evaluar precisión
	preds := model.Predict(XTest)
	accuracy := calculateAccuracy(preds, YTest)

	fmt.Printf("Training time: %s\n", elapsed)
	fmt.Printf("First 3 weights: %.4f %.4f %.4f ...\n", model.Weights[0], model.Weights[1], model.Weights[2])
	fmt.Printf("Bias: %.4f\n", model.Bias)
	fmt.Printf("Accuracy: %.2f%%\n", accuracy*100)
}
