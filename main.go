package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"time"

	"TP_concurrente/models/regression"
)

var X [][]float64
var Y []float64
var model *regression.LogisticRegressionConcurrent
var trained bool
var outputFile *os.File

func main() {
	var err error
	outputFile, err = os.Create("output.txt")
	if err != nil {
		fmt.Println("Error creando output.txt:", err)
		return
	}
	defer outputFile.Close()

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("\n========= MENÚ PRINCIPAL =========")
		fmt.Println("1. Cargar dataset")
		fmt.Println("2. Entrenar modelo")
		fmt.Println("3. Evaluar precisión")
		fmt.Println("4. Ejecutar simulación Promela")
		fmt.Println("5. Salir")
		fmt.Print("Seleccione una opción: ")

		scanner.Scan()
		switch scanner.Text() {
		case "1":
			loadDatasetMenu()
		case "2":
			trainModelMenu()
		case "3":
			evaluateModelMenu()
		case "4":
			runPromelaSimulation()
		case "5":
			fmt.Println("Saliendo del programa.")
			return
		default:
			fmt.Println("Opción inválida.")
		}
	}
}

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

	colIndex := make(map[string]int)
	for i, name := range headers {
		colIndex[name] = i
	}

	requiredCols := []string{"amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"}
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
		for _, col := range requiredCols[:5] {
			val, err := strconv.ParseFloat(record[colIndex[col]], 64)
			if err != nil {
				val = 0.0
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

func loadDatasetMenu() {
	var err error
	X, Y, err = loadDataset("Cifer-Fraud-Detection-Dataset-AF-part-1-14.csv")
	if err != nil {
		fmt.Println("Error al cargar dataset:", err)
		return
	}
	trained = false
	fmt.Fprintf(outputFile, "✅ Dataset cargado con %d muestras\n", len(X))
	fmt.Println("Dataset cargado exitosamente.")
}

func trainModelMenu() {
	if len(X) == 0 {
		fmt.Println("Primero debe cargar el dataset.")
		return
	}
	model = regression.NewLogisticRegressionConcurrent(0.01, 500)
	splitIndex := int(0.8 * float64(len(X)))
	XTrain, YTrain := X[:splitIndex], Y[:splitIndex]
	start := time.Now()
	model.TrainConcurrently(XTrain, YTrain, 8)
	elapsed := time.Since(start)
	trained = true
	fmt.Fprintf(outputFile, "✅ Modelo entrenado en %s\n", elapsed)
	fmt.Printf("Modelo entrenado en %s\n", elapsed)
}

func evaluateModelMenu() {
	if !trained {
		fmt.Println("Primero debe entrenar el modelo.")
		return
	}
	splitIndex := int(0.8 * float64(len(X)))
	XTest, YTest := X[splitIndex:], Y[splitIndex:]
	preds := model.Predict(XTest)
	acc := calculateAccuracy(preds, YTest)
	fmt.Printf("Precisión del modelo: %.2f%%\n", acc*100)
	fmt.Fprintf(outputFile, "🎯 Precisión del modelo: %.2f%%\n", acc*100)
}

func runPromelaSimulation() {
	fmt.Println("Ejecutando simulación con Promela...")
	commands := []string{
		"spin -a logistic.pml",
		"gcc -o pan pan.c",
		"./pan -a",
	}
	for _, cmdStr := range commands {
		fmt.Println("->", cmdStr)
		cmd := exec.Command("bash", "-c", cmdStr)
		cmd.Stdout = outputFile
		cmd.Stderr = outputFile
		err := cmd.Run()
		if err != nil {
			fmt.Println("❌ Error al ejecutar:", cmdStr)
			fmt.Fprintf(outputFile, "❌ Error en comando: %s -> %v\n", cmdStr, err)
			return
		}
	}
	fmt.Println("✅ Simulación ejecutada. Resultados en output.txt")
}

func calculateAccuracy(preds, actuals []float64) float64 {
	correct := 0
	for i := 0; i < len(preds); i++ {
		if preds[i] == actuals[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(preds))
}
