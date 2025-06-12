package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
)

// Global model config instance
var modelConfig ModelConfig

// Struct to parse the model's config.json
type ModelConfig struct {
	ID2Label map[string]string `json:"id2label"`
}

// Structs for communicating with the UI
type RequestPayload struct {
	Input string `json:"input"`
}
type ResponsePayload struct {
	InputText      string `json:"input_text"`
	PredictedLabel string `json:"predicted_label"`
	Error          string `json:"error,omitempty"`
}

func main() {

	// Load the model config at startup (we still need this)
    configPath := "./distilbert-sst2-ggml/config.json"
	configFile, err := os.ReadFile(configPath)
	if err != nil {
		log.Fatalf("Failed to read model config from '%s': %v", configPath, err)
	}
	err = json.Unmarshal(configFile, &modelConfig)
	if err != nil {
		log.Fatalf("Failed to parse model config: %v", err)
	}
	log.Println("Model config loaded successfully.")

	// Serve the frontend static files and the API endpoint
	fs := http.FileServer(http.Dir("./frontend"))
	http.Handle("/", fs)
	http.HandleFunc("/infer", handleInference)

	log.Println("Starting server on :8080...")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func handleInference(w http.ResponseWriter, r *http.Request) {
	var payload RequestPayload
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// --- 1. Call Python script for tokenization ---
    tokenizerDir := "./distilbert-sst2-ggml"
	// Execute `python3 tokenize_script.py <tokenizer_directory>`
	pyCmd := exec.Command("python3", "tokenize_script.py", tokenizerDir)
	// Pass the input text from the UI to the Python script's standard input
	pyCmd.Stdin = strings.NewReader(payload.Input)

	var pyStdout, pyStderr bytes.Buffer
	pyCmd.Stdout = &pyStdout
	pyCmd.Stderr = &pyStderr

	err := pyCmd.Run()
	if err != nil {
		log.Printf("Python script error: %s", pyStderr.String())
		http.Error(w, fmt.Sprintf("Tokenization failed: %s", pyStderr.String()), http.StatusInternalServerError)
		return
	}

	inputStringForCpp := strings.TrimSpace(pyStdout.String())
	// ---

	// 2. Execute the C++ host application as a subprocess
	hostAppPath := "./ml_host_prod_go"
    modelPath := "./distilbert-sst2-ggml/model.ggml"
	enclavePath := "./enclave/enclave_prod.signed.so"

	cppCmd := exec.Command(hostAppPath, modelPath, enclavePath, "--use-stdin")
	cppCmd.Stdin = strings.NewReader(inputStringForCpp)

	var cppStdout, cppStderr bytes.Buffer
	cppCmd.Stdout = &cppStdout
	cppCmd.Stderr = &cppStderr

	runErr := cppCmd.Run()
	if runErr != nil {
		log.Printf("Host app stderr: %s", cppStderr.String())
		http.Error(w, fmt.Sprintf("Inference failed: %s", cppStderr.String()), http.StatusInternalServerError)
		return
	}

	// 3. Process the output (logits) from the C++ app
	outputParts := strings.Split(strings.TrimSpace(cppStdout.String()), ",")
	logits := make([]float32, len(outputParts))
	for i, part := range outputParts {
		val, _ := strconv.ParseFloat(strings.TrimSpace(part), 32)
		logits[i] = float32(val)
	}

	var maxLogit float32 = -1e9
	var predictedIndex int = 0
	for i, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
			predictedIndex = i
		}
	}

	predictedLabel := modelConfig.ID2Label[strconv.Itoa(predictedIndex)]

	// 4. Send the final, human-readable result to the UI
	resp := ResponsePayload{
		InputText:      payload.Input,
		PredictedLabel: predictedLabel,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(&resp)
}
