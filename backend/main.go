package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"github.com/sugarme/tokenizer"
)

// Global variables
var tk *tokenizer.Tokenizer
var modelConfig ModelConfig
var tokenizerMutex = &sync.Mutex{}

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
	var err error // Declare err here

	// --- FIX 1: Correctly capture both return values ---
	tokenizerPath := "./distilbert-sst2-onnx/tokenizer.json"
	tk = tokenizer.NewTokenizerFromFile(tokenizerPath)
	// ---

	// Now, this check will work correctly
	if tk == nil {
		log.Fatalf("Failed to load tokenizer from '%s': %v", tokenizerPath, err)
	}
	log.Println("Tokenizer loaded successfully.")

	configPath := "./distilbert-sst2-onnx/config.json"
	configFile, err := os.ReadFile(configPath)
	if err != nil {
		log.Fatalf("Failed to read model config from '%s': %v", configPath, err)
	}
	err = json.Unmarshal(configFile, &modelConfig)
	if err != nil {
		log.Fatalf("Failed to parse model config: %v", err)
	}
	log.Println("Model config loaded successfully.")

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

	tokenizerMutex.Lock()
	defer tokenizerMutex.Unlock()

	encoded, err := tk.Encode(tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(payload.Input)), false)
	if err != nil {
		http.Error(w, fmt.Sprintf("Tokenization failed: %v", err), http.StatusInternalServerError)
		return
	}

	inputIDs := encoded.GetIds()

	var idStrings []string
	for _, id := range inputIDs {
		idStrings = append(idStrings, strconv.Itoa(int(id)))
	}
	inputStringForCpp := strings.Join(idStrings, ",")

	hostAppPath := "./ml_host_prod_go"
	// --- FIX 2: Correct the model path ---
	modelPath := "./distilbert-sst2-onnx/model.onnx"
	// ---
	enclavePath := "./enclave/enclave_prod.signed.so"

	cmd := exec.Command(hostAppPath, modelPath, enclavePath, "--use-stdin")
	cmd.Stdin = strings.NewReader(inputStringForCpp)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()
	if runErr != nil {
		log.Printf("Host app stderr: %s", stderr.String())
		http.Error(w, fmt.Sprintf("Inference failed: %s", stderr.String()), http.StatusInternalServerError)
		return
	}

	outputParts := strings.Split(strings.TrimSpace(stdout.String()), ",")
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

	resp := ResponsePayload{
		InputText:      payload.Input,
		PredictedLabel: predictedLabel,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(&resp)
}