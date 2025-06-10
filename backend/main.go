package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"path/filepath"
	"strings"
)

// RequestPayload defines the structure of the incoming JSON from the UI
type RequestPayload struct {
	Input string `json:"input"`
}

// ResponsePayload defines the structure of the JSON sent back to the UI
type ResponsePayload struct {
	Output string `json:"output"`
	Error  string `json:"error,omitempty"`
}

func main() {
	// --- Serve the Frontend UI ---
	// The executable will be in the root of our container, so paths are relative to that.
	fs := http.FileServer(http.Dir("./frontend"))
	http.Handle("/", fs)

	// --- Handle Inference API Requests ---
	http.HandleFunc("/infer", handleInference)

	log.Println("Starting server on :8080...")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func handleInference(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
		return
	}

	var payload RequestPayload
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// --- Prepare to call the C++ host application ---
	// Paths are based on the final Docker container's file structure
	hostAppPath := "./ml_host_prod_go" // A slightly renamed host app
	modelPath := "./model/simple_model.onnx"
	enclavePath := "./enclave/enclave_prod.signed.so"

	cmd := exec.Command(hostAppPath, modelPath, enclavePath, "--use-stdin") // Add a flag to our host app

	// Pass the input from the UI to the C++ host's stdin
	cmd.Stdin = strings.NewReader(payload.Input)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	log.Printf("Running command: %s with input: %s", strings.Join(cmd.Args, " "), payload.Input)
	err := cmd.Run()
	if err != nil {
		log.Printf("Host app error: %v", err)
		log.Printf("Host app stderr: %s", stderr.String())
		http.Error(w, fmt.Sprintf("Inference failed: %s", stderr.String()), http.StatusInternalServerError)
		return
	}

	// --- Send response back to UI ---
	resp := ResponsePayload{
		Output: stdout.String(),
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("Failed to write response: %v", err)
	}
}