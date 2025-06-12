package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"sync"
)

// EnclaveWorker manages the long-running C++ worker process.
type EnclaveWorker struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner
	mutex  sync.Mutex // Ensures only one request is processed at a time.
}

// Global instance of our worker.
var worker *EnclaveWorker

// Structs for UI communication.
type RequestPayload struct {
	Input string `json:"input"`
}

type ResponsePayload struct {
	InputText  string    `json:"input_text"`
	Sentiment  string    `json:"sentiment"`
	Embeddings []float32 `json:"embeddings,omitempty"`
	Error      string    `json:"error,omitempty"`
}

// Starts the C++ worker process and sets up pipes for communication.
func startEnclaveWorker() (*EnclaveWorker, error) {
	hostAppPath := "./ml_host_prod_go"
	modelPath := "./model/bert.bin"
	enclavePath := "./enclave/enclave_prod.signed.so"

	cmd := exec.Command(hostAppPath, modelPath, enclavePath)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to get stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to get stdout pipe: %w", err)
	}

	// MODIFIED: Correctly set up a pipe to continuously read from the worker's stderr.
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to get stderr pipe: %w", err)
	}

	log.Println("Starting C++ enclave worker...")
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start C++ worker: %w", err)
	}
	log.Println("C++ enclave worker started successfully.")

	// MODIFIED: This goroutine now continuously scans the stderr pipe and logs any output from the C++ worker.
	// This will reveal why the worker is exiting after the first inference.
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			log.Printf("[C++ Worker Stderr]: %s", scanner.Text())
		}
	}()

	return &EnclaveWorker{
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewScanner(stdout),
		mutex:  sync.Mutex{},
	}, nil
}

// The main inference handler.
func handleInference(w http.ResponseWriter, r *http.Request) {
	var payload RequestPayload
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// --- 1. Tokenization (as a short-lived process) ---
	tokenizerDir := "./tokenizer"
	pyCmd := exec.Command("python3", "tokenize_script.py", tokenizerDir)
	pyCmd.Stdin = strings.NewReader(payload.Input)

	var pyStdout, pyStderr strings.Builder
	pyCmd.Stdout = &pyStdout
	pyCmd.Stderr = &pyStderr

	if err := pyCmd.Run(); err != nil {
		log.Printf("Python script error: %s", pyStderr.String())
		http.Error(w, fmt.Sprintf("Tokenization failed: %s", pyStderr.String()), http.StatusInternalServerError)
		return
	}
	tokenString := strings.TrimSpace(pyStdout.String())

	// --- 2. Inference via C++ Worker ---
	worker.mutex.Lock()
	defer worker.mutex.Unlock()

	if _, err := fmt.Fprintln(worker.stdin, tokenString); err != nil {
		// This error happens when the C++ worker has already exited.
		log.Printf("Error sending data to C++ worker: %v", err)
		http.Error(w, "Failed to send data to C++ worker. The worker may have crashed.", http.StatusInternalServerError)
		return
	}

	if !worker.stdout.Scan() {
		if err := worker.stdout.Err(); err != nil {
			log.Printf("Error reading from C++ worker stdout: %v", err)
		}
		http.Error(w, "Failed to read data from C++ worker", http.StatusInternalServerError)
		return
	}
	resultString := worker.stdout.Text()

	// --- 3. Process the output ---
	outputParts := strings.Split(strings.TrimSpace(resultString), ",")
	embeddings := make([]float32, len(outputParts))
	var sum float64
	for i, part := range outputParts {
		val, _ := strconv.ParseFloat(strings.TrimSpace(part), 64)
		embeddings[i] = float32(val)
		sum += val
	}

	// --- 4. Classify Sentiment ---
	average := sum / float64(len(embeddings))
	var sentiment string
	if average > 0.01 {
		sentiment = "Positive"
	} else if average < -0.01 {
		sentiment = "Negative"
	} else {
		sentiment = "Neutral"
	}
	log.Printf("Input: '%s', Sentiment: %s (Avg: %f)", payload.Input, sentiment, average)

	// --- 5. Send Final Response ---
	resp := ResponsePayload{
		InputText: payload.Input,
		Sentiment: sentiment,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(&resp)
}

func main() {
	var err error
	worker, err = startEnclaveWorker()
	if err != nil {
		log.Fatalf("FATAL: Could not start the enclave worker process: %v", err)
	}
	defer func() {
		log.Println("Closing stdin to C++ worker...")
		worker.stdin.Close()
		log.Println("Waiting for C++ worker to exit...")
		worker.cmd.Wait()
		log.Println("C++ worker exited.")
	}()

	fs := http.FileServer(http.Dir("./frontend"))
	http.Handle("/", fs)
	http.HandleFunc("/infer", handleInference)

	log.Println("Starting server on :8080...")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
