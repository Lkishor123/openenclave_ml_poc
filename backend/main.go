package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
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
	InputText string `json:"input_text"`
	Sentiment string `json:"sentiment"`
	Error     string `json:"error,omitempty"`
}

// --- Sentiment Analysis Logic ---

// Pre-computed embeddings for reference sentences.
var (
	// Embedding for: "This statement is good, positive, and happy."
	positiveReferenceEmbedding = []float32{0.034, -0.027, 0.033, 0.020, 0.079, 0.011, 0.047, 0.023, -0.052, -0.069, -0.028, -0.055, -0.035, 0.043, -0.035, 0.052, 0.053, 0.007, 0.025, 0.060, 0.011, -0.040, -0.050, 0.014, 0.010, 0.026, 0.067, 0.001, -0.051, -0.032, 0.030, -0.037, 0.006, -0.078, -0.003, -0.030, 0.010, 0.000, -0.037, -0.033, 0.008, -0.028, 0.050, -0.017, -0.035, 0.034, -0.003, 0.049, 0.031, -0.022, -0.063, 0.034, 0.018, -0.011, -0.017, 0.039, -0.015, -0.011, 0.017, -0.061, -0.001, -0.026, 0.042, 0.003, 0.042, -0.009, -0.044, 0.046, -0.024, -0.030, 0.015, -0.034, -0.018, -0.003, 0.040, -0.023, 0.007, -0.005, 0.060, 0.018, 0.015, 0.007, 0.028, 0.076, 0.000, 0.014, -0.007, 0.011, -0.039, 0.056, 0.003, -0.055, 0.042, 0.020, 0.071, -0.029, 0.024, 0.016, -0.005, 0.049, -0.004, -0.022, 0.034, 0.014, -0.091, 0.009, 0.018, -0.060, -0.002, 0.013, 0.010, 0.008, -0.055, -0.013, -0.037, 0.074, 0.046, -0.026, -0.108, -0.017, 0.022, 0.038, 0.035, 0.075, 0.012, 0.010, -0.005, 0.046, -0.003, -0.014, 0.039, 0.076, -0.020, -0.046, 0.015, -0.031, -0.007, -0.048, 0.023, 0.003, -0.012, -0.014, 0.041, 0.024, 0.038, -0.032, 0.014, -0.025, -0.023, 0.020, -0.004, -0.021, 0.010, -0.055, -0.022, 0.062, -0.003, 0.032, -0.058, 0.024, 0.026, 0.017, 0.002, 0.014, -0.069, 0.030, 0.061, 0.037, -0.025, 0.016, -0.014, 0.041, 0.047, -0.032, -0.034, -0.001, 0.068, -0.006, 0.003, 0.020, -0.080, 0.060, 0.003, 0.013, -0.003, -0.052, 0.014, -0.018, 0.022, 0.000, -0.062, -0.074, -0.000, -0.034, 0.025, 0.053, -0.023, 0.010, 0.033, 0.029, 0.003, 0.019, 0.010, -0.040, -0.021, -0.033, 0.013, 0.030, -0.001, 0.020, -0.031, -0.011, 0.098, -0.024, -0.005, 0.025, 0.010, -0.064, -0.008, -0.031, 0.018, 0.004, -0.033, -0.008, -0.038, 0.068, 0.032, -0.077, 0.010, 0.034, -0.067, -0.012, 0.011, 0.006, 0.029, 0.019, 0.016, -0.026, 0.091, -0.048, 0.004, 0.055, 0.004, 0.048, 0.036, -0.038, 0.008, 0.008, -0.030, -0.017, -0.014, -0.001, -0.016, 0.026, -0.028, -0.073, -0.017, -0.058, 0.031, 0.077, -0.040, -0.010, 0.042, 0.019, -0.028, -0.045, -0.026, -0.013, 0.048, 0.010, 0.054, 0.006, 0.033, 0.022, 0.024, -0.034, 0.035, -0.014, -0.032, -0.080, -0.026, 0.046, -0.061, -0.032, 0.026, -0.027, -0.024, -0.035, -0.007, 0.004, 0.016, -0.007, -0.012, -0.022, 0.061, 0.011, 0.007, -0.019, -0.019, -0.011, 0.019, 0.040, 0.004, -0.015, 0.002, 0.012, 0.051, -0.087, -0.247, 0.012, -0.006, -0.040, 0.023, 0.006, 0.031, -0.038, -0.028, -0.039, -0.013, -0.004, 0.020, 0.048, 0.049, 0.011, -0.008, -0.036, 0.020, 0.011, -0.019, -0.053, -0.000, 0.030, -0.001, -0.007, -0.064, -0.002, -0.051, 0.002, 0.005, -0.002, 0.022, -0.021, -0.013, 0.024, 0.047, -0.039, -0.002, -0.021, 0.021, -0.049, -0.033, -0.031, 0.038, -0.023, -0.007, -0.041, 0.005, 0.031, -0.003, 0.023, 0.060, -0.020, -0.041, -0.013, -0.023, 0.050, -0.053, -0.021, 0.013, -0.010, -0.055, -0.013, 0.020, -0.036, -0.013, -0.080, 0.032, 0.037, -0.024, 0.006, 0.013, -0.069, 0.025, 0.013, -0.046, -0.048, -0.016, 0.009, -0.012, -0.071, 0.012, 0.021, -0.001, 0.024, 0.005, -0.037, -0.053, 0.010, 0.044, -0.020, -0.036, 0.027, 0.045, 0.016, 0.021, -0.012, 0.032, -0.013, 0.072, -0.012, 0.033, 0.020, 0.004, -0.021, -0.072, 0.003, 0.078, 0.039, -0.006, -0.013, 0.029, -0.037, 0.026, -0.052, 0.061, 0.038, -0.004, 0.001, 0.019, 0.001, -0.051, 0.029, -0.071, 0.018, 0.026, 0.012, -0.015, 0.008, -0.012, -0.004, -0.018, -0.019, 0.007, 0.033, -0.057, -0.080, 0.036, 0.029, -0.010, -0.006, -0.023, 0.047, 0.037, 0.030, -0.032, -0.020, -0.044, 0.008, -0.064, -0.029, -0.039, -0.071, -0.000, -0.032, 0.006, -0.040, 0.065, -0.031, -0.010, -0.047, -0.016, 0.002, -0.015, 0.070, -0.007, -0.015, -0.017, -0.002, 0.035, 0.021, 0.001, -0.022, -0.023, 0.041, -0.045, -0.009, 0.054, 0.034, 0.005, -0.018, -0.012, -0.046, 0.038, -0.008, -0.013, -0.035, -0.010, 0.042, -0.012, 0.030, 0.010, 0.000, -0.067, -0.074, -0.001, 0.002, -0.011, -0.020, -0.016, -0.030, -0.027, 0.055, 0.036, -0.111, 0.042, 0.017, 0.019, 0.043, -0.019, -0.029, -0.016, 0.018, -0.023, -0.012, 0.010, -0.000, 0.001, -0.047, -0.017, -0.025, -0.011, -0.016, -0.028, -0.003, -0.040, -0.047, -0.016, -0.028, 0.030, -0.033, 0.033, 0.036, -0.005, -0.009, -0.016, -0.004, -0.017, 0.035, 0.025, -0.017, -0.021, -0.045, 0.003, 0.017, -0.047, 0.017, -0.029, 0.009, 0.007, 0.006, -0.031, -0.041, 0.027, -0.039, -0.007, 0.026, -0.015, -0.038, 0.005, -0.015, -0.059, 0.053, 0.052, 0.037, 0.019, -0.053, 0.022, -0.023, 0.026, -0.001, -0.020, 0.011, 0.038, -0.024, -0.017, -0.028, 0.034, -0.069, -0.007, 0.047, -0.036, 0.032, -0.009, -0.012, -0.002, 0.023, -0.037, 0.029, -0.016, 0.024, -0.001, 0.048, 0.003, -0.004, 0.058, 0.024, -0.020, 0.011, -0.022, 0.032, 0.036, -0.010, -0.028, 0.050, 0.019, -0.024, -0.015, 0.026, 0.012, -0.071, 0.048, 0.078, -0.075, 0.008, 0.014, -0.027, 0.038, 0.027, 0.053, 0.016, 0.029, -0.033, 0.001, 0.021, -0.015, 0.020, 0.015, 0.076, 0.013, 0.047, -0.015, 0.066, 0.008, -0.063, -0.009, -0.013, -0.004, 0.028, -0.039, 0.038, 0.022, 0.030, 0.043, -0.011, 0.001, -0.007, 0.046, 0.020, 0.028, 0.058, 0.014, -0.058, 0.041, -0.016, 0.031, 0.012, -0.007, 0.011, 0.031, -0.000, 0.085, 0.029, -0.021, -0.021, 0.011, 0.001, -0.020, 0.024, 0.024, 0.002, -0.004, 0.029, 0.009, 0.026, -0.037, -0.024, 0.005, 0.029, 0.019, 0.003, -0.030, -0.012, -0.024, -0.024, 0.015, 0.023, -0.055, -0.047, 0.013, -0.038, -0.013, 0.022, -0.069, -0.043, 0.042, 0.075, -0.023, 0.023, 0.090, 0.013, -0.004, -0.035, 0.055, 0.020, 0.007, 0.016, -0.084, 0.069, 0.003, 0.020, -0.065, 0.012, -0.007, -0.030, 0.029, 0.047, -0.030, -0.042, 0.005, -0.006, -0.005, 0.071, -0.038, 0.059, 0.008, -0.036, -0.014, 0.010, 0.033, -0.019, 0.003, -0.013, -0.033, -0.075, -0.010, -0.017, 0.027, -0.064, 0.037, -0.016, 0.026, 0.022, -0.041, -0.025, -0.045, -0.012, -0.094, -0.025, 0.019, -0.024, 0.014, -0.041, 0.015, 0.021, 0.023, 0.002, 0.029, 0.021}
)

// Calculates the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	var dotProduct, aMag, bMag float64
	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i] * b[i])
		aMag += float64(a[i] * a[i])
		bMag += float64(b[i] * b[i])
	}
	if aMag == 0 || bMag == 0 {
		return 0.0
	}
	return dotProduct / (math.Sqrt(aMag) * math.Sqrt(bMag))
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

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to get stderr pipe: %w", err)
	}

	log.Println("Starting C++ enclave worker...")
	if err := cmd.Start(); err != nil {
		// It's helpful to try and read from stderr if the start fails
		var errBuf bytes.Buffer
		io.Copy(&errBuf, stderr)
		return nil, fmt.Errorf("failed to start C++ worker: %w. Stderr: %s", err, errBuf.String())
	}
	log.Println("C++ enclave worker started successfully.")

	// Goroutine to continuously log any errors from the C++ worker.
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

	worker.mutex.Lock()
	defer worker.mutex.Unlock()

	// --- 1. Tokenization ---
	tokenizerDir := "./tokenizer"
	pyCmd := exec.Command("python3", "tokenize_script.py", tokenizerDir)
	pyCmd.Stdin = strings.NewReader(payload.Input)

	pyOutput, err := pyCmd.CombinedOutput()
	if err != nil {
		log.Printf("Tokenization script failed: %s", string(pyOutput))
		http.Error(w, "Tokenization failed", http.StatusInternalServerError)
		return
	}
	tokenString := strings.TrimSpace(string(pyOutput))

	// --- 2. Inference via C++ Worker ---
	if _, err := fmt.Fprintln(worker.stdin, tokenString); err != nil {
		log.Printf("Error sending data to C++ worker: %v. The worker may have crashed.", err)
		http.Error(w, "Failed to send data to C++ worker", http.StatusInternalServerError)
		// Attempt to restart the worker if it has crashed.
		var startErr error
		worker, startErr = startEnclaveWorker()
		if startErr != nil {
			log.Fatalf("FATAL: Could not restart the enclave worker: %v", startErr)
		}
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
	for i, part := range outputParts {
		val, _ := strconv.ParseFloat(strings.TrimSpace(part), 64)
		embeddings[i] = float32(val)
	}

	// --- 4. Classify Sentiment using Cosine Similarity ---
	posSimilarity := cosineSimilarity(embeddings, positiveReferenceEmbedding)
	negSimilarity := cosineSimilarity(embeddings, negativeReferenceEmbedding)

	var sentiment string
	if posSimilarity > negSimilarity {
		sentiment = "Positive"
	} else if negSimilarity > posSimilarity {
		sentiment = "Negative"
	} else {
		sentiment = "Neutral"
	}
	log.Printf("Input: '%s', Sentiment: %s (Pos-Sim: %f, Neg-Sim: %f)", payload.Input, sentiment, posSimilarity, negSimilarity)

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
	
	// This will ensure that the C++ process is terminated gracefully when the Go app exits.
	defer func() {
		log.Println("Shutting down server. Closing C++ worker stdin...")
		worker.stdin.Close()
		log.Println("Waiting for C++ worker to exit...")
		worker.cmd.Wait()
		log.Println("C++ worker exited cleanly.")
	}()

	fs := http.FileServer(http.Dir("./frontend"))
	http.Handle("/", fs)
	http.HandleFunc("/infer", handleInference)

	log.Println("Starting server on :8080...")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

