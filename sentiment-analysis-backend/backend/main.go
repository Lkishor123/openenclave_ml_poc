package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/rs/cors"
)

// --- Structs for API communication ---
type RequestPayload struct {
	Input string `json:"input"`
}

type ResponsePayload struct {
	InputText string `json:"input_text,omitempty"`
	Sentiment string `json:"sentiment,omitempty"`
	Error     string `json:"error,omitempty"`
}

// --- NEW STRUCT for Attestation Response ---
type AttestationResponse struct {
	EvidenceHex string `json:"evidence_hex,omitempty"`
	Error       string `json:"error,omitempty"`
}

// --- Sentiment Analysis Logic (Unchanged) ---

// Pre-computed embeddings for reference sentences.
var positiveReferenceEmbedding = []float32{0.006, 0.022, 0.057, 0.026, 0.012, 0.031, 0.006, 0.014, -0.002, 0.013, 0.009, 0.000, -0.025, -0.010, -0.004, 0.010, 0.013, 0.035, 0.004, 0.053, 0.009, -0.012, -0.058, 0.022, 0.010, 0.003, -0.019, 0.057, -0.031, -0.019, 0.030, 0.005, -0.005, -0.066, 0.040, -0.015, 0.011, 0.012, -0.036, -0.026, -0.027, -0.031, 0.025, -0.005, -0.032, 0.012, -0.015, 0.026, -0.011, 0.009, -0.073, 0.019, -0.033, 0.001, 0.050, 0.020, -0.004, -0.022, 0.016, 0.023, -0.000, -0.008, 0.020, 0.000, 0.013, -0.006, -0.028, -0.018, -0.033, 0.009, -0.023, -0.035, -0.020, -0.046, 0.033, -0.025, -0.044, -0.003, 0.043, 0.045, 0.006, -0.020, 0.045, 0.076, -0.006, 0.017, -0.022, 0.004, -0.043, 0.021, -0.040, -0.004, 0.101, 0.002, 0.015, -0.034, 0.024, 0.001, 0.032, 0.029, -0.009, -0.055, 0.009, -0.061, -0.034, -0.037, -0.037, -0.004, 0.027, 0.050, -0.001, 0.027, -0.028, 0.009, -0.043, 0.104, 0.053, 0.018, -0.029, 0.006, 0.008, 0.027, 0.040, 0.075, -0.003, -0.000, -0.027, 0.009, -0.037, -0.023, 0.036, 0.072, -0.013, -0.048, -0.012, -0.003, 0.003, -0.074, 0.055, -0.058, -0.012, -0.015, 0.004, -0.017, 0.054, -0.010, -0.041, -0.035, 0.008, -0.003, -0.016, 0.020, 0.024, -0.015, -0.016, 0.040, -0.032, 0.050, -0.015, -0.012, 0.008, -0.029, 0.009, 0.003, -0.004, 0.015, 0.029, 0.013, 0.011, 0.024, -0.057, -0.040, 0.046, 0.030, 0.023, 0.021, 0.071, -0.011, -0.001, 0.011, -0.065, -0.002, -0.027, 0.055, -0.011, 0.007, 0.035, -0.007, 0.033, -0.066, -0.087, -0.050, 0.024, 0.012, 0.020, -0.005, -0.022, -0.026, 0.007, 0.023, -0.012, 0.028, 0.035, -0.097, -0.038, 0.047, 0.035, -0.049, -0.054, 0.002, -0.054, 0.018, 0.072, -0.013, -0.029, 0.027, 0.013, -0.038, 0.009, 0.003, 0.039, -0.008, -0.074, 0.029, 0.029, 0.063, 0.050, -0.030, 0.036, 0.049, -0.044, -0.061, -0.029, 0.042, -0.012, 0.076, 0.061, -0.018, 0.063, -0.007, -0.017, 0.058, 0.026, 0.048, -0.005, 0.026, -0.016, 0.001, -0.041, -0.008, 0.076, -0.013, 0.044, 0.018, 0.030, -0.063, 0.050, -0.034, 0.045, 0.034, 0.008, 0.007, 0.059, 0.033, 0.008, -0.067, -0.044, 0.011, -0.012, -0.008, 0.034, -0.032, -0.005, 0.024, 0.040, -0.033, 0.050, -0.001, -0.051, -0.068, 0.050, 0.071, 0.039, 0.001, 0.071, -0.071, -0.047, -0.026, -0.047, 0.037, 0.065, 0.023, -0.050, 0.002, -0.025, 0.022, -0.001, 0.011, -0.010, -0.030, 0.040, -0.070, -0.051, 0.034, 0.047, -0.014, 0.039, -0.067, -0.296, 0.022, 0.004, -0.020, 0.031, -0.040, -0.009, -0.012, -0.033, 0.029, 0.034, 0.046, 0.047, 0.005, 0.013, 0.030, -0.015, 0.042, -0.007, -0.007, -0.001, -0.027, 0.007, -0.004, -0.022, 0.006, -0.054, -0.000, -0.039, -0.023, -0.009, -0.037, -0.019, -0.020, 0.017, 0.027, 0.011, -0.010, -0.018, -0.044, -0.008, -0.036, -0.002, -0.017, 0.063, 0.043, 0.016, -0.035, -0.034, 0.035, -0.016, -0.025, -0.005, 0.017, -0.004, 0.025, 0.014, 0.019, 0.003, 0.042, -0.023, -0.020, -0.034, -0.003, 0.022, -0.065, -0.005, -0.040, 0.056, 0.028, 0.039, 0.001, 0.018, -0.043, -0.032, 0.035, -0.007, -0.005, -0.009, 0.053, -0.039, -0.010, -0.007, 0.022, -0.049, -0.022, -0.022, -0.021, -0.001, -0.009, 0.045, -0.030, -0.000, -0.011, 0.044, -0.004, -0.036, -0.021, 0.022, -0.069, 0.060, -0.001, 0.032, -0.011, 0.048, -0.008, -0.012, -0.016, 0.082, 0.028, -0.047, 0.012, 0.010, -0.027, 0.051, -0.017, 0.049, 0.042, -0.032, -0.010, -0.000, 0.037, -0.034, 0.023, -0.072, -0.005, 0.015, 0.035, -0.019, 0.007, -0.019, -0.032, -0.041, -0.030, 0.036, 0.037, -0.036, -0.073, -0.000, 0.009, 0.027, -0.027, -0.004, 0.012, 0.017, 0.008, -0.011, -0.020, -0.062, 0.044, -0.034, -0.040, -0.043, -0.023, 0.029, 0.013, 0.011, 0.052, 0.016, -0.046, -0.028, -0.033, -0.069, -0.002, 0.039, 0.077, -0.022, -0.065, 0.011, 0.033, 0.041, -0.045, -0.025, -0.003, -0.055, 0.011, 0.003, 0.040, 0.041, -0.016, -0.024, -0.014, -0.006, 0.012, 0.057, 0.021, -0.007, -0.055, -0.039, 0.055, -0.009, 0.058, -0.016, -0.042, -0.049, -0.100, -0.017, 0.008, -0.050, -0.004, -0.005, -0.032, -0.072, 0.033, -0.020, -0.081, 0.029, 0.009, 0.046, 0.008, -0.006, 0.006, 0.025, -0.017, -0.077, 0.014, 0.014, -0.025, 0.009, -0.059, 0.036, -0.087, -0.038, 0.000, -0.026, 0.006, 0.025, 0.019, 0.003, -0.042, 0.083, -0.022, 0.055, 0.064, -0.033, 0.037, -0.049, -0.015, -0.012, 0.039, 0.014, -0.008, -0.022, 0.018, -0.014, -0.014, 0.025, -0.052, -0.046, 0.035, -0.013, 0.029, -0.045, 0.001, 0.008, 0.001, -0.011, 0.022, -0.088, 0.015, -0.005, -0.018, 0.002, 0.035, 0.016, 0.005, 0.026, -0.039, 0.015, -0.013, 0.059, -0.032, -0.013, 0.030, 0.030, 0.012, -0.005, 0.025, 0.043, -0.018, 0.017, 0.031, 0.054, 0.062, 0.030, -0.039, 0.031, -0.054, -0.024, 0.005, -0.032, 0.048, 0.014, -0.004, -0.038, 0.022, 0.081, 0.021, -0.030, 0.025, -0.014, 0.000, -0.000, 0.005, -0.033, 0.004, 0.022, -0.018, -0.016, 0.022, 0.028, -0.069, 0.043, 0.049, 0.004, -0.013, 0.039, -0.023, 0.018, 0.006, 0.025, -0.031, 0.012, -0.024, 0.014, 0.016, -0.020, 0.017, 0.008, 0.017, 0.005, 0.037, -0.005, 0.034, -0.016, -0.049, -0.027, -0.007, -0.034, 0.049, -0.029, 0.011, 0.017, 0.047, -0.028, 0.016, 0.010, -0.038, 0.028, -0.007, 0.012, -0.009, 0.017, -0.015, 0.024, -0.046, 0.038, -0.027, -0.041, 0.007, 0.003, -0.025, 0.071, -0.036, -0.015, -0.027, 0.025, 0.006, -0.014, 0.059, -0.002, 0.007, -0.015, 0.032, 0.001, -0.000, -0.040, -0.032, 0.029, 0.064, -0.038, 0.038, -0.006, -0.028, -0.049, -0.005, -0.036, -0.006, -0.012, -0.039, 0.033, -0.027, -0.018, -0.034, 0.002, -0.009, 0.069, 0.029, -0.027, 0.047, 0.043, 0.053, -0.084, -0.016, 0.036, 0.051, 0.007, -0.018, -0.054, 0.012, -0.006, 0.033, -0.026, 0.009, -0.048, 0.011, 0.008, 0.034, -0.017, -0.042, -0.023, 0.011, 0.014, 0.013, -0.013, 0.080, -0.009, -0.029, -0.044, 0.085, 0.037, 0.008, -0.026, -0.031, -0.057, -0.101, -0.033, -0.036, 0.006, -0.038, 0.039, -0.053, 0.024, -0.005, -0.038, -0.041, -0.026, -0.021, -0.081, -0.057, -0.021, -0.026, 0.045, -0.002, 0.042, -0.019, 0.017, -0.005, 0.015, 0.021}
var negativeReferenceEmbedding = []float32{0.022, 0.025, 0.025, -0.054, 0.041, -0.018, 0.005, 0.010, 0.026, 0.003, -0.019, 0.007, -0.051, 0.037, -0.005, 0.053, 0.034, -0.021, 0.037, -0.003, 0.010, 0.049, -0.015, -0.040, -0.006, 0.027, 0.029, 0.044, -0.041, -0.017, 0.060, -0.011, 0.028, 0.002, 0.043, -0.007, -0.055, -0.006, -0.037, -0.020, -0.032, -0.005, -0.031, -0.031, -0.062, 0.032, -0.002, -0.019, -0.030, -0.025, -0.057, -0.024, 0.079, -0.026, -0.032, -0.019, 0.007, -0.090, -0.005, 0.048, 0.022, 0.018, -0.005, -0.005, 0.008, 0.001, -0.009, 0.047, -0.039, -0.011, -0.020, -0.016, -0.011, 0.027, 0.019, -0.063, -0.004, 0.041, 0.026, 0.055, 0.021, 0.029, 0.035, 0.052, -0.043, -0.022, -0.010, -0.001, 0.002, 0.060, -0.038, 0.016, 0.073, 0.005, 0.020, -0.049, 0.041, -0.012, 0.008, 0.025, -0.016, -0.040, 0.020, 0.023, -0.015, 0.005, 0.010, 0.039, 0.004, -0.029, 0.014, 0.037, -0.036, -0.050, -0.049, 0.039, 0.032, -0.015, 0.010, -0.023, 0.024, -0.032, -0.018, 0.088, 0.022, 0.056, 0.018, 0.015, 0.038, -0.031, 0.034, 0.064, 0.006, -0.035, -0.017, 0.008, -0.012, -0.004, 0.063, 0.029, 0.000, -0.066, -0.071, -0.012, -0.016, -0.047, -0.018, -0.021, 0.037, 0.045, -0.026, 0.006, 0.013, -0.078, 0.004, 0.026, -0.001, -0.013, -0.029, 0.052, 0.005, 0.002, 0.001, 0.032, -0.002, 0.009, 0.001, 0.032, -0.015, -0.034, -0.065, -0.014, -0.002, 0.015, -0.012, -0.006, 0.097, -0.005, -0.007, 0.010, -0.061, 0.006, -0.001, 0.035, -0.042, 0.022, 0.040, -0.057, -0.029, 0.051, -0.062, -0.060, 0.017, -0.012, 0.047, -0.030, -0.090, -0.025, 0.071, 0.061, -0.053, 0.035, 0.028, -0.013, -0.039, 0.023, 0.013, -0.049, -0.002, 0.015, -0.007, 0.003, 0.108, -0.001, 0.031, 0.035, 0.055, 0.020, 0.024, 0.003, -0.045, -0.007, -0.011, 0.015, -0.016, 0.065, 0.024, -0.034, 0.025, -0.060, -0.019, -0.048, -0.022, -0.002, 0.003, 0.053, 0.000, -0.029, 0.024, -0.013, -0.007, 0.040, -0.017, 0.047, -0.031, 0.008, -0.071, -0.031, 0.004, -0.013, 0.003, -0.008, 0.013, -0.023, -0.028, -0.036, -0.012, 0.016, 0.039, 0.047, 0.020, 0.033, -0.036, -0.028, -0.054, -0.066, -0.076, 0.071, -0.060, 0.045, 0.008, -0.019, 0.032, 0.041, 0.027, -0.043, 0.019, -0.013, -0.011, -0.024, -0.036, 0.015, -0.008, -0.051, 0.058, -0.049, 0.080, 0.031, 0.004, -0.031, 0.055, 0.035, -0.031, -0.009, 0.018, -0.025, -0.012, 0.035, 0.022, 0.023, 0.013, 0.002, 0.010, 0.031, -0.010, -0.037, 0.031, -0.060, -0.230, 0.047, 0.012, -0.005, 0.039, -0.034, 0.031, -0.035, -0.075, -0.005, -0.027, 0.062, -0.016, 0.022, -0.006, -0.022, -0.038, -0.023, 0.001, 0.021, -0.010, -0.019, 0.020, -0.005, 0.017, 0.074, 0.050, -0.002, -0.009, -0.000, 0.029, -0.033, 0.020, 0.035, 0.027, -0.002, 0.020, -0.011, -0.012, -0.018, -0.041, -0.069, 0.046, 0.010, 0.012, 0.026, 0.041, -0.060, -0.008, 0.057, 0.010, -0.085, -0.040, -0.016, 0.009, -0.021, -0.017, -0.008, -0.058, 0.005, -0.038, 0.008, -0.057, 0.022, 0.054, -0.005, 0.019, -0.067, 0.050, -0.034, 0.026, -0.052, 0.055, -0.052, -0.051, 0.008, -0.054, 0.052, -0.019, -0.026, -0.098, -0.040, 0.033, 0.010, 0.039, 0.010, 0.023, 0.029, 0.045, -0.008, 0.040, -0.011, 0.001, -0.054, 0.011, 0.110, 0.021, -0.026, -0.004, 0.012, 0.086, -0.023, 0.048, -0.017, 0.031, 0.002, -0.008, -0.011, -0.022, 0.017, 0.027, 0.038, 0.030, -0.029, -0.063, -0.004, -0.007, 0.022, -0.008, -0.009, 0.037, 0.018, -0.017, 0.014, -0.039, 0.039, 0.011, 0.053, -0.000, -0.023, -0.001, 0.017, -0.004, -0.039, -0.013, -0.003, -0.014, -0.030, -0.018, -0.026, 0.050, 0.008, -0.005, 0.006, -0.018, 0.051, 0.011, -0.061, -0.025, 0.000, -0.041, -0.066, -0.065, -0.040, -0.011, 0.037, -0.017, -0.007, 0.004, -0.058, -0.067, -0.022, -0.031, -0.059, 0.028, 0.019, -0.016, 0.025, 0.012, 0.026, 0.004, 0.021, 0.002, 0.029, -0.005, -0.012, -0.017, 0.017, 0.019, 0.020, -0.030, 0.040, 0.013, -0.033, 0.070, -0.015, 0.002, 0.013, 0.044, -0.031, -0.047, -0.006, 0.084, 0.036, -0.024, -0.092, 0.030, 0.041, -0.039, 0.017, 0.048, 0.074, -0.037, -0.009, -0.003, -0.006, 0.004, -0.016, -0.002, 0.010, -0.019, -0.018, -0.042, -0.026, -0.046, -0.014, 0.014, -0.060, 0.045, 0.001, -0.009, -0.002, -0.016, 0.030, 0.002, 0.056, 0.012, -0.012, 0.026, -0.086, 0.066, 0.013, -0.007, -0.002, -0.019, -0.032, -0.045, 0.024, 0.004, 0.056, 0.027, -0.010, -0.003, -0.026, 0.008, 0.032, 0.038, -0.053, -0.006, 0.022, -0.005, -0.027, -0.023, -0.008, 0.057, -0.054, 0.005, -0.018, -0.054, 0.012, 0.029, 0.018, -0.043, 0.038, 0.006, 0.079, 0.043, -0.027, -0.022, 0.046, -0.039, -0.020, 0.005, 0.016, -0.017, -0.003, -0.014, -0.032, 0.003, -0.082, 0.000, -0.028, -0.017, 0.010, 0.000, 0.014, 0.048, -0.050, -0.043, -0.054, 0.007, 0.003, 0.070, 0.020, -0.036, 0.028, 0.055, 0.023, -0.004, 0.007, -0.041, 0.016, -0.002, 0.032, 0.003, 0.004, -0.003, -0.004, 0.026, 0.033, -0.000, -0.037, 0.068, -0.037, -0.026, 0.063, -0.011, -0.012, -0.020, -0.074, 0.004, -0.019, 0.007, 0.033, -0.033, 0.019, 0.000, 0.031, 0.065, 0.046, 0.051, 0.047, -0.072, 0.063, -0.014, -0.048, 0.006, -0.034, 0.001, 0.009, -0.073, 0.003, -0.012, 0.052, 0.004, 0.008, 0.016, -0.027, -0.003, -0.015, -0.010, 0.062, 0.068, -0.017, 0.016, -0.012, 0.021, 0.044, -0.050, 0.030, -0.021, -0.098, 0.007, 0.032, -0.020, -0.003, 0.078, -0.015, -0.020, -0.039, -0.071, 0.037, -0.037, -0.050, -0.047, 0.018, 0.019, -0.019, -0.037, 0.051, -0.021, 0.011, -0.010, -0.030, -0.019, -0.066, 0.032, -0.005, 0.041, -0.051, -0.013, -0.009, -0.014, -0.025, -0.005, -0.019, 0.031, -0.008, -0.001, 0.048, 0.080, 0.025, -0.051, 0.040, 0.009, 0.052, 0.008, -0.047, -0.074, 0.041, 0.022, 0.003, -0.040, 0.057, 0.032, 0.016, 0.054, 0.014, -0.039, -0.035, -0.009, -0.057, -0.016, 0.021, -0.002, 0.056, 0.028, -0.038, 0.004, -0.013, 0.029, -0.021, -0.019, -0.016, -0.052, -0.041, 0.004, -0.022, 0.011, -0.055, 0.015, -0.037, -0.038, 0.015, 0.014, 0.025, -0.054, -0.024, 0.029, -0.009, 0.049, -0.001, -0.022, -0.021, 0.017, -0.054, -0.006, 0.020, 0.008, 0.082}

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

// --- C++ Worker Management (Unchanged) ---
var inferenceMutex sync.Mutex
var (
	workerCmd    *exec.Cmd
	workerStdin  io.WriteCloser
	workerStdout *bufio.Reader
)

// startWorker launches the C++ inference process once and keeps stdin/stdout pipes open.
// If the worker exits, the goroutine below resets the globals so the next call
// to runInference starts a new worker. This way the backend self-heals if the
// C++ process crashes.
func startWorker() error {
	hostAppPath := "./ml_host_prod_go"
	modelPath := "./model/bert.bin"
	enclavePath := "./enclave/enclave_prod.signed.so"

	workerCmd = exec.Command(hostAppPath, modelPath, enclavePath, "--use-stdin")
	var err error
	workerStdin, err = workerCmd.StdinPipe()
	if err != nil {
		return err
	}
	stdoutPipe, err := workerCmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderrPipe, err := workerCmd.StderrPipe()
	if err != nil {
		return err
	}
	workerStdout = bufio.NewReader(stdoutPipe)
	if err := workerCmd.Start(); err != nil {
		return err
	}

	go func() {
		scanner := bufio.NewScanner(stderrPipe)
		for scanner.Scan() {
			log.Printf("[worker stderr] %s", scanner.Text())
		}
	}()

	go func(cmd *exec.Cmd, stdin io.WriteCloser, stdout io.ReadCloser) {
		if err := cmd.Wait(); err != nil {
			log.Printf("worker exited: %v", err)
		}
		stdin.Close()
		stdout.Close()
		inferenceMutex.Lock()
		workerCmd = nil
		workerStdin = nil
		workerStdout = nil
		inferenceMutex.Unlock()
	}(workerCmd, workerStdin, stdoutPipe)

	return nil
}

// runInference sends a token string to the persistent worker and reads one line of output.
func runInference(tokenString string) (string, error) {
	inferenceMutex.Lock()
	defer inferenceMutex.Unlock()

	if workerCmd == nil {
		if err := startWorker(); err != nil {
			return "", err
		}
	}

	if _, err := io.WriteString(workerStdin, tokenString+"\n"); err != nil {
		return "", err
	}
	line, err := workerStdout.ReadString('\n')
	if err != nil {
		return "", err
	}
	return line, nil
}

// --- Utility Functions ---

// Helper to write JSON errors
func writeJSONError(w http.ResponseWriter, message string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(ResponsePayload{Error: message})
}

// --- NEW Utility for Attestation Errors ---
func writeAttestationError(w http.ResponseWriter, message string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(AttestationResponse{Error: message})
}

// --- Middleware for Authentication ---

func authMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Get the JWT secret from environment variables
		jwtSecret := os.Getenv("SUPABASE_JWT_SECRET")
		if jwtSecret == "" {
			log.Println("Error: SUPABASE_JWT_SECRET environment variable not set.")
			writeJSONError(w, "Server configuration error", http.StatusInternalServerError)
			return
		}

		// Get the token from the Authorization header
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			writeJSONError(w, "Authorization header required", http.StatusUnauthorized)
			return
		}

		// The header should be in the format "Bearer <token>"
		tokenString := strings.TrimPrefix(authHeader, "Bearer ")
		if tokenString == authHeader { // No "Bearer " prefix found
			writeJSONError(w, "Invalid Authorization header format", http.StatusUnauthorized)
			return
		}

		// Parse and validate the token
		token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
			// Check the signing method
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, errors.New("unexpected signing method")
			}
			return []byte(jwtSecret), nil
		})

		if err != nil {
			log.Printf("JWT validation error: %v", err)
			writeJSONError(w, "Invalid or expired token", http.StatusUnauthorized)
			return
		}

		if !token.Valid {
			writeJSONError(w, "Invalid token", http.StatusUnauthorized)
			return
		}

		// If the token is valid, call the next handler
		next.ServeHTTP(w, r)
	}
}

// --- API Handlers ---

// --- NEW Attestation Handler ---
func handleAttestation(w http.ResponseWriter, r *http.Request) {
	log.Println("Received request for attestation evidence.")

	// Define paths to the C++ host and its dependencies
	hostAppPath := "./ml_host_prod_go"
	modelPath := "./model/bert.bin"
	enclavePath := "./enclave/enclave_prod.signed.so"

	// Execute the host application with the --attest flag
	cmd := exec.Command(hostAppPath, modelPath, enclavePath, "--attest")
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Attestation host process failed: %v\nOutput: %s", err, string(output))
		writeAttestationError(w, "Failed to generate attestation evidence from host.", http.StatusInternalServerError)
		return
	}

	// The C++ host prints the hex-encoded evidence to stdout.
	evidenceHex := strings.TrimSpace(string(output))

	log.Printf("Successfully generated attestation evidence.")

	// Send the evidence back to the client
	resp := AttestationResponse{
		EvidenceHex: evidenceHex,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleInference(w http.ResponseWriter, r *http.Request) {
	var payload RequestPayload
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		writeJSONError(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// --- 1. Tokenization ---
	tokenizerDir := "./tokenizer"
	pyCmd := exec.Command("python3", "tokenize_script.py", tokenizerDir)
	pyCmd.Stdin = strings.NewReader(payload.Input)

	pyOutput, err := pyCmd.CombinedOutput()
	if err != nil {
		log.Printf("Tokenization script failed: %s", string(pyOutput))
		writeJSONError(w, "Tokenization failed", http.StatusInternalServerError)
		return
	}
	// Split output by newline and take the last non-empty line.
	lines := strings.Split(string(pyOutput), "\n")
	tokenString := ""
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line != "" {
			tokenString = line
			break
		}
	}
	tokenString = strings.TrimSpace(tokenString)

	// --- 2. Inference via C++ Worker ---
	resultString, err := runInference(tokenString)
	if err != nil {
		log.Printf("Inference process failed: %v", err)
		writeJSONError(w, "Failed to run inference", http.StatusInternalServerError)
		return
	}

	// --- 3. Process the output ---
	// The C++ process might print "bert_load_from_file: using CPU backend" on its first run.
	// We need to find the line that actually contains the comma-separated embeddings.
	var embeddingLine string
	scanner := bufio.NewScanner(strings.NewReader(resultString))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, ",") {
			embeddingLine = line
			break
		}
	}
	if embeddingLine == "" {
		log.Printf("Could not find embedding vector in C++ output: %s", resultString)
		writeJSONError(w, "Failed to parse inference output", http.StatusInternalServerError)
		return
	}

	outputParts := strings.Split(strings.TrimSpace(embeddingLine), ",")
	embeddings := make([]float32, len(outputParts))
	for i, part := range outputParts {
		val, _ := strconv.ParseFloat(strings.TrimSpace(part), 64)
		embeddings[i] = float32(val)
	}

	// --- 4. Classify Sentiment ---
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

// --- Main Function ---

func main() {
	// Start the persistent C++ worker process for inference
	if err := startWorker(); err != nil {
		log.Fatalf("failed to start worker: %v", err)
	}

	// Create a new ServeMux to register handlers
	mux := http.NewServeMux()

	// --- REGISTER NEW ATTESTATION ENDPOINT ---
	// This endpoint is public and does not require JWT authentication.
	mux.HandleFunc("/api/attest", handleAttestation)

	// The inference endpoint remains protected by the auth middleware.
	mux.HandleFunc("/api/analyze", authMiddleware(handleInference))

	allowedOrigins := []string{"http://localhost:3000"}
	if envOrigin := strings.TrimSpace(os.Getenv("ALLOWED_ORIGIN")); envOrigin != "" {
		allowedOrigins = append(allowedOrigins, envOrigin)
	}

	c := cors.New(cors.Options{
		// Add the public IP of the frontend here via ALLOWED_ORIGIN env var
		AllowedOrigins:   allowedOrigins,
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"Authorization", "Content-Type"},
		AllowCredentials: true,
		Debug:            true,
	})

	handler := c.Handler(mux)

	// Create a server with a timeout
	server := &http.Server{
		Addr:         ":8080",
		Handler:      handler,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	log.Println("Starting server on :8080...")
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
