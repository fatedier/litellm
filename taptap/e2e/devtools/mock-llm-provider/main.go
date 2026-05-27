package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type responsesRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"`
}

type serverConfig struct {
	name      string
	addr      string
	status    int
	failFirst int64
}

type mockServer struct {
	config       serverConfig
	requestCount atomic.Int64
	mu           sync.Mutex
	events       []map[string]any
}

func main() {
	config := parseFlags()
	server := &mockServer{config: config}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", server.handleHealth)
	mux.HandleFunc("/events", server.handleEvents)
	mux.HandleFunc("/reset", server.handleReset)
	mux.HandleFunc("/responses", server.handleResponses)
	mux.HandleFunc("/v1/responses", server.handleResponses)

	httpServer := &http.Server{
		Addr:              config.addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf("mock llm provider %s listening on %s", config.name, config.addr)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func parseFlags() serverConfig {
	var config serverConfig
	flag.StringVar(&config.name, "name", "mock", "provider name printed in request logs")
	flag.StringVar(&config.addr, "addr", ":18081", "listen address")
	flag.IntVar(&config.status, "status", http.StatusOK, "HTTP status to return when fail-first is not active")
	flag.Int64Var(&config.failFirst, "fail-first", 0, "number of initial requests that should return 503")
	flag.Parse()

	if config.status < 100 || config.status > 599 {
		fmt.Fprintln(os.Stderr, "--status must be a valid HTTP status code")
		os.Exit(2)
	}
	if config.failFirst < 0 {
		fmt.Fprintln(os.Stderr, "--fail-first must be >= 0")
		os.Exit(2)
	}
	return config
}

func (s *mockServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET is supported")
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
		"name":   s.config.name,
		"count":  s.requestCount.Load(),
	})
}

func (s *mockServer) handleEvents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET is supported")
		return
	}

	s.mu.Lock()
	events := make([]map[string]any, len(s.events))
	copy(events, s.events)
	s.mu.Unlock()

	body := map[string]any{
		"name":   s.config.name,
		"count":  s.requestCount.Load(),
		"events": events,
	}
	if len(events) > 0 {
		body["last_event"] = events[len(events)-1]
	}
	writeJSON(w, http.StatusOK, body)
}

func (s *mockServer) handleReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is supported")
		return
	}

	s.requestCount.Store(0)
	s.mu.Lock()
	s.events = nil
	s.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
		"name":   s.config.name,
		"count":  0,
	})
}

func (s *mockServer) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is supported")
		return
	}

	count := s.requestCount.Add(1)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	var request responsesRequest
	if len(body) > 0 {
		if err := json.Unmarshal(body, &request); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_json", err.Error())
			return
		}
	}

	requestID := r.Header.Get("x-request-id")
	if requestID == "" {
		requestID = r.Header.Get("x-litellm-call-id")
	}

	logEvent := map[string]any{
		"provider":            s.config.name,
		"path":                r.URL.Path,
		"method":              r.Method,
		"count":               count,
		"model":               request.Model,
		"request_id":          requestID,
		"remote":              r.RemoteAddr,
		"user_agent":          r.Header.Get("User-Agent"),
		"x_client_version":    r.Header.Get("X-Client-Version"),
		"x_not_allowlisted":   r.Header.Get("X-Not-Allowlisted"),
		"x_provider_required": r.Header.Get("X-Provider-Required"),
	}
	s.recordEvent(logEvent)
	writeLogEvent(logEvent)

	status := s.config.status
	if count <= s.config.failFirst {
		status = http.StatusServiceUnavailable
	}
	if status >= 400 {
		writeError(w, status, "mock_provider_error", http.StatusText(status))
		return
	}

	writeJSON(w, status, map[string]any{
		"id":                  "resp_" + randomHex(12),
		"object":              "response",
		"created_at":          time.Now().Unix(),
		"status":              "completed",
		"model":               request.Model,
		"output":              []any{},
		"parallel_tool_calls": true,
		"tool_choice":         "auto",
		"tools":               []any{},
		"usage": map[string]any{
			"input_tokens":  1,
			"output_tokens": 1,
			"total_tokens":  2,
		},
	})
}

func (s *mockServer) recordEvent(event map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, event)
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(body); err != nil {
		log.Printf("failed to write response: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, code string, message string) {
	if strings.TrimSpace(message) == "" {
		message = http.StatusText(status)
	}
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    "mock_error",
			"param":   nil,
			"code":    code,
		},
	})
}

func writeLogEvent(event map[string]any) {
	data, err := json.Marshal(event)
	if err != nil {
		log.Printf("failed to marshal log event: %v", err)
		return
	}
	fmt.Println(string(data))
}

func randomHex(size int) string {
	bytes := make([]byte, size)
	if _, err := rand.Read(bytes); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(bytes)
}
