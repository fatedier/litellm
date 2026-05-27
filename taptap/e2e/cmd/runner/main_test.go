package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestDefaultJobsIsFour(t *testing.T) {
	if defaultJobs != 4 {
		t.Fatalf("defaultJobs = %d, want 4", defaultJobs)
	}
}

func TestParsePathTokensSupportsQuotedObjectKeys(t *testing.T) {
	tests := []struct {
		name string
		path string
		want []any
	}{
		{
			name: "array index",
			path: "content[1].type",
			want: []any{"content", 1, "type"},
		},
		{
			name: "quoted dotted key",
			path: `["dashscope/glm-5.2"].mode`,
			want: []any{"dashscope/glm-5.2", "mode"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := parsePathTokens(test.path)
			if len(got) != len(test.want) {
				t.Fatalf("parsePathTokens(%q) returned %#v, want %#v", test.path, got, test.want)
			}
			for index := range test.want {
				if got[index] != test.want[index] {
					t.Errorf("parsePathTokens(%q)[%d] = %#v, want %#v", test.path, index, got[index], test.want[index])
				}
			}
		})
	}
}

func TestExtractSavedValueBySourceSupportsQuotedJSONKeys(t *testing.T) {
	result := stepResult{
		ResponseJSON: map[string]any{
			"dashscope/glm-5.2": map[string]any{
				"input_cost_per_token": 0.0000014,
			},
		},
	}

	value, err := extractSavedValueBySource(`json["dashscope/glm-5.2"].input_cost_per_token`, result)
	if err != nil {
		t.Fatalf("extractSavedValueBySource returned error: %v", err)
	}
	if value != 0.0000014 {
		t.Fatalf("extractSavedValueBySource returned %#v, want %v", value, 0.0000014)
	}
}

func TestProbePNGFileCountsTransparentPixels(t *testing.T) {
	var encoded bytes.Buffer
	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	img.SetNRGBA(0, 0, color.NRGBA{R: 255, A: 0})
	img.SetNRGBA(1, 0, color.NRGBA{R: 255, A: 255})
	if err := png.Encode(&encoded, img); err != nil {
		t.Fatalf("encode PNG: %v", err)
	}

	path := t.TempDir() + "/image.png"
	if err := os.WriteFile(path, encoded.Bytes(), 0o600); err != nil {
		t.Fatalf("write PNG: %v", err)
	}
	probe, err := probePNGFile(path)
	if err != nil {
		t.Fatalf("probePNGFile returned error: %v", err)
	}
	if probe.Format != "png" || probe.Width != 2 || probe.Height != 2 || probe.TotalPixels != 4 || probe.TransparentPixels != 3 {
		t.Fatalf("probePNGFile returned %#v, want PNG 2x2 with 3 transparent pixels", probe)
	}
	if err := assertImageExpectations("image", map[string]any{
		"format": "png", "width_gte": 2, "height_gte": 2, "transparent_fraction_gte": 0.5,
	}, probe); err != nil {
		t.Fatalf("assertImageExpectations returned error: %v", err)
	}
}

func TestRunnerRunCasesUsesBoundedConcurrencyAndStableOrder(t *testing.T) {
	var active atomic.Int32
	var maximum atomic.Int32

	r := runner{
		jobs: 2,
		runCaseFunc: func(currentCase e2eCase, output io.Writer) string {
			current := active.Add(1)
			for {
				previous := maximum.Load()
				if current <= previous || maximum.CompareAndSwap(previous, current) {
					break
				}
			}

			fmt.Fprintf(output, "case=%s\n", currentCase.ID)
			time.Sleep(20 * time.Millisecond)
			active.Add(-1)
			return "passed"
		},
	}

	results := r.runCases([]e2eCase{{ID: "first"}, {ID: "second"}, {ID: "third"}})
	if maximum.Load() != 2 {
		t.Fatalf("maximum concurrent cases = %d, want 2", maximum.Load())
	}
	if len(results) != 3 {
		t.Fatalf("runCases returned %d results, want 3", len(results))
	}
	for index, wantID := range []string{"first", "second", "third"} {
		if results[index].index != index {
			t.Errorf("result %d has index %d, want %d", index, results[index].index, index)
		}
		if results[index].output != fmt.Sprintf("case=%s\n", wantID) {
			t.Errorf("result %d output = %q, want case output for %s", index, results[index].output, wantID)
		}
	}
}

func TestRunnerRunCasesFailFastStopsDispatchingNewCases(t *testing.T) {
	var startedMu sync.Mutex
	started := map[string]bool{}

	r := runner{
		jobs:     1,
		failFast: true,
		runCaseFunc: func(currentCase e2eCase, output io.Writer) string {
			startedMu.Lock()
			started[currentCase.ID] = true
			startedMu.Unlock()
			if currentCase.ID == "first" {
				return "failed"
			}
			return "passed"
		},
	}

	results := r.runCases([]e2eCase{{ID: "first"}, {ID: "second"}, {ID: "third"}})
	if len(results) != 1 {
		t.Fatalf("runCases returned %d results after fail-fast, want 1", len(results))
	}
	startedMu.Lock()
	secondStarted := started["second"]
	thirdStarted := started["third"]
	startedMu.Unlock()
	if secondStarted || thirdStarted {
		t.Fatalf("fail-fast dispatched cases after the first failure: started=%v", started)
	}
}
