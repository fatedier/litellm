package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"image/png"
	"io"
	"math/rand"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const defaultJobs = 4

var envPattern = regexp.MustCompile(`\$\{([^}]+)\}`)

type multiFlag []string

func (m *multiFlag) String() string {
	return strings.Join(*m, ",")
}

func (m *multiFlag) Set(value string) error {
	*m = append(*m, value)
	return nil
}

type options struct {
	BaseURL   string
	CasesDir  string
	Cases     multiFlag
	Scenarios multiFlag
	PatchIDs  multiFlag
	Timeout   float64
	Jobs      int
	FailFast  bool
	List      bool
	DryRun    bool
}

type e2eError struct {
	message string
}

func (e *e2eError) Error() string {
	return e.message
}

type skipError struct {
	message string
}

func (e *skipError) Error() string {
	return e.message
}

type e2eCase struct {
	ID          string         `yaml:"id"`
	PatchID     string         `yaml:"patch_id"`
	Status      string         `yaml:"status"`
	Description string         `yaml:"description"`
	Variables   map[string]any `yaml:"variables"`
	Shared      sharedConfig   `yaml:"shared"`
	Scenarios   []scenario     `yaml:"scenarios"`
	Path        string         `yaml:"-"`
}

type sharedConfig struct {
	TimeoutSeconds float64        `yaml:"timeout_seconds"`
	Headers        map[string]any `yaml:"headers"`
}

type scenario struct {
	ID          string         `yaml:"id"`
	Description string         `yaml:"description"`
	Variables   map[string]any `yaml:"variables"`
	RequiresEnv []string       `yaml:"requires_env"`
	Steps       []step         `yaml:"steps"`
	Cleanup     []step         `yaml:"cleanup"`
}

type step struct {
	ID                    string         `yaml:"id"`
	RequiresEnv           []string       `yaml:"requires_env"`
	Request               map[string]any `yaml:"request"`
	Wait                  map[string]any `yaml:"wait"`
	MediaProbe            map[string]any `yaml:"media_probe"`
	ImageProbe            map[string]any `yaml:"image_probe"`
	FrameExtract          map[string]any `yaml:"frame_extract"`
	CostEstimate          map[string]any `yaml:"cost_estimate"`
	NovaDiscountDiscovery map[string]any `yaml:"nova_discount_discovery"`
	CostDiscountAssertion map[string]any `yaml:"cost_discount_assertion"`
	Expect                map[string]any `yaml:"expect"`
	Save                  []saveItem     `yaml:"save"`
}

type saveItem struct {
	From           string   `yaml:"from"`
	FromAny        []string `yaml:"from_any"`
	DecodeBase64To string   `yaml:"decode_base64_to"`
	As             string   `yaml:"as"`
}

type runner struct {
	baseURL        string
	defaultTimeout time.Duration
	jobs           int
	failFast       bool
	dryRun         bool
	client         *http.Client
	output         io.Writer
	runCaseFunc    func(e2eCase, io.Writer) string
}

type caseRunResult struct {
	index  int
	status string
	output string
}

type stepResult struct {
	StatusCode    int
	Body          string
	BodyBytes     []byte
	Headers       http.Header
	ResponseJSON  any
	SavedBodyPath string
	MediaProbe    *mediaProbeResult
	ImageProbe    *imageProbeResult
	FrameOutputs  map[string]frameExtractOutput
	CostEstimate  map[string]any
	Values        map[string]any
}

type mediaProbeResult struct {
	Format  mediaProbeFormat   `json:"format"`
	Streams []mediaProbeStream `json:"streams"`
}

type imageProbeResult struct {
	Format            string `json:"format"`
	Width             int    `json:"width"`
	Height            int    `json:"height"`
	TotalPixels       int    `json:"total_pixels"`
	TransparentPixels int    `json:"transparent_pixels"`
}

type mediaProbeFormat struct {
	Filename        string  `json:"filename"`
	DurationSeconds float64 `json:"duration_seconds"`
}

type mediaProbeStream struct {
	CodecType       string  `json:"codec_type"`
	Width           int     `json:"width"`
	Height          int     `json:"height"`
	DurationSeconds float64 `json:"duration_seconds"`
}

type frameExtractOutput struct {
	Path      string  `json:"path"`
	AtSeconds float64 `json:"at_seconds"`
}

func main() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: %s [flags]\n", os.Args[0])
		flag.PrintDefaults()
	}

	opts := parseFlags()

	caseFiles, err := discoverCaseFiles(opts.CasesDir, opts.Cases)
	if err != nil {
		exitError(err)
	}

	cases, err := loadCases(caseFiles)
	if err != nil {
		exitError(err)
	}

	filteredCases := filterCases(cases, toSet(opts.PatchIDs), toSet(opts.Scenarios))
	if opts.List {
		printCaseListing(filteredCases)
		return
	}

	if opts.BaseURL == "" {
		exitError(&e2eError{message: "base URL is required. Pass --base-url or set TAPTAP_E2E_BASE_URL."})
	}
	if opts.Jobs < 1 {
		exitError(&e2eError{message: "jobs must be at least 1"})
	}

	r := runner{
		baseURL:        strings.TrimRight(opts.BaseURL, "/") + "/",
		defaultTimeout: time.Duration(opts.Timeout * float64(time.Second)),
		jobs:           opts.Jobs,
		failFast:       opts.FailFast,
		dryRun:         opts.DryRun,
		client:         &http.Client{},
		output:         os.Stdout,
	}

	os.Exit(r.run(filteredCases))
}

func parseFlags() options {
	opts := options{}
	flag.StringVar(&opts.BaseURL, "base-url", os.Getenv("TAPTAP_E2E_BASE_URL"), "Base URL for the deployed test environment. Defaults to TAPTAP_E2E_BASE_URL.")
	flag.StringVar(&opts.CasesDir, "cases-dir", defaultCasesDir(), "Directory containing YAML case files.")
	flag.Var(&opts.Cases, "case", "Specific case file path or case id to run. Can be passed multiple times.")
	flag.Var(&opts.Scenarios, "scenario", "Only run specific scenario ids. Can be passed multiple times.")
	flag.Var(&opts.PatchIDs, "patch-id", "Only run cases for the given patch id. Can be passed multiple times.")
	flag.Float64Var(&opts.Timeout, "timeout", 30.0, "Default request timeout in seconds when not set in YAML.")
	flag.IntVar(&opts.Jobs, "jobs", defaultJobs, "Maximum number of cases to run concurrently. Defaults to 4.")
	flag.BoolVar(&opts.FailFast, "fail-fast", false, "Stop the current case after the first failure and stop dispatching new cases after a failed case.")
	flag.BoolVar(&opts.List, "list", false, "List discovered cases and scenarios without executing them.")
	flag.BoolVar(&opts.DryRun, "dry-run", false, "Resolve and print requests without sending them.")
	flag.Parse()
	return opts
}

func defaultCasesDir() string {
	candidates := []string{
		filepath.Join("taptap", "e2e", "cases"),
		"cases",
	}
	for _, candidate := range candidates {
		info, err := os.Stat(candidate)
		if err == nil && info.IsDir() {
			return candidate
		}
	}
	return filepath.Join("taptap", "e2e", "cases")
}

func discoverCaseFiles(casesDir string, selectedCases []string) ([]string, error) {
	info, err := os.Stat(casesDir)
	if err != nil || !info.IsDir() {
		return nil, &e2eError{message: fmt.Sprintf("cases directory does not exist: %s", casesDir)}
	}

	entries, err := os.ReadDir(casesDir)
	if err != nil {
		return nil, err
	}

	available := map[string]string{}
	var all []string
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".yaml") {
			continue
		}
		path := filepath.Join(casesDir, entry.Name())
		all = append(all, path)
		available[strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))] = path
	}
	sort.Strings(all)
	if len(selectedCases) == 0 {
		return all, nil
	}

	var resolved []string
	for _, rawCase := range selectedCases {
		if _, err := os.Stat(rawCase); err == nil {
			resolved = append(resolved, rawCase)
			continue
		}
		if path, ok := available[rawCase]; ok {
			resolved = append(resolved, path)
			continue
		}
		return nil, &e2eError{message: fmt.Sprintf("could not resolve case %q", rawCase)}
	}
	return resolved, nil
}

func loadCases(caseFiles []string) ([]e2eCase, error) {
	var cases []e2eCase
	for _, caseFile := range caseFiles {
		content, err := os.ReadFile(caseFile)
		if err != nil {
			return nil, err
		}
		var currentCase e2eCase
		if err := yaml.Unmarshal(content, &currentCase); err != nil {
			return nil, &e2eError{message: fmt.Sprintf("failed to parse %s: %v", caseFile, err)}
		}
		currentCase.Path = caseFile
		cases = append(cases, currentCase)
	}
	return cases, nil
}

func filterCases(cases []e2eCase, patchIDs map[string]struct{}, scenarioIDs map[string]struct{}) []e2eCase {
	var filtered []e2eCase
	for _, currentCase := range cases {
		if len(patchIDs) > 0 {
			if _, ok := patchIDs[currentCase.PatchID]; !ok {
				continue
			}
		}
		if len(scenarioIDs) > 0 {
			var selected []scenario
			for _, currentScenario := range currentCase.Scenarios {
				if _, ok := scenarioIDs[currentScenario.ID]; ok {
					selected = append(selected, currentScenario)
				}
			}
			if len(selected) == 0 {
				continue
			}
			currentCase.Scenarios = selected
		}
		filtered = append(filtered, currentCase)
	}
	return filtered
}

func printCaseListing(cases []e2eCase) {
	for _, currentCase := range cases {
		fmt.Printf("%s  patch=%s  file=%s\n", currentCase.ID, currentCase.PatchID, currentCase.Path)
		for _, currentScenario := range currentCase.Scenarios {
			fmt.Printf("  - %s\n", currentScenario.ID)
		}
	}
}

func (r *runner) run(cases []e2eCase) int {
	results := r.runCases(cases)
	summary := map[string]int{
		"passed":  0,
		"failed":  0,
		"skipped": 0,
	}

	for _, result := range results {
		r.printf("%s", result.output)
		summary[result.status]++
	}

	r.printf("\nSummary: passed=%d failed=%d skipped=%d\n", summary["passed"], summary["failed"], summary["skipped"])
	if summary["failed"] > 0 {
		return 1
	}
	return 0
}

func (r *runner) runCases(cases []e2eCase) []caseRunResult {
	if len(cases) == 0 {
		return nil
	}

	jobs := r.jobs
	if jobs < 1 {
		jobs = 1
	}
	if jobs > len(cases) {
		jobs = len(cases)
	}

	resultsChannel := make(chan caseRunResult, jobs)
	nextIndex := 0
	activeJobs := 0
	dispatch := func() {
		for activeJobs < jobs && nextIndex < len(cases) {
			index := nextIndex
			currentCase := cases[index]
			nextIndex++
			activeJobs++
			go func(index int, currentCase e2eCase) {
				resultsChannel <- r.runCaseResult(index, currentCase)
			}(index, currentCase)
		}
	}

	dispatch()
	results := make([]caseRunResult, 0, len(cases))
	for activeJobs > 0 {
		result := <-resultsChannel
		activeJobs--
		results = append(results, result)
		if !(r.failFast && result.status == "failed") {
			dispatch()
		}
	}

	sort.Slice(results, func(left, right int) bool {
		return results[left].index < results[right].index
	})
	return results
}

func (r *runner) runCaseResult(index int, currentCase e2eCase) caseRunResult {
	var output bytes.Buffer
	var status string
	if r.runCaseFunc != nil {
		status = r.runCaseFunc(currentCase, &output)
	} else {
		caseRunner := *r
		caseRunner.output = &output
		status = caseRunner.runCase(currentCase)
	}
	return caseRunResult{index: index, status: status, output: output.String()}
}

func (r *runner) runCase(currentCase e2eCase) string {
	r.printf("\nCase: %s (%s)\n", currentCase.ID, currentCase.PatchID)

	caseFailed := false
	caseRan := false
	for _, currentScenario := range currentCase.Scenarios {
		result := r.runScenario(currentCase, currentScenario)
		if result == "failed" {
			caseFailed = true
		}
		if result == "passed" {
			caseRan = true
		}
		if result == "failed" && r.failFast {
			break
		}
	}

	if caseFailed {
		return "failed"
	}
	if caseRan {
		return "passed"
	}
	return "skipped"
}

func (r *runner) runScenario(currentCase e2eCase, currentScenario scenario) string {
	missingEnv := missingRequiredEnv(currentScenario.RequiresEnv)
	if len(missingEnv) > 0 {
		r.printf("  SKIP %s: missing env %s\n", currentScenario.ID, strings.Join(missingEnv, ", "))
		return "skipped"
	}

	r.printf("  Scenario: %s\n", currentScenario.ID)

	context := map[string]any{
		"base_url": strings.TrimRight(r.baseURL, "/"),
		"run_id":   generateRunID(currentCase.ID, currentScenario.ID),
		"utc_date": time.Now().UTC().Format("2006-01-02"),
	}
	artifactsDir := filepath.Join(os.TempDir(), "taptap-e2e", currentCase.ID, context["run_id"].(string))
	if err := os.MkdirAll(artifactsDir, 0o755); err != nil {
		r.printf("    FAIL artifacts dir: %v\n", err)
		return "failed"
	}
	context["artifacts_dir"] = artifactsDir

	caseVariables, err := resolveToMap(currentCase.Variables, context)
	if err != nil {
		r.printf("    FAIL case variables: %v\n", err)
		return "failed"
	}
	mergeMap(context, caseVariables)

	scenarioVariables, err := resolveToMap(currentScenario.Variables, context)
	if err != nil {
		r.printf("    FAIL scenario variables: %v\n", err)
		return "failed"
	}

	mergeMap(context, scenarioVariables)
	savedValues := map[string]any{}

	scenarioFailed := false
	scenarioSkipped := false
	for _, currentStep := range currentScenario.Steps {
		if err := r.runStep(currentStep, currentCase.Shared, context, savedValues); err != nil {
			if skip, ok := err.(*skipError); ok {
				r.printf("    SKIP %s: %v\n", currentStep.ID, skip)
				scenarioSkipped = true
				break
			}
			r.printf("    FAIL %s: %v\n", currentStep.ID, err)
			scenarioFailed = true
			if r.failFast {
				break
			}
		}
	}

	if !scenarioSkipped {
		for _, cleanupStep := range currentScenario.Cleanup {
			if err := r.runStep(cleanupStep, currentCase.Shared, context, savedValues); err != nil {
				r.printf("    CLEANUP FAIL %s: %v\n", cleanupStep.ID, err)
				scenarioFailed = true
			}
		}
	}

	if scenarioFailed {
		return "failed"
	}
	if scenarioSkipped {
		return "skipped"
	}
	return "passed"
}

func (r *runner) printf(format string, args ...any) {
	output := r.output
	if output == nil {
		output = os.Stdout
	}
	_, _ = fmt.Fprintf(output, format, args...)
}

func (r *runner) runStep(currentStep step, shared sharedConfig, context map[string]any, savedValues map[string]any) error {
	missingEnv := missingRequiredEnv(currentStep.RequiresEnv)
	if len(missingEnv) > 0 {
		return &e2eError{message: fmt.Sprintf("missing env for step %s: %s", currentStep.ID, strings.Join(missingEnv, ", "))}
	}

	stepContext := copyMap(context)
	mergeMap(stepContext, savedValues)

	var result stepResult
	var err error

	switch {
	case len(currentStep.Request) > 0:
		result, err = r.runHTTPRequestStep(currentStep.ID, currentStep.Request, currentStep.Expect, shared, stepContext)
	case len(currentStep.Wait) > 0:
		result, err = r.runWaitStep(currentStep.ID, currentStep.Wait, stepContext)
	case len(currentStep.MediaProbe) > 0:
		result, err = r.runMediaProbeStep(currentStep.ID, currentStep.MediaProbe, currentStep.Expect, stepContext)
	case len(currentStep.ImageProbe) > 0:
		result, err = r.runImageProbeStep(currentStep.ID, currentStep.ImageProbe, currentStep.Expect, stepContext)
	case len(currentStep.FrameExtract) > 0:
		result, err = r.runFrameExtractStep(currentStep.ID, currentStep.FrameExtract, currentStep.Expect, stepContext)
	case len(currentStep.CostEstimate) > 0:
		result, err = r.runCostEstimateStep(currentStep.ID, currentStep.CostEstimate, currentStep.Expect, stepContext)
	case len(currentStep.NovaDiscountDiscovery) > 0:
		result, err = r.runNovaDiscountDiscoveryStep(currentStep.ID, currentStep.NovaDiscountDiscovery, stepContext)
	case len(currentStep.CostDiscountAssertion) > 0:
		result, err = r.runCostDiscountAssertionStep(currentStep.ID, currentStep.CostDiscountAssertion, stepContext)
	default:
		return &e2eError{message: fmt.Sprintf("step %s must define request, wait, media_probe, image_probe, frame_extract, cost_estimate, nova_discount_discovery, or cost_discount_assertion", currentStep.ID)}
	}
	if err != nil {
		return err
	}

	if err := persistSavedValues(currentStep.Save, result, savedValues, stepContext); err != nil {
		return err
	}
	return nil
}

func (r *runner) runWaitStep(stepID string, waitSpec map[string]any, context map[string]any) (stepResult, error) {
	resolvedWait, err := resolveToMap(waitSpec, context)
	if err != nil {
		return stepResult{}, err
	}

	seconds := getFloatWithDefault(resolvedWait, "seconds", 0)
	if seconds <= 0 {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: wait.seconds must be > 0", stepID)}
	}

	r.printf("    STEP %s: wait %.0fs\n", stepID, seconds)
	if r.dryRun {
		return stepResult{}, nil
	}

	time.Sleep(time.Duration(seconds * float64(time.Second)))
	return stepResult{}, nil
}

func (r *runner) runHTTPRequestStep(stepID string, request map[string]any, expect map[string]any, shared sharedConfig, stepContext map[string]any) (stepResult, error) {
	requestSpec := map[string]any{
		"headers":         copyMapAny(shared.Headers),
		"timeout_seconds": shared.TimeoutSeconds,
	}
	requestSpec = deepMergeMaps(requestSpec, request)

	resolvedRequest, err := resolveToMap(requestSpec, stepContext)
	if err != nil {
		return stepResult{}, err
	}

	method := getStringWithDefault(resolvedRequest, "method", http.MethodGet)
	requestURL, err := buildRequestURL(r.baseURL, resolvedRequest)
	if err != nil {
		return stepResult{}, err
	}

	r.printf("    STEP %s: %s %s\n", stepID, method, requestURL)
	if r.dryRun {
		return stepResult{}, nil
	}

	result, err := r.executeHTTPRequest(stepID, resolvedRequest, expect, stepContext)
	if err != nil {
		return stepResult{}, err
	}

	if savePath, ok := resolvedRequest["save_body_to"]; ok {
		writtenPath, err := writeResponseBodyToFile(fmt.Sprint(savePath), stepContext, result.BodyBytes)
		if err != nil {
			return stepResult{}, err
		}
		result.SavedBodyPath = writtenPath
	}

	return result, nil
}

func (r *runner) executeHTTPRequest(stepID string, requestSpec map[string]any, expect map[string]any, context map[string]any) (stepResult, error) {
	timeoutSeconds := getFloatWithDefault(requestSpec, "timeout_seconds", r.defaultTimeout.Seconds())
	pollSpec := toMap(requestSpec["poll"])

	if len(pollSpec) == 0 {
		result, err := r.performHTTPRequest(requestSpec, timeoutSeconds)
		if err != nil {
			return stepResult{}, err
		}
		if err := assertResponse(stepID, expect, context, result); err != nil {
			return stepResult{}, err
		}
		return result, nil
	}

	pollTimeout := getFloatWithDefault(pollSpec, "timeout_seconds", timeoutSeconds)
	intervalSeconds := getFloatWithDefault(pollSpec, "interval_seconds", 5.0)
	deadline := time.Now().Add(time.Duration(pollTimeout * float64(time.Second)))

	for attempt := 1; ; attempt++ {
		result, err := r.performHTTPRequest(requestSpec, timeoutSeconds)
		if err != nil {
			return stepResult{}, err
		}
		if err := assertResponse(stepID, expect, context, result); err != nil {
			return stepResult{}, err
		}

		failureMatched, failureValue, err := matchPollCondition(result.ResponseJSON, toMap(pollSpec["failure"]))
		if err != nil {
			return stepResult{}, err
		}
		if failureMatched {
			return stepResult{}, &e2eError{message: fmt.Sprintf("%s: poll reached failure state %q", stepID, failureValue)}
		}

		successMatched, successValue, err := matchPollCondition(result.ResponseJSON, toMap(pollSpec["success"]))
		if err != nil {
			return stepResult{}, err
		}
		if successMatched {
			r.printf("      poll success after %d attempt(s): %s\n", attempt, successValue)
			return result, nil
		}

		if time.Now().After(deadline) {
			return stepResult{}, &e2eError{message: fmt.Sprintf("%s: poll timed out after %.1fs", stepID, pollTimeout)}
		}
		time.Sleep(time.Duration(intervalSeconds * float64(time.Second)))
	}
}

func (r *runner) performHTTPRequest(requestSpec map[string]any, timeoutSeconds float64) (stepResult, error) {
	method := getStringWithDefault(requestSpec, "method", http.MethodGet)
	requestURL, err := buildRequestURL(r.baseURL, requestSpec)
	if err != nil {
		return stepResult{}, err
	}

	req, err := buildHTTPRequest(method, requestURL, requestSpec)
	if err != nil {
		return stepResult{}, err
	}

	client := *r.client
	client.Timeout = time.Duration(timeoutSeconds * float64(time.Second))

	resp, err := client.Do(req)
	if err != nil {
		return stepResult{}, err
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return stepResult{}, err
	}

	result := stepResult{
		StatusCode: resp.StatusCode,
		Body:       string(bodyBytes),
		BodyBytes:  bodyBytes,
		Headers:    resp.Header.Clone(),
	}
	if parsedJSON, ok := tryParseJSON(bodyBytes); ok {
		result.ResponseJSON = parsedJSON
	}

	return result, nil
}

func (r *runner) runMediaProbeStep(stepID string, mediaProbe map[string]any, expect map[string]any, context map[string]any) (stepResult, error) {
	resolvedProbe, err := resolveToMap(mediaProbe, context)
	if err != nil {
		return stepResult{}, err
	}

	path := getStringWithDefault(resolvedProbe, "path", "")
	if path == "" {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: media_probe.path is required", stepID)}
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(fmt.Sprint(context["artifacts_dir"]), path)
	}

	r.printf("    STEP %s: ffprobe %s\n", stepID, path)
	if r.dryRun {
		return stepResult{}, nil
	}

	probe, err := probeMediaFile(path)
	if err != nil {
		return stepResult{}, err
	}

	result := stepResult{
		MediaProbe: probe,
	}
	if err := assertResponse(stepID, expect, context, result); err != nil {
		return stepResult{}, err
	}
	return result, nil
}

func (r *runner) runImageProbeStep(stepID string, imageProbe map[string]any, expect map[string]any, context map[string]any) (stepResult, error) {
	resolvedProbe, err := resolveToMap(imageProbe, context)
	if err != nil {
		return stepResult{}, err
	}

	path := getStringWithDefault(resolvedProbe, "path", "")
	if path == "" {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: image_probe.path is required", stepID)}
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(fmt.Sprint(context["artifacts_dir"]), path)
	}

	r.printf("    STEP %s: inspect image %s\n", stepID, path)
	if r.dryRun {
		return stepResult{}, nil
	}

	probe, err := probePNGFile(path)
	if err != nil {
		return stepResult{}, err
	}

	result := stepResult{ImageProbe: probe}
	if err := assertResponse(stepID, expect, context, result); err != nil {
		return stepResult{}, err
	}
	return result, nil
}

func (r *runner) runFrameExtractStep(stepID string, frameExtract map[string]any, expect map[string]any, context map[string]any) (stepResult, error) {
	resolvedSpec, err := resolveToMap(frameExtract, context)
	if err != nil {
		return stepResult{}, err
	}

	inputPath := getStringWithDefault(resolvedSpec, "path", "")
	if inputPath == "" {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: frame_extract.path is required", stepID)}
	}
	if !filepath.IsAbs(inputPath) {
		inputPath = filepath.Join(fmt.Sprint(context["artifacts_dir"]), inputPath)
	}

	outputItems := toSlice(resolvedSpec["outputs"])
	if len(outputItems) == 0 {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: frame_extract.outputs is required", stepID)}
	}

	r.printf("    STEP %s: extract frames from %s\n", stepID, inputPath)
	if r.dryRun {
		return stepResult{}, nil
	}

	frameOutputs := map[string]frameExtractOutput{}
	for _, rawOutput := range outputItems {
		outputSpec := toMap(rawOutput)
		name := getStringWithDefault(outputSpec, "name", "")
		if name == "" {
			return stepResult{}, &e2eError{message: fmt.Sprintf("%s: each frame_extract output needs a name", stepID)}
		}
		atSeconds := getFloatWithDefault(outputSpec, "at_seconds", -1)
		if atSeconds < 0 {
			return stepResult{}, &e2eError{message: fmt.Sprintf("%s: output %q needs non-negative at_seconds", stepID, name)}
		}
		filename := getStringWithDefault(outputSpec, "filename", "")
		if filename == "" {
			filename = fmt.Sprintf("%s.png", name)
		}
		outputPath := filename
		if !filepath.IsAbs(outputPath) {
			outputPath = filepath.Join(fmt.Sprint(context["artifacts_dir"]), outputPath)
		}
		if err := extractVideoFrame(inputPath, outputPath, atSeconds); err != nil {
			return stepResult{}, err
		}
		frameOutputs[name] = frameExtractOutput{
			Path:      outputPath,
			AtSeconds: atSeconds,
		}
	}

	return stepResult{FrameOutputs: frameOutputs}, nil
}

func (r *runner) runCostEstimateStep(stepID string, costEstimate map[string]any, expect map[string]any, context map[string]any) (stepResult, error) {
	resolvedSpec, err := resolveToMap(costEstimate, context)
	if err != nil {
		return stepResult{}, err
	}

	r.printf("    STEP %s: estimate cost\n", stepID)
	if r.dryRun {
		return stepResult{}, nil
	}

	usage := toMap(resolvedSpec["usage"])
	pricing := toMap(resolvedSpec["pricing"])
	tolerance := toMap(resolvedSpec["tolerance"])

	inputTextTokens, err := toFloat(usage["input_text_tokens"])
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid usage.input_text_tokens: %v", stepID, err)}
	}
	inputSeconds, err := toFloatDefault(usage["input_seconds"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid usage.input_seconds: %v", stepID, err)}
	}
	inputCharacters, err := toFloatDefault(usage["input_characters"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid usage.input_characters: %v", stepID, err)}
	}
	cacheCreationInputTokens, err := toFloatDefault(usage["cache_creation_input_tokens"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid usage.cache_creation_input_tokens: %v", stepID, err)}
	}
	cacheReadInputTokens, err := toFloatDefault(usage["cache_read_input_tokens"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid usage.cache_read_input_tokens: %v", stepID, err)}
	}
	inputImageTokens, err := toFloatDefault(usage["input_image_tokens"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid usage.input_image_tokens: %v", stepID, err)}
	}
	outputTextTokens, err := toFloatDefault(usage["output_text_tokens"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid usage.output_text_tokens: %v", stepID, err)}
	}
	outputImageTokens, err := toFloatDefault(usage["output_image_tokens"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid usage.output_image_tokens: %v", stepID, err)}
	}

	inputCostPerToken, err := toFloat(pricing["input_cost_per_token"])
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid pricing.input_cost_per_token: %v", stepID, err)}
	}
	inputCostPerSecond, err := toFloatDefault(pricing["input_cost_per_second"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid pricing.input_cost_per_second: %v", stepID, err)}
	}
	inputCostPerCharacter, err := toFloatDefault(pricing["input_cost_per_character"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid pricing.input_cost_per_character: %v", stepID, err)}
	}
	cacheCreationInputTokenCost, err := toFloatDefault(pricing["cache_creation_input_token_cost"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid pricing.cache_creation_input_token_cost: %v", stepID, err)}
	}
	cacheReadInputTokenCost, err := toFloatDefault(pricing["cache_read_input_token_cost"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid pricing.cache_read_input_token_cost: %v", stepID, err)}
	}
	inputCostPerImageToken, err := toFloatDefault(pricing["input_cost_per_image_token"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid pricing.input_cost_per_image_token: %v", stepID, err)}
	}
	outputCostPerToken, err := toFloatDefault(pricing["output_cost_per_token"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid pricing.output_cost_per_token: %v", stepID, err)}
	}
	outputCostPerImageToken, err := toFloatDefault(pricing["output_cost_per_image_token"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid pricing.output_cost_per_image_token: %v", stepID, err)}
	}

	expectedCost := (inputTextTokens * inputCostPerToken) +
		(inputSeconds * inputCostPerSecond) +
		(inputCharacters * inputCostPerCharacter) +
		(cacheCreationInputTokens * cacheCreationInputTokenCost) +
		(cacheReadInputTokens * cacheReadInputTokenCost) +
		(inputImageTokens * inputCostPerImageToken) +
		(outputTextTokens * outputCostPerToken) +
		(outputImageTokens * outputCostPerImageToken)

	toleranceAbsolute, err := toFloatDefault(tolerance["absolute"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid tolerance.absolute: %v", stepID, err)}
	}
	tolerancePercent, err := toFloatDefault(tolerance["percent"], 0)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid tolerance.percent: %v", stepID, err)}
	}
	toleranceValue := toleranceAbsolute
	if tolerancePercent > 0 {
		percentDelta := expectedCost * tolerancePercent
		if percentDelta > toleranceValue {
			toleranceValue = percentDelta
		}
	}

	result := stepResult{
		CostEstimate: map[string]any{
			"expected_cost":   expectedCost,
			"tolerance_value": toleranceValue,
			"min_cost":        expectedCost - toleranceValue,
			"max_cost":        expectedCost + toleranceValue,
		},
	}
	if err := assertResponse(stepID, expect, context, result); err != nil {
		return stepResult{}, err
	}
	return result, nil
}

func (r *runner) runNovaDiscountDiscoveryStep(stepID string, discountSpec map[string]any, context map[string]any) (stepResult, error) {
	resolvedSpec, err := resolveToMap(discountSpec, context)
	if err != nil {
		return stepResult{}, err
	}

	modelName := getStringWithDefault(resolvedSpec, "model_name", "")
	if modelName == "" {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: nova_discount_discovery.model_name is required", stepID)}
	}
	masterKey := getStringWithDefault(resolvedSpec, "master_key", "")
	if masterKey == "" {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: nova_discount_discovery.master_key is required", stepID)}
	}

	r.printf("    STEP %s: discover nova_cost_discount for %s\n", stepID, modelName)
	if r.dryRun {
		return stepResult{}, nil
	}

	requestSpec := map[string]any{
		"method": "GET",
		"path":   "/model/info",
		"headers": map[string]any{
			"Authorization": "Bearer " + masterKey,
		},
		"timeout_seconds": getFloatWithDefault(resolvedSpec, "timeout_seconds", r.defaultTimeout.Seconds()),
	}
	result, err := r.executeHTTPRequest(stepID, requestSpec, map[string]any{"status": 200}, context)
	if err != nil {
		return stepResult{}, err
	}

	discount, deploymentCount, err := findStableNovaDiscount(result.ResponseJSON, modelName)
	if err != nil {
		return stepResult{}, err
	}

	r.printf("      discovered nova_cost_discount=%g across %d deployment(s)\n", discount, deploymentCount)
	result.Values = map[string]any{
		"nova_cost_discount": discount,
		"model_name":         modelName,
		"deployment_count":   deploymentCount,
	}
	return result, nil
}

func (r *runner) runCostDiscountAssertionStep(stepID string, assertionSpec map[string]any, context map[string]any) (stepResult, error) {
	resolvedSpec, err := resolveToMap(assertionSpec, context)
	if err != nil {
		return stepResult{}, err
	}

	r.printf("    STEP %s: assert cost discount\n", stepID)
	if r.dryRun {
		return stepResult{}, nil
	}

	expectedDiscount, err := toFloat(resolvedSpec["expected_discount_percent"])
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid expected_discount_percent: %v", stepID, err)}
	}
	actualDiscount, err := toFloat(resolvedSpec["discount_percent"])
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid discount_percent: %v", stepID, err)}
	}
	originalCost, err := toFloat(resolvedSpec["original_cost"])
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid original_cost: %v", stepID, err)}
	}
	discountAmount, err := toFloat(resolvedSpec["discount_amount"])
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid discount_amount: %v", stepID, err)}
	}
	totalCost, err := toFloat(resolvedSpec["total_cost"])
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid total_cost: %v", stepID, err)}
	}

	tolerance := toMap(resolvedSpec["tolerance"])
	absoluteTolerance, err := toFloatDefault(tolerance["absolute"], 0.000001)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid tolerance.absolute: %v", stepID, err)}
	}
	percentTolerance, err := toFloatDefault(tolerance["percent"], 0.01)
	if err != nil {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: invalid tolerance.percent: %v", stepID, err)}
	}

	if expectedDiscount <= 0 || expectedDiscount >= 1 {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: expected_discount_percent must be between 0 and 1, got %g", stepID, expectedDiscount)}
	}
	if originalCost <= 0 {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: original_cost must be > 0, got %g", stepID, originalCost)}
	}
	if discountAmount <= 0 {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: discount_amount must be > 0, got %g", stepID, discountAmount)}
	}
	if totalCost <= 0 {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: total_cost must be > 0, got %g", stepID, totalCost)}
	}

	if !withinTolerance(actualDiscount, expectedDiscount, absoluteTolerance, percentTolerance) {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: discount_percent expected %g, got %g", stepID, expectedDiscount, actualDiscount)}
	}

	expectedDiscountAmount := originalCost * actualDiscount
	if !withinTolerance(discountAmount, expectedDiscountAmount, absoluteTolerance, percentTolerance) {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: discount_amount expected %g, got %g", stepID, expectedDiscountAmount, discountAmount)}
	}

	expectedTotalCost := originalCost * (1 - actualDiscount)
	if !withinTolerance(totalCost, expectedTotalCost, absoluteTolerance, percentTolerance) {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: total_cost expected %g, got %g", stepID, expectedTotalCost, totalCost)}
	}

	if !withinTolerance(originalCost-discountAmount, totalCost, absoluteTolerance, percentTolerance) {
		return stepResult{}, &e2eError{message: fmt.Sprintf("%s: original_cost - discount_amount expected %g, got total_cost %g", stepID, originalCost-discountAmount, totalCost)}
	}

	return stepResult{Values: map[string]any{
		"expected_total_cost":      expectedTotalCost,
		"expected_discount_amount": expectedDiscountAmount,
	}}, nil
}

func buildHTTPRequest(method, requestURL string, requestSpec map[string]any) (*http.Request, error) {
	var bodyReader io.Reader
	var multipartContentType string

	if filesValue, ok := requestSpec["files"]; ok {
		dataValue := toMap(requestSpec["data"])
		filesSpec := toMap(filesValue)
		bodyBuffer := &bytes.Buffer{}
		writer := multipart.NewWriter(bodyBuffer)

		for key, value := range dataValue {
			if err := writer.WriteField(key, fmt.Sprint(value)); err != nil {
				return nil, err
			}
		}

		for fieldName, rawSpec := range filesSpec {
			fileSpec := toMap(rawSpec)
			filePath := ""
			fileName := ""
			contentType := ""

			if specPath, ok := rawSpec.(string); ok {
				filePath = specPath
			} else {
				filePath = getStringWithDefault(fileSpec, "path", "")
				fileName = getStringWithDefault(fileSpec, "filename", "")
				contentType = getStringWithDefault(fileSpec, "content_type", "")
			}

			if filePath == "" {
				return nil, &e2eError{message: fmt.Sprintf("multipart file %q requires a path", fieldName)}
			}

			file, err := os.Open(filePath)
			if err != nil {
				return nil, err
			}

			if fileName == "" {
				fileName = filepath.Base(filePath)
			}
			if contentType == "" {
				contentType = "application/octet-stream"
			}

			partHeaders := make(textproto.MIMEHeader)
			partHeaders.Set("Content-Type", contentType)
			partHeaders.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"; filename="%s"`, escapeQuotes(fieldName), escapeQuotes(fileName)))

			part, err := writer.CreatePart(partHeaders)
			if err != nil {
				file.Close()
				return nil, err
			}
			if _, err := io.Copy(part, file); err != nil {
				file.Close()
				return nil, err
			}
			file.Close()
		}

		if err := writer.Close(); err != nil {
			return nil, err
		}

		bodyReader = bodyBuffer
		multipartContentType = writer.FormDataContentType()
	} else if jsonValue, ok := requestSpec["json"]; ok {
		payload, err := json.Marshal(jsonValue)
		if err != nil {
			return nil, err
		}
		bodyReader = bytes.NewReader(payload)
	} else if dataValue, ok := requestSpec["data"]; ok {
		formData := url.Values{}
		for key, value := range toMap(dataValue) {
			formData.Set(key, fmt.Sprint(value))
		}
		bodyReader = strings.NewReader(formData.Encode())
	} else if rawBody, ok := requestSpec["body"]; ok {
		bodyReader = strings.NewReader(fmt.Sprint(rawBody))
	}

	req, err := http.NewRequest(method, requestURL, bodyReader)
	if err != nil {
		return nil, err
	}

	for headerName, headerValue := range toStringMap(requestSpec["headers"]) {
		req.Header.Set(headerName, headerValue)
	}
	if multipartContentType != "" && req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", multipartContentType)
	}
	if _, ok := requestSpec["json"]; ok && req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/json")
	}
	if _, ok := requestSpec["data"]; ok && req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}
	return req, nil
}

func buildRequestURL(baseURL string, requestSpec map[string]any) (string, error) {
	if rawURL, ok := requestSpec["url"].(string); ok && rawURL != "" {
		return rawURL, nil
	}

	path, ok := requestSpec["path"].(string)
	if !ok || path == "" {
		return "", &e2eError{message: "request must define either 'path' or 'url'"}
	}

	base, err := url.Parse(baseURL)
	if err != nil {
		return "", err
	}
	resolved, err := base.Parse(strings.TrimLeft(path, "/"))
	if err != nil {
		return "", err
	}

	if params, ok := requestSpec["params"]; ok {
		query := resolved.Query()
		for key, value := range toMap(params) {
			query.Set(key, fmt.Sprint(value))
		}
		resolved.RawQuery = query.Encode()
	}

	return resolved.String(), nil
}

func assertResponse(stepID string, expect map[string]any, context map[string]any, result stepResult) error {
	if rawStatus, ok := expect["status"]; ok {
		expectedStatus, err := toInt(rawStatus)
		if err != nil {
			return err
		}
		if result.StatusCode != expectedStatus {
			return &e2eError{message: fmt.Sprintf("%s: expected status %d, got %d", stepID, expectedStatus, result.StatusCode)}
		}
	}

	if rawStatuses, ok := expect["status_in"]; ok {
		var allowed []int
		for _, rawStatus := range toSlice(rawStatuses) {
			parsed, err := toInt(rawStatus)
			if err != nil {
				return err
			}
			allowed = append(allowed, parsed)
		}
		if !containsInt(allowed, result.StatusCode) {
			return &e2eError{message: fmt.Sprintf("%s: expected status in %v, got %d", stepID, allowed, result.StatusCode)}
		}
	}

	if rawEquals, ok := expect["json_equals"]; ok {
		if result.ResponseJSON == nil {
			return &e2eError{message: fmt.Sprintf("%s: expected JSON response for json_equals", stepID)}
		}
		for path, expectedValue := range toMap(rawEquals) {
			actualValue, err := lookupPath(result.ResponseJSON, path)
			if err != nil {
				return err
			}
			resolvedExpected, err := resolveValue(expectedValue, context)
			if err != nil {
				return err
			}
			if !valuesEqual(actualValue, resolvedExpected) {
				return &e2eError{message: fmt.Sprintf("%s: json.%s expected %#v, got %#v", stepID, path, resolvedExpected, actualValue)}
			}
		}
	}

	if rawNotEquals, ok := expect["json_not_equals"]; ok {
		if result.ResponseJSON == nil {
			return &e2eError{message: fmt.Sprintf("%s: expected JSON response for json_not_equals", stepID)}
		}
		for path, unexpectedValue := range toMap(rawNotEquals) {
			actualValue, err := lookupPath(result.ResponseJSON, path)
			if err != nil {
				return err
			}
			resolvedUnexpected, err := resolveValue(unexpectedValue, context)
			if err != nil {
				return err
			}
			if valuesEqual(actualValue, resolvedUnexpected) {
				return &e2eError{message: fmt.Sprintf("%s: json.%s unexpectedly matched %#v", stepID, path, resolvedUnexpected)}
			}
		}
	}

	if rawContains, ok := expect["json_contains"]; ok {
		if result.ResponseJSON == nil {
			return &e2eError{message: fmt.Sprintf("%s: expected JSON response for json_contains", stepID)}
		}
		for path, expectedSubstring := range toMap(rawContains) {
			actualValue, err := lookupPath(result.ResponseJSON, path)
			if err != nil {
				return err
			}
			resolvedSubstring, err := resolveValue(expectedSubstring, context)
			if err != nil {
				return err
			}
			if !strings.Contains(fmt.Sprint(actualValue), fmt.Sprint(resolvedSubstring)) {
				return &e2eError{message: fmt.Sprintf("%s: json.%s does not contain %q; actual=%#v", stepID, path, fmt.Sprint(resolvedSubstring), actualValue)}
			}
		}
	}

	if rawNumberGTE, ok := expect["json_number_gte"]; ok {
		if result.ResponseJSON == nil {
			return &e2eError{message: fmt.Sprintf("%s: expected JSON response for json_number_gte", stepID)}
		}
		for path, expectedValue := range toMap(rawNumberGTE) {
			actualValue, err := lookupPath(result.ResponseJSON, path)
			if err != nil {
				return err
			}
			actualNumber, err := toFloat(actualValue)
			if err != nil {
				return &e2eError{message: fmt.Sprintf("%s: json.%s is not numeric: %v", stepID, path, err)}
			}
			resolvedExpected, err := resolveValue(expectedValue, context)
			if err != nil {
				return err
			}
			expectedNumber, err := toFloat(resolvedExpected)
			if err != nil {
				return &e2eError{message: fmt.Sprintf("%s: expected numeric threshold for json_number_gte.%s: %v", stepID, path, err)}
			}
			if actualNumber < expectedNumber {
				return &e2eError{message: fmt.Sprintf("%s: json.%s expected >= %v, got %v", stepID, path, expectedNumber, actualNumber)}
			}
		}
	}

	if rawNumberLTE, ok := expect["json_number_lte"]; ok {
		if result.ResponseJSON == nil {
			return &e2eError{message: fmt.Sprintf("%s: expected JSON response for json_number_lte", stepID)}
		}
		for path, expectedValue := range toMap(rawNumberLTE) {
			actualValue, err := lookupPath(result.ResponseJSON, path)
			if err != nil {
				return err
			}
			actualNumber, err := toFloat(actualValue)
			if err != nil {
				return &e2eError{message: fmt.Sprintf("%s: json.%s is not numeric: %v", stepID, path, err)}
			}
			resolvedExpected, err := resolveValue(expectedValue, context)
			if err != nil {
				return err
			}
			expectedNumber, err := toFloat(resolvedExpected)
			if err != nil {
				return &e2eError{message: fmt.Sprintf("%s: expected numeric threshold for json_number_lte.%s: %v", stepID, path, err)}
			}
			if actualNumber > expectedNumber {
				return &e2eError{message: fmt.Sprintf("%s: json.%s expected <= %v, got %v", stepID, path, expectedNumber, actualNumber)}
			}
		}
	}

	if rawExists, ok := expect["json_exists"]; ok {
		if result.ResponseJSON == nil {
			return &e2eError{message: fmt.Sprintf("%s: expected JSON response for json_exists", stepID)}
		}
		for _, rawPath := range toSlice(rawExists) {
			if _, err := lookupPath(result.ResponseJSON, fmt.Sprint(rawPath)); err != nil {
				return err
			}
		}
	}

	if rawHeaderEquals, ok := expect["header_equals"]; ok {
		for headerName, expectedValue := range toMap(rawHeaderEquals) {
			resolvedExpected, err := resolveValue(expectedValue, context)
			if err != nil {
				return err
			}
			actualValue := result.Headers.Get(headerName)
			if actualValue != fmt.Sprint(resolvedExpected) {
				return &e2eError{message: fmt.Sprintf("%s: header %q expected %q, got %q", stepID, headerName, fmt.Sprint(resolvedExpected), actualValue)}
			}
		}
	}

	if rawHeaderExists, ok := expect["header_exists"]; ok {
		for _, rawHeader := range toSlice(rawHeaderExists) {
			headerName := fmt.Sprint(rawHeader)
			if result.Headers.Get(headerName) == "" {
				return &e2eError{message: fmt.Sprintf("%s: missing expected header %q", stepID, headerName)}
			}
		}
	}

	if rawBodyContains, ok := expect["body_contains"]; ok {
		for _, rawSubstring := range toSlice(rawBodyContains) {
			resolvedSubstring, err := resolveValue(rawSubstring, context)
			if err != nil {
				return err
			}
			if !strings.Contains(result.Body, fmt.Sprint(resolvedSubstring)) {
				return &e2eError{message: fmt.Sprintf("%s: response body does not contain %q", stepID, fmt.Sprint(resolvedSubstring))}
			}
		}
	}

	if rawMedia, ok := expect["media"]; ok {
		if result.MediaProbe == nil {
			return &e2eError{message: fmt.Sprintf("%s: expected media probe result", stepID)}
		}
		if err := assertMediaExpectations(stepID, toMap(rawMedia), result.MediaProbe); err != nil {
			return err
		}
	}

	if rawImage, ok := expect["image"]; ok {
		if result.ImageProbe == nil {
			return &e2eError{message: fmt.Sprintf("%s: expected image probe result", stepID)}
		}
		if err := assertImageExpectations(stepID, toMap(rawImage), result.ImageProbe); err != nil {
			return err
		}
	}

	return nil
}

func persistSavedValues(saveItems []saveItem, result stepResult, savedValues map[string]any, context map[string]any) error {
	for _, item := range saveItems {
		value, err := extractSavedValue(item, result)
		if err != nil {
			return err
		}
		if item.DecodeBase64To != "" {
			value, err = decodeBase64ToFile(value, item.DecodeBase64To, context)
			if err != nil {
				return err
			}
		}
		savedValues[item.As] = value
	}
	return nil
}

func decodeBase64ToFile(value any, outputPathTemplate string, context map[string]any) (string, error) {
	outputPath, err := resolveString(outputPathTemplate, context)
	if err != nil {
		return "", err
	}
	encoded := strings.TrimSpace(fmt.Sprint(value))
	if comma := strings.Index(encoded, ","); comma >= 0 && strings.Contains(encoded[:comma], "base64") {
		encoded = encoded[comma+1:]
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(outputPath, decoded, 0o644); err != nil {
		return "", err
	}
	return outputPath, nil
}

func extractSavedValue(item saveItem, result stepResult) (any, error) {
	if len(item.FromAny) > 0 {
		for _, source := range item.FromAny {
			value, err := extractSavedValueBySource(source, result)
			if err == nil {
				return value, nil
			}
		}
		return nil, &e2eError{message: fmt.Sprintf("unable to resolve any save source for %q", item.As)}
	}
	return extractSavedValueBySource(item.From, result)
}

func extractSavedValueBySource(source string, result stepResult) (any, error) {
	switch {
	case source == "body":
		return result.Body, nil
	case source == "file.path":
		if result.SavedBodyPath == "" {
			return nil, &e2eError{message: "cannot save file.path when response body was not written to disk"}
		}
		return result.SavedBodyPath, nil
	case source == "file.size":
		if result.SavedBodyPath == "" {
			return nil, &e2eError{message: "cannot save file.size when response body was not written to disk"}
		}
		info, err := os.Stat(result.SavedBodyPath)
		if err != nil {
			return nil, err
		}
		return info.Size(), nil
	case strings.HasPrefix(source, "json.") || strings.HasPrefix(source, "json["):
		if result.ResponseJSON == nil {
			return nil, &e2eError{message: "cannot save from json.* when response is not JSON"}
		}
		return lookupPath(result.ResponseJSON, strings.TrimPrefix(strings.TrimPrefix(source, "json."), "json"))
	case strings.HasPrefix(source, "media."):
		if result.MediaProbe == nil {
			return nil, &e2eError{message: "cannot save from media.* when media probe did not run"}
		}
		return lookupPath(result.MediaProbe.asMap(), strings.TrimPrefix(source, "media."))
	case strings.HasPrefix(source, "frames."):
		if len(result.FrameOutputs) == 0 {
			return nil, &e2eError{message: "cannot save from frames.* when frame extraction did not run"}
		}
		return lookupPath(frameOutputsAsMap(result.FrameOutputs), strings.TrimPrefix(source, "frames."))
	case strings.HasPrefix(source, "estimate."):
		if len(result.CostEstimate) == 0 {
			return nil, &e2eError{message: "cannot save from estimate.* when cost_estimate did not run"}
		}
		return lookupPath(result.CostEstimate, strings.TrimPrefix(source, "estimate."))
	case strings.HasPrefix(source, "value."):
		if len(result.Values) == 0 {
			return nil, &e2eError{message: "cannot save from value.* when step did not produce values"}
		}
		return lookupPath(result.Values, strings.TrimPrefix(source, "value."))
	case strings.HasPrefix(source, "header."):
		return result.Headers.Get(strings.TrimPrefix(source, "header.")), nil
	default:
		return nil, &e2eError{message: fmt.Sprintf("unsupported save source %q", source)}
	}
}

func resolveToMap(value any, context map[string]any) (map[string]any, error) {
	if value == nil {
		return map[string]any{}, nil
	}
	resolved, err := resolveValue(value, context)
	if err != nil {
		return nil, err
	}
	resolvedMap, ok := resolved.(map[string]any)
	if !ok {
		return nil, &e2eError{message: "resolved value is not a map"}
	}
	return resolvedMap, nil
}

func resolveValue(value any, context map[string]any) (any, error) {
	switch typed := value.(type) {
	case nil:
		return nil, nil
	case string:
		return resolveString(typed, context)
	case []any:
		resolved := make([]any, 0, len(typed))
		for _, item := range typed {
			current, err := resolveValue(item, context)
			if err != nil {
				return nil, err
			}
			resolved = append(resolved, current)
		}
		return resolved, nil
	case map[string]any:
		if len(typed) == 1 {
			if repeatSpec, ok := typed["repeat"]; ok {
				return resolveRepeatValue(repeatSpec, context)
			}
		}
		resolved := map[string]any{}
		for key, item := range typed {
			current, err := resolveValue(item, context)
			if err != nil {
				return nil, err
			}
			resolved[key] = current
		}
		return resolved, nil
	default:
		return value, nil
	}
}

func resolveRepeatValue(value any, context map[string]any) (string, error) {
	spec, ok := value.(map[string]any)
	if !ok {
		return "", &e2eError{message: "repeat must be a map"}
	}

	textValue, err := resolveValue(spec["text"], context)
	if err != nil {
		return "", err
	}
	text := fmt.Sprint(textValue)

	countValue, err := resolveValue(spec["count"], context)
	if err != nil {
		return "", err
	}
	count, err := toInt(countValue)
	if err != nil {
		return "", &e2eError{message: fmt.Sprintf("invalid repeat count: %v", err)}
	}
	if count < 0 {
		return "", &e2eError{message: "repeat count must be >= 0"}
	}

	separator := ""
	if separatorValue, ok := spec["separator"]; ok {
		resolvedSeparator, err := resolveValue(separatorValue, context)
		if err != nil {
			return "", err
		}
		separator = fmt.Sprint(resolvedSeparator)
	}

	if count == 0 {
		return "", nil
	}

	values := make([]string, count)
	for i := 0; i < count; i++ {
		values[i] = text
	}
	return strings.Join(values, separator), nil
}

func resolveString(value string, context map[string]any) (string, error) {
	var resolveErr error
	resolved := envPattern.ReplaceAllStringFunc(value, func(match string) string {
		if resolveErr != nil {
			return ""
		}

		expression := strings.TrimSuffix(strings.TrimPrefix(match, "${"), "}")
		key := expression
		defaultValue := ""
		hasDefault := false
		if strings.Contains(expression, ":") {
			parts := strings.SplitN(expression, ":", 2)
			key = parts[0]
			defaultValue = parts[1]
			hasDefault = true
		}

		if envValue, ok := os.LookupEnv(key); ok {
			return envValue
		}
		if contextValue, ok := lookupContextValue(key, context); ok {
			return fmt.Sprint(contextValue)
		}
		if hasDefault {
			return defaultValue
		}

		resolveErr = &e2eError{message: fmt.Sprintf("missing substitution value for %q", key)}
		return ""
	})

	if resolveErr != nil {
		return "", resolveErr
	}
	return resolved, nil
}

func lookupContextValue(key string, context map[string]any) (any, bool) {
	if value, ok := context[key]; ok {
		return value, true
	}
	value, err := lookupPath(context, key)
	if err != nil {
		return nil, false
	}
	return value, true
}

func lookupPath(data any, path string) (any, error) {
	current := data
	for _, token := range parsePathTokens(path) {
		switch typedToken := token.(type) {
		case string:
			currentMap, ok := current.(map[string]any)
			if !ok {
				return nil, &e2eError{message: fmt.Sprintf("path not found: %s", path)}
			}
			next, ok := currentMap[typedToken]
			if !ok {
				return nil, &e2eError{message: fmt.Sprintf("path not found: %s", path)}
			}
			current = next
		case int:
			currentSlice, ok := current.([]any)
			if !ok || typedToken < 0 || typedToken >= len(currentSlice) {
				return nil, &e2eError{message: fmt.Sprintf("path not found: %s", path)}
			}
			current = currentSlice[typedToken]
		default:
			return nil, &e2eError{message: fmt.Sprintf("unsupported token in path: %v", token)}
		}
	}
	return current, nil
}

func parsePathTokens(path string) []any {
	var tokens []any
	for index := 0; index < len(path); {
		switch path[index] {
		case '.':
			index++
		case '[':
			token, nextIndex, ok := parseBracketPathToken(path, index)
			if !ok {
				tokens = append(tokens, path[index:])
				return tokens
			}
			tokens = append(tokens, token)
			index = nextIndex
		default:
			startIndex := index
			for index < len(path) && path[index] != '.' && path[index] != '[' {
				index++
			}
			tokens = append(tokens, path[startIndex:index])
		}
	}
	return tokens
}

func parseBracketPathToken(path string, startIndex int) (any, int, bool) {
	index := startIndex + 1
	if index >= len(path) {
		return nil, 0, false
	}

	if path[index] == '"' {
		quoteStart := index
		index++
		for index < len(path) {
			if path[index] == '\\' {
				index += 2
				continue
			}
			if path[index] == '"' && index+1 < len(path) && path[index+1] == ']' {
				value, err := strconv.Unquote(path[quoteStart : index+1])
				if err != nil {
					return nil, 0, false
				}
				return value, index + 2, true
			}
			index++
		}
		return nil, 0, false
	}

	endIndex := strings.IndexByte(path[index:], ']')
	if endIndex == -1 {
		return nil, 0, false
	}
	indexValue, err := strconv.Atoi(path[index : index+endIndex])
	if err != nil {
		return nil, 0, false
	}
	return indexValue, index + endIndex + 1, true
}

func missingRequiredEnv(required []string) []string {
	var missing []string
	for _, key := range required {
		if _, ok := os.LookupEnv(key); !ok {
			missing = append(missing, key)
		}
	}
	return missing
}

func deepMergeMaps(base map[string]any, override map[string]any) map[string]any {
	merged := copyMapAny(base)
	for key, value := range override {
		existing, existingOK := merged[key].(map[string]any)
		overrideMap, overrideOK := value.(map[string]any)
		if existingOK && overrideOK {
			merged[key] = deepMergeMaps(existing, overrideMap)
			continue
		}
		merged[key] = value
	}
	return merged
}

func tryParseJSON(body []byte) (any, bool) {
	var parsed any
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, false
	}
	return parsed, true
}

func toMap(value any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	return map[string]any{}
}

func toStringMap(value any) map[string]string {
	resolved := map[string]string{}
	for key, current := range toMap(value) {
		resolved[key] = fmt.Sprint(current)
	}
	return resolved
}

func escapeQuotes(value string) string {
	return strings.ReplaceAll(value, `"`, `\"`)
}

func toSlice(value any) []any {
	if value == nil {
		return nil
	}
	if typed, ok := value.([]any); ok {
		return typed
	}
	return nil
}

func toSet(values []string) map[string]struct{} {
	if len(values) == 0 {
		return nil
	}
	resolved := map[string]struct{}{}
	for _, current := range values {
		resolved[current] = struct{}{}
	}
	return resolved
}

func mergeMap(dst map[string]any, src map[string]any) {
	for key, value := range src {
		dst[key] = value
	}
}

func copyMap(source map[string]any) map[string]any {
	return copyMapAny(source)
}

func copyMapAny(source map[string]any) map[string]any {
	if source == nil {
		return map[string]any{}
	}
	copied := make(map[string]any, len(source))
	for key, value := range source {
		copied[key] = value
	}
	return copied
}

func getStringWithDefault(values map[string]any, key, fallback string) string {
	if value, ok := values[key]; ok {
		return fmt.Sprint(value)
	}
	return fallback
}

func getFloatWithDefault(values map[string]any, key string, fallback float64) float64 {
	value, ok := values[key]
	if !ok {
		return fallback
	}

	switch typed := value.(type) {
	case float64:
		return typed
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	case string:
		parsed, err := strconv.ParseFloat(typed, 64)
		if err == nil {
			return parsed
		}
	}
	return fallback
}

func toInt(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int64:
		return int(typed), nil
	case float64:
		return int(typed), nil
	case string:
		return strconv.Atoi(typed)
	default:
		return 0, &e2eError{message: fmt.Sprintf("cannot convert %T to int", value)}
	}
}

func toFloat(value any) (float64, error) {
	switch typed := value.(type) {
	case float64:
		return typed, nil
	case float32:
		return float64(typed), nil
	case int:
		return float64(typed), nil
	case int64:
		return float64(typed), nil
	case int32:
		return float64(typed), nil
	case json.Number:
		return typed.Float64()
	case string:
		return strconv.ParseFloat(typed, 64)
	default:
		return 0, &e2eError{message: fmt.Sprintf("cannot convert %T to float", value)}
	}
}

func toFloatDefault(value any, fallback float64) (float64, error) {
	if value == nil {
		return fallback, nil
	}
	return toFloat(value)
}

func containsInt(values []int, target int) bool {
	for _, current := range values {
		if current == target {
			return true
		}
	}
	return false
}

func valuesEqual(left any, right any) bool {
	leftBytes, leftErr := json.Marshal(left)
	rightBytes, rightErr := json.Marshal(right)
	if leftErr != nil || rightErr != nil {
		return fmt.Sprint(left) == fmt.Sprint(right)
	}
	return bytes.Equal(leftBytes, rightBytes)
}

func writeResponseBodyToFile(rawPath string, context map[string]any, body []byte) (string, error) {
	savePath := rawPath
	if !filepath.IsAbs(savePath) {
		artifactsDir, _ := context["artifacts_dir"].(string)
		savePath = filepath.Join(artifactsDir, savePath)
	}
	if err := os.MkdirAll(filepath.Dir(savePath), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(savePath, body, 0o644); err != nil {
		return "", err
	}
	return savePath, nil
}

func matchPollCondition(responseJSON any, condition map[string]any) (bool, string, error) {
	if len(condition) == 0 || responseJSON == nil {
		return false, "", nil
	}

	hasLengthCondition := false
	if rawLengthGTE, ok := condition["length_gte"]; ok {
		hasLengthCondition = true
		expectedLength, err := toInt(rawLengthGTE)
		if err != nil {
			return false, "", err
		}
		if currentSlice, ok := responseJSON.([]any); ok && len(currentSlice) >= expectedLength {
			return true, fmt.Sprint(len(currentSlice)), nil
		}
	}

	if matched, value, ok, err := matchPollNumberComparisons(responseJSON, condition, "json_number_gte", true); ok {
		return matched, value, err
	}
	if matched, value, ok, err := matchPollNumberComparisons(responseJSON, condition, "json_number_lte", false); ok {
		return matched, value, err
	}

	var candidatePaths []string
	if rawPath, ok := condition["json_path"]; ok {
		candidatePaths = append(candidatePaths, fmt.Sprint(rawPath))
	}
	if rawPaths, ok := condition["json_path_any"]; ok {
		for _, rawPath := range toSlice(rawPaths) {
			candidatePaths = append(candidatePaths, fmt.Sprint(rawPath))
		}
	}
	if len(candidatePaths) == 0 {
		if hasLengthCondition {
			return false, "", nil
		}
		return false, "", &e2eError{message: "poll condition must define json_path or json_path_any"}
	}

	var actualValues []string
	for _, path := range candidatePaths {
		value, err := lookupPath(responseJSON, path)
		if err == nil {
			actualValues = append(actualValues, strings.ToLower(strings.TrimSpace(fmt.Sprint(value))))
		}
	}
	if len(actualValues) == 0 {
		return false, "", nil
	}

	for _, rawValue := range toSlice(condition["values"]) {
		expected := strings.ToLower(strings.TrimSpace(fmt.Sprint(rawValue)))
		for _, actual := range actualValues {
			if actual == expected {
				return true, actual, nil
			}
		}
	}

	return false, "", nil
}

func matchPollNumberComparisons(responseJSON any, condition map[string]any, key string, gte bool) (bool, string, bool, error) {
	rawComparisons, ok := condition[key]
	if !ok {
		return false, "", false, nil
	}

	comparisons := toMap(rawComparisons)
	if len(comparisons) == 0 {
		return false, "", true, &e2eError{message: fmt.Sprintf("poll condition %s must define at least one JSON path", key)}
	}

	var matchedValues []string
	for path, rawExpected := range comparisons {
		actualValue, err := lookupPath(responseJSON, path)
		if err != nil {
			return false, "", true, nil
		}
		actualNumber, err := toFloat(actualValue)
		if err != nil {
			return false, "", true, &e2eError{message: fmt.Sprintf("poll condition %s.%s actual value is not numeric: %v", key, path, err)}
		}
		expectedNumber, err := toFloat(rawExpected)
		if err != nil {
			return false, "", true, &e2eError{message: fmt.Sprintf("poll condition %s.%s expected value is not numeric: %v", key, path, err)}
		}

		if gte {
			if actualNumber < expectedNumber {
				return false, "", true, nil
			}
		} else if actualNumber > expectedNumber {
			return false, "", true, nil
		}
		matchedValues = append(matchedValues, fmt.Sprintf("%s=%g", path, actualNumber))
	}

	return true, strings.Join(matchedValues, ","), true, nil
}

func findStableNovaDiscount(responseJSON any, modelName string) (float64, int, error) {
	data, err := lookupPath(responseJSON, "data")
	if err != nil {
		return 0, 0, err
	}
	deployments := toSlice(data)
	if len(deployments) == 0 {
		return 0, 0, &skipError{message: "/model/info returned no deployments"}
	}

	var discount *float64
	matchedDeployments := 0
	for _, rawDeployment := range deployments {
		deployment := toMap(rawDeployment)
		if fmt.Sprint(deployment["model_name"]) != modelName {
			continue
		}
		matchedDeployments++

		currentDiscount, ok, err := deploymentNovaCostDiscount(deployment)
		if err != nil {
			return 0, 0, err
		}
		if !ok {
			return 0, matchedDeployments, &skipError{message: fmt.Sprintf("%s deployment is missing nova_cost_discount", modelName)}
		}
		if currentDiscount <= 0 || currentDiscount >= 1 {
			return 0, matchedDeployments, &skipError{message: fmt.Sprintf("%s deployment has non-discount value %g", modelName, currentDiscount)}
		}
		if discount == nil {
			value := currentDiscount
			discount = &value
			continue
		}
		if absFloat(*discount-currentDiscount) > 1e-12 {
			return 0, matchedDeployments, &skipError{message: fmt.Sprintf("%s deployments have different nova_cost_discount values", modelName)}
		}
	}

	if matchedDeployments == 0 {
		return 0, 0, &skipError{message: fmt.Sprintf("/model/info has no deployment for %s", modelName)}
	}
	if discount == nil {
		return 0, matchedDeployments, &skipError{message: fmt.Sprintf("%s has no nova_cost_discount", modelName)}
	}
	return *discount, matchedDeployments, nil
}

func deploymentNovaCostDiscount(deployment map[string]any) (float64, bool, error) {
	for _, path := range []string{
		"litellm_params.nova_cost_discount",
		"model_info.nova_cost_discount",
	} {
		value, err := lookupPath(deployment, path)
		if err != nil || value == nil {
			continue
		}
		parsed, err := toFloat(value)
		if err != nil {
			return 0, false, err
		}
		return parsed, true, nil
	}
	return 0, false, nil
}

func withinTolerance(actual, expected, absoluteTolerance, percentTolerance float64) bool {
	tolerance := absoluteTolerance
	if percentTolerance > 0 {
		percentValue := absFloat(expected) * percentTolerance
		if percentValue > tolerance {
			tolerance = percentValue
		}
	}
	return absFloat(actual-expected) <= tolerance
}

func assertMediaExpectations(stepID string, expect map[string]any, probe *mediaProbeResult) error {
	videoStreams := probe.videoStreams()
	if raw := expect["video_streams_gte"]; raw != nil {
		expected, err := toInt(raw)
		if err != nil {
			return err
		}
		if len(videoStreams) < expected {
			return &e2eError{message: fmt.Sprintf("%s: expected at least %d video streams, got %d", stepID, expected, len(videoStreams))}
		}
	}

	primaryStream, hasVideo := probe.primaryVideoStream()
	if raw := expect["width_gte"]; raw != nil {
		if !hasVideo {
			return &e2eError{message: fmt.Sprintf("%s: no video stream available for width check", stepID)}
		}
		expected, err := toInt(raw)
		if err != nil {
			return err
		}
		if primaryStream.Width < expected {
			return &e2eError{message: fmt.Sprintf("%s: width expected >= %d, got %d", stepID, expected, primaryStream.Width)}
		}
	}
	if raw := expect["height_gte"]; raw != nil {
		if !hasVideo {
			return &e2eError{message: fmt.Sprintf("%s: no video stream available for height check", stepID)}
		}
		expected, err := toInt(raw)
		if err != nil {
			return err
		}
		if primaryStream.Height < expected {
			return &e2eError{message: fmt.Sprintf("%s: height expected >= %d, got %d", stepID, expected, primaryStream.Height)}
		}
	}

	durationSeconds := probe.Format.DurationSeconds
	if durationSeconds == 0 && hasVideo {
		durationSeconds = primaryStream.DurationSeconds
	}
	if raw := expect["duration_seconds_gte"]; raw != nil {
		expected := getFloatWithDefault(map[string]any{"value": raw}, "value", 0)
		if durationSeconds < expected {
			return &e2eError{message: fmt.Sprintf("%s: duration expected >= %.2f, got %.2f", stepID, expected, durationSeconds)}
		}
	}
	if raw := expect["duration_seconds_lte"]; raw != nil {
		expected := getFloatWithDefault(map[string]any{"value": raw}, "value", 0)
		if durationSeconds > expected {
			return &e2eError{message: fmt.Sprintf("%s: duration expected <= %.2f, got %.2f", stepID, expected, durationSeconds)}
		}
	}
	if raw := expect["aspect_ratio"]; raw != nil {
		if !hasVideo {
			return &e2eError{message: fmt.Sprintf("%s: no video stream available for aspect ratio check", stepID)}
		}
		expectedRatio, err := parseAspectRatio(fmt.Sprint(raw))
		if err != nil {
			return err
		}
		tolerance := getFloatWithDefault(expect, "aspect_ratio_tolerance", 0.03)
		actualRatio := float64(primaryStream.Width) / float64(primaryStream.Height)
		if absFloat(actualRatio-expectedRatio) > tolerance {
			return &e2eError{message: fmt.Sprintf("%s: aspect ratio expected %.4f ± %.4f, got %.4f", stepID, expectedRatio, tolerance, actualRatio)}
		}
	}

	return nil
}

func assertImageExpectations(stepID string, expect map[string]any, probe *imageProbeResult) error {
	if raw := expect["format"]; raw != nil && probe.Format != fmt.Sprint(raw) {
		return &e2eError{message: fmt.Sprintf("%s: image format expected %q, got %q", stepID, fmt.Sprint(raw), probe.Format)}
	}
	if raw := expect["width_gte"]; raw != nil {
		expected, err := toInt(raw)
		if err != nil {
			return err
		}
		if probe.Width < expected {
			return &e2eError{message: fmt.Sprintf("%s: image width expected >= %d, got %d", stepID, expected, probe.Width)}
		}
	}
	if raw := expect["height_gte"]; raw != nil {
		expected, err := toInt(raw)
		if err != nil {
			return err
		}
		if probe.Height < expected {
			return &e2eError{message: fmt.Sprintf("%s: image height expected >= %d, got %d", stepID, expected, probe.Height)}
		}
	}
	if raw := expect["transparent_pixels_gte"]; raw != nil {
		expected, err := toInt(raw)
		if err != nil {
			return err
		}
		if probe.TransparentPixels < expected {
			return &e2eError{message: fmt.Sprintf("%s: transparent pixels expected >= %d, got %d", stepID, expected, probe.TransparentPixels)}
		}
	}
	if raw := expect["transparent_fraction_gte"]; raw != nil {
		expected, err := toFloat(raw)
		if err != nil {
			return err
		}
		if expected < 0 || expected > 1 {
			return &e2eError{message: fmt.Sprintf("%s: transparent fraction threshold must be between 0 and 1, got %g", stepID, expected)}
		}
		if probe.TotalPixels == 0 {
			return &e2eError{message: fmt.Sprintf("%s: image has no pixels", stepID)}
		}
		actual := float64(probe.TransparentPixels) / float64(probe.TotalPixels)
		if actual < expected {
			return &e2eError{message: fmt.Sprintf("%s: transparent pixel fraction expected >= %.4f, got %.4f", stepID, expected, actual)}
		}
	}
	return nil
}

func probeMediaFile(path string) (*mediaProbeResult, error) {
	command := exec.Command(
		"ffprobe",
		"-v", "error",
		"-print_format", "json",
		"-show_format",
		"-show_streams",
		path,
	)
	output, err := command.Output()
	if err != nil {
		return nil, &e2eError{message: fmt.Sprintf("ffprobe failed for %s: %v", path, err)}
	}

	var raw struct {
		Format struct {
			Filename string `json:"filename"`
			Duration string `json:"duration"`
		} `json:"format"`
		Streams []struct {
			CodecType string `json:"codec_type"`
			Width     int    `json:"width"`
			Height    int    `json:"height"`
			Duration  string `json:"duration"`
		} `json:"streams"`
	}
	if err := json.Unmarshal(output, &raw); err != nil {
		return nil, err
	}

	probe := &mediaProbeResult{
		Format: mediaProbeFormat{
			Filename:        raw.Format.Filename,
			DurationSeconds: safeParseFloat(raw.Format.Duration),
		},
	}
	for _, stream := range raw.Streams {
		probe.Streams = append(probe.Streams, mediaProbeStream{
			CodecType:       stream.CodecType,
			Width:           stream.Width,
			Height:          stream.Height,
			DurationSeconds: safeParseFloat(stream.Duration),
		})
	}
	return probe, nil
}

func probePNGFile(path string) (*imageProbeResult, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, &e2eError{message: fmt.Sprintf("read image %s: %v", path, err)}
	}
	decoded, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, &e2eError{message: fmt.Sprintf("decode PNG %s: %v", path, err)}
	}

	bounds := decoded.Bounds()
	probe := &imageProbeResult{
		Format:      "png",
		Width:       bounds.Dx(),
		Height:      bounds.Dy(),
		TotalPixels: bounds.Dx() * bounds.Dy(),
	}
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			_, _, _, alpha := decoded.At(x, y).RGBA()
			if alpha == 0 {
				probe.TransparentPixels++
			}
		}
	}
	return probe, nil
}

func extractVideoFrame(inputPath string, outputPath string, atSeconds float64) error {
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	command := exec.Command(
		"ffmpeg",
		"-y",
		"-ss", fmt.Sprintf("%.3f", atSeconds),
		"-i", inputPath,
		"-vframes", "1",
		"-update", "1",
		outputPath,
	)
	output, err := command.CombinedOutput()
	if err != nil {
		return &e2eError{message: fmt.Sprintf("ffmpeg frame extraction failed: %v (%s)", err, strings.TrimSpace(string(output)))}
	}
	return nil
}

func (m *mediaProbeResult) videoStreams() []mediaProbeStream {
	if m == nil {
		return nil
	}
	var streams []mediaProbeStream
	for _, stream := range m.Streams {
		if strings.EqualFold(stream.CodecType, "video") {
			streams = append(streams, stream)
		}
	}
	return streams
}

func (m *mediaProbeResult) primaryVideoStream() (mediaProbeStream, bool) {
	streams := m.videoStreams()
	if len(streams) == 0 {
		return mediaProbeStream{}, false
	}
	return streams[0], true
}

func (m *mediaProbeResult) asMap() map[string]any {
	streams := make([]any, 0, len(m.Streams))
	for _, stream := range m.Streams {
		streams = append(streams, map[string]any{
			"codec_type":       stream.CodecType,
			"width":            stream.Width,
			"height":           stream.Height,
			"duration_seconds": stream.DurationSeconds,
		})
	}
	return map[string]any{
		"format": map[string]any{
			"filename":         m.Format.Filename,
			"duration_seconds": m.Format.DurationSeconds,
		},
		"streams": streams,
	}
}

func frameOutputsAsMap(outputs map[string]frameExtractOutput) map[string]any {
	result := map[string]any{}
	for name, output := range outputs {
		result[name] = map[string]any{
			"path":       output.Path,
			"at_seconds": output.AtSeconds,
		}
	}
	return result
}

func parseAspectRatio(value string) (float64, error) {
	parts := strings.Split(value, ":")
	if len(parts) != 2 {
		return 0, &e2eError{message: fmt.Sprintf("invalid aspect ratio %q", value)}
	}
	width, err := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	if err != nil {
		return 0, err
	}
	height, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err != nil {
		return 0, err
	}
	if height == 0 {
		return 0, &e2eError{message: fmt.Sprintf("invalid aspect ratio %q", value)}
	}
	return width / height, nil
}

func safeParseFloat(value string) float64 {
	parsed, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
	if err != nil {
		return 0
	}
	return parsed
}

func absFloat(value float64) float64 {
	if value < 0 {
		return -value
	}
	return value
}

func exitError(err error) {
	if err == nil {
		return
	}
	fmt.Fprintf(os.Stderr, "error: %v\n", err)
	os.Exit(2)
}

func generateRunID(caseID string, scenarioID string) string {
	seed := fmt.Sprintf("%s:%s:%d", caseID, scenarioID, time.Now().UnixNano())
	hasher := fnv.New32a()
	_, _ = hasher.Write([]byte(seed))
	suffix := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(1000000)
	return fmt.Sprintf("%08x-%06d", hasher.Sum32(), suffix)
}
