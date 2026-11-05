package scraper

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

var batchYearRe = regexp.MustCompile(`\b(202[5-9])\b`)

// ParseQueryHeuristic parses a natural-language query into a JobSearchRequirement in Go.
func ParseQueryHeuristic(query string) JobSearchRequirement {
	lower := strings.ToLower(query)
	batches := batchYearRe.FindAllString(lower, -1)

	jobType := "internship"
	if strings.Contains(lower, "full time") || strings.Contains(lower, "full-time") {
		jobType = "full_time"
	}

	roles := []string{"Software Engineering", "SDE"}
	if strings.Contains(lower, "backend") {
		roles = append(roles, "Backend Developer")
	}
	if strings.Contains(lower, "data") {
		roles = []string{"Data Science", "Machine Learning"}
	}

	var locs []string
	cityMap := map[string]string{
		"bangalore": "Bengaluru",
		"bengaluru": "Bengaluru",
		"gurgaon":   "Gurugram",
		"gurugram":  "Gurugram",
		"hyderabad": "Hyderabad",
		"pune":      "Pune",
		"mumbai":    "Mumbai",
		"remote":    "Remote",
		"india":     "India",
	}
	for k, v := range cityMap {
		if strings.Contains(lower, k) {
			locs = append(locs, v)
		}
	}
	if len(locs) == 0 {
		locs = []string{"India"}
	}

	return JobSearchRequirement{
		RawQuery:        query,
		RoleKeywords:    roles,
		JobType:         jobType,
		TargetBatches:   batches,
		ExperienceLevel: "Fresher / 0 years / Student",
		Locations:       locs,
		WorkMode:        "any",
	}
}

// Build16SiteSearchQueries constructs 16 per-portal DuckDuckGo search queries without quotes around batch/location.
func Build16SiteSearchQueries(req JobSearchRequirement) []string {
	portals := GetTargetPortals()
	roleTerm := "software engineering"
	if len(req.RoleKeywords) > 0 {
		roleTerm = req.RoleKeywords[0]
	}
	typeTerm := "internship"
	if req.JobType == "full_time" {
		typeTerm = "jobs"
	}
	batchTerm := "fresher"
	if len(req.TargetBatches) > 0 {
		batchTerm = req.TargetBatches[0]
	}
	locTerm := "(India)"
	if len(req.Locations) > 0 {
		locTerm = fmt.Sprintf("(%s)", strings.Join(req.Locations, " OR "))
	}

	queries := make([]string, 0, len(portals))
	for _, p := range portals {
		q := fmt.Sprintf("site:%s (%s OR SDE) %s %s %s", p.Domain, roleTerm, typeTerm, batchTerm, locTerm)
		queries = append(queries, q)
	}
	return queries
}

// BuildDirectPortalURLs returns direct entry and search URLs across all 16 portals.
func BuildDirectPortalURLs(req JobSearchRequirement) []PortalConfig {
	portals := GetTargetPortals()
	kw := url.QueryEscape(strings.Join(append(req.RoleKeywords, req.TargetBatches...), " "))
	loc := url.QueryEscape(strings.Join(req.Locations, " "))
	out := make([]PortalConfig, 0, len(portals)+2)
	out = append(out, PortalConfig{
		ID:       "linkedin",
		Name:     "LinkedIn Jobs Search",
		Domain:   "linkedin.com",
		EntryURL: fmt.Sprintf("https://www.linkedin.com/jobs/search/?keywords=%s&location=%s", kw, loc),
		Render:   "never",
	})
	out = append(out, portals...)
	return out
}

// FetchHTML fetches page HTML in Go with browser headers and timeout.
func FetchHTML(targetURL string, timeout time.Duration) (string, error) {
	client := &http.Client{Timeout: timeout}
	req, err := http.NewRequest("GET", targetURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}

// MLBridgeRequest is sent from Go to Python `scraper.ml_bridge`.
type MLBridgeRequest struct {
	Requirement JobSearchRequirement `json:"requirement"`
	TopK        int                  `json:"top_k"`
	Candidates  []DOMCandidate       `json:"candidates"`
}

// MLBridgeResponse is returned by Python `scraper.ml_bridge` to Go.
type MLBridgeResponse struct {
	JobListings     []DOMCandidate `json:"job_listings"`
	PaginationLinks []DOMCandidate `json:"pagination_links"`
	NoiseCount      int            `json:"noise_count"`
	TotalCandidates int            `json:"total_candidates"`
}

// CallPythonMLBridge invokes `python3 -m scraper.ml_bridge` to run Stage 1 Bi-Encoder + Stage 2 Cross-Encoder.
func CallPythonMLBridge(pythonBin string, req JobSearchRequirement, candidates []DOMCandidate, topK int) (*MLBridgeResponse, error) {
	if pythonBin == "" {
		pythonBin = "python3"
	}
	payload := MLBridgeRequest{
		Requirement: req,
		TopK:        topK,
		Candidates:  candidates,
	}
	inputBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	cmd := exec.Command(pythonBin, "-m", "scraper.ml_bridge")
	cmd.Stdin = bytes.NewReader(inputBytes)
	var outBuf, errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ml_bridge failed (%v): %s", err, errBuf.String())
	}

	var parsed MLBridgeResponse
	if err := json.Unmarshal(outBuf.Bytes(), &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse ml_bridge output: %v", err)
	}
	return &parsed, nil
}

