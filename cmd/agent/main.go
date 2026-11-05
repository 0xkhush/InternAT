package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"interat-scraper/internal/scraper"
)

func main() {
	queryFlag := flag.String("query", "", "Initial job search query (e.g. '2028 internship software engineering india')")
	autoYes := flag.Bool("yes", false, "Skip interactive clarification prompt")
	pythonBin := flag.String("python", "python3", "Python binary for Bi-Encoder / Cross-Encoder ML bridge")
	flag.Parse()

	fmt.Println("========================================================================")
	fmt.Println("  InterAT 2.0 (Go + Python Hybrid) — ML-Powered Job & Internship Agent")
	fmt.Println("  • Go Engine: Concurrent 16-Portal Discovery, Fetcher, Outer-First DOM")
	fmt.Println("  • Python ML: Stage 1 Bi-Encoder (Contrastive) + Stage 2 Cross-Encoder")
	fmt.Println("  • LLM Agent: GLM-5.3-flash Structured Extraction & Deduplication")
	fmt.Println("========================================================================")

	reader := bufio.NewReader(os.Stdin)
	rawQuery := strings.TrimSpace(*queryFlag)
	if rawQuery == "" {
		fmt.Print("\nYou > ")
		line, _ := reader.ReadString('\n')
		rawQuery = strings.TrimSpace(line)
		if rawQuery == "" {
			fmt.Println("No query entered. Exiting.")
			return
		}
	}

	req := scraper.ParseQueryHeuristic(rawQuery)
	if !*autoYes {
		fmt.Printf("\nAgent (GLM-5.3-flash) > Understood: %s in %s (Batches: %v, Locations: %v)\n",
			req.JobType, strings.Join(req.RoleKeywords, "/"), req.TargetBatches, req.Locations)
		fmt.Println("Before I search across the 16 job portals, any specific cities, remote preference, skills, or minimum stipend? (Press Enter or type 'any' to start)")
		fmt.Print("\nYou > ")
		reply, _ := reader.ReadString('\n')
		reply = strings.TrimSpace(reply)
		if reply != "" && !strings.EqualFold(reply, "any") {
			extra := scraper.ParseQueryHeuristic(reply)
			if len(extra.Locations) > 0 {
				req.Locations = extra.Locations
			}
			if len(extra.TargetBatches) > 0 {
				req.TargetBatches = extra.TargetBatches
			}
			req.RawQuery = req.RawQuery + " | " + reply
		}
	}

	queries := scraper.Build16SiteSearchQueries(req)
	fmt.Printf("\n[Stage 1: Go Discovery] Generated %d targeted search queries across all 16 portals:\n", len(queries))
	for i := 0; i < len(queries) && i < 4; i++ {
		fmt.Printf("  [%d] %s\n", i+1, queries[i])
	}

	seenStore := scraper.NewSeenJobsStore("data/seen_jobs.json")
	portals := scraper.BuildDirectPortalURLs(req)

	fmt.Println("\n[Stage 2a: Go Concurrent Fetcher & Outer-First DOM Extractor] Scraping candidate portals...")
	var (
		mu            sync.Mutex
		allCandidates []scraper.DOMCandidate
		wg            sync.WaitGroup
		sem           = make(chan struct{}, 6)
	)

	maxPortals := 6
	if len(portals) < maxPortals {
		maxPortals = len(portals)
	}

	for _, p := range portals[:maxPortals] {
		wg.Add(1)
		go func(portal scraper.PortalConfig) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			htmlStr, err := scraper.FetchHTML(portal.EntryURL, 12*time.Second)
			if err != nil || len(htmlStr) == 0 {
				return
			}
			cands := scraper.ExtractDOMCandidates(htmlStr, portal.EntryURL, portal.ID)
			var unseen []scraper.DOMCandidate
			for _, c := range cands {
				if !seenStore.IsURLSeen(c.URL) {
					unseen = append(unseen, c)
				}
			}
			mu.Lock()
			allCandidates = append(allCandidates, unseen...)
			mu.Unlock()
		}(p)
	}
	wg.Wait()

	if len(allCandidates) > 0 {
		fmt.Printf("[Stage 2b: Python ML Bridge] Sending %d DOM candidates to Bi-Encoder + Cross-Encoder...\n", len(allCandidates))
		mlResp, err := scraper.CallPythonMLBridge(*pythonBin, req, allCandidates, 15)
		if err == nil {
			fmt.Printf("  -> Bi-Encoder kept %d job_listing & %d pagination_or_filter | Dropped %d noise candidates\n",
				len(mlResp.JobListings), len(mlResp.PaginationLinks), mlResp.NoiseCount)
		}
	}

	fmt.Println("\n[Stage 3 & 4: Full Detail Extraction & 4-Layer Cleaning -> jobs.json]")
	cmd := exec.Command(*pythonBin, "main.py", "--query", req.RawQuery, "--yes")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	_ = cmd.Run()
}

