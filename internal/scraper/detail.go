package scraper

import (
	"encoding/json"
	"html"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"
)

var (
	jsonldScriptRe      = regexp.MustCompile(`(?is)<script[^>]+type\s*=\s*["']application/ld\+json["'][^>]*>(.*?)</script>`)
	trailingCommaRe     = regexp.MustCompile(`,\s*([}\]])`)
	promoLineRe         = regexp.MustCompile(`(?i)(join\s+(?:our\s+)?(?:whatsapp|telegram)|whatsapp\s+(?:group|channel)|telegram\s+(?:group|channel)|t\.me/|whatsapp\.com/channel|follow\s+us\s+on\s+(?:instagram|linkedin|twitter|telegram|whatsapp)|subscribe\s+to\s+our\s+newsletter|free\s+interview\s+prep)`)
	companyNoiseRe      = regexp.MustCompile(`(?i)\s*(?:hiring\s+off\s+campus|off\s+campus\s+drive|hiring\s+freshers|hiring|recruitment\s*\d*|careers?)\s*$`)
	externalApplyTextRe = regexp.MustCompile(`(?i)\b(apply\s*now|apply\s*here|apply\s*online|apply\s*on\s*company|registration\s*link|application\s*link)\b`)
	h1Re                = regexp.MustCompile(`(?is)<h1\b[^>]*>(.*?)</h1>`)
	pTagRe              = regexp.MustCompile(`(?is)<(p|li|br|div)\b[^>]*>`)
)

// CleanTextBlock strips HTML tags, entities, and WhatsApp/Telegram promotional lines in Go.
func CleanTextBlock(raw string) string {
	if strings.TrimSpace(raw) == "" {
		return ""
	}
	unesc := html.UnescapeString(raw)
	withBreaks := pTagRe.ReplaceAllString(unesc, "\n")
	noTags := htmlTagsRe.ReplaceAllString(withBreaks, " ")

	var lines []string
	seen := map[string]bool{}
	for _, ln := range strings.Split(noTags, "\n") {
		cleanLn := strings.TrimSpace(wsRe.ReplaceAllString(ln, " "))
		if cleanLn == "" {
			continue
		}
		if promoLineRe.MatchString(cleanLn) {
			continue
		}
		key := strings.ToLower(cleanLn)
		if !seen[key] {
			seen[key] = true
			lines = append(lines, cleanLn)
		}
	}
	return strings.Join(lines, "\n")
}

// CleanCompanyName strips 'Hiring Off Campus', 'Hiring', 'Recruitment' suffixes.
func CleanCompanyName(raw string) string {
	cleaned := strings.TrimSpace(html.UnescapeString(raw))
	cleaned = companyNoiseRe.ReplaceAllString(cleaned, "")
	cleaned = strings.Trim(cleaned, " -–|•:")
	if cleaned == "" {
		return "Not Disclosed"
	}
	return cleaned
}

func findJobPostingJSONLD(v interface{}) map[string]interface{} {
	switch val := v.(type) {
	case map[string]interface{}:
		if t, ok := val["@type"].(string); ok && t == "JobPosting" {
			return val
		}
		if graph, ok := val["@graph"].([]interface{}); ok {
			for _, item := range graph {
				if found := findJobPostingJSONLD(item); found != nil {
					return found
				}
			}
		}
		for _, child := range val {
			if found := findJobPostingJSONLD(child); found != nil {
				return found
			}
		}
	case []interface{}:
		for _, item := range val {
			if found := findJobPostingJSONLD(item); found != nil {
				return found
			}
		}
	}
	return nil
}

func resolveOutboundApplyURL(rawHTML, sourceURL string) string {
	srcParsed, err := url.Parse(sourceURL)
	srcHost := ""
	if err == nil {
		srcHost = strings.ToLower(srcParsed.Hostname())
	}
	cands := ExtractDOMCandidates(rawHTML, sourceURL, "detail")
	for _, c := range cands {
		uLower := strings.ToLower(c.URL)
		if strings.Contains(uLower, "whatsapp.com") || strings.Contains(uLower, "t.me") ||
			strings.Contains(uLower, "telegram") || strings.Contains(uLower, "instagram.com") {
			continue
		}
		uParsed, err := url.Parse(c.URL)
		if err != nil {
			continue
		}
		host := strings.ToLower(uParsed.Hostname())
		if externalApplyTextRe.MatchString(c.Text) && host != "" && host != srcHost {
			return c.URL
		}
	}
	return sourceURL
}

// ExtractJobRecordFromHTML extracts a clean JobRecord from a detail page's HTML in Go (JSON-LD + 4-Layer Cleaning).
func ExtractJobRecordFromHTML(rawHTML, sourceURL, sourcePortal string, relevanceScore float64) JobRecord {
	var jobLD map[string]interface{}
	matches := jsonldScriptRe.FindAllStringSubmatch(rawHTML, -1)
	for _, m := range matches {
		sanitized := trailingCommaRe.ReplaceAllString(strings.TrimSpace(m[1]), "$1")
		var parsed interface{}
		if err := json.Unmarshal([]byte(sanitized), &parsed); err == nil {
			if found := findJobPostingJSONLD(parsed); found != nil {
				jobLD = found
				break
			}
		}
	}

	title := ""
	company := "Not Disclosed"
	desc := ""
	postedDate := "Not Specified"

	if jobLD != nil {
		if t, ok := jobLD["title"].(string); ok {
			title = strings.TrimSpace(t)
		}
		if org, ok := jobLD["hiringOrganization"].(map[string]interface{}); ok {
			if n, ok := org["name"].(string); ok {
				company = CleanCompanyName(n)
			}
		}
		if d, ok := jobLD["description"].(string); ok {
			desc = CleanTextBlock(d)
		}
		if pd, ok := jobLD["datePosted"].(string); ok && strings.TrimSpace(pd) != "" {
			postedDate = strings.TrimSpace(pd)
		}
	}

	if title == "" {
		if hm := h1Re.FindStringSubmatch(rawHTML); len(hm) > 1 {
			title = CleanTextBlock(hm[1])
		}
	}
	if desc == "" {
		desc = CleanTextBlock(rawHTML)
		if len(desc) > 600 {
			desc = desc[:600]
		}
	}

	applyURL := resolveOutboundApplyURL(rawHTML, sourceURL)
	batches := batchYearRe.FindAllString(title+" "+desc, -1)
	if len(batches) == 0 {
		batches = []string{"Not Specified"}
	}

	rec := JobRecord{
		JobID:               ComputeJobID(company, title, applyURL),
		Title:               title,
		Company:             company,
		JobType:             "Internship",
		BatchEligible:       batches,
		ExperienceRequired:  "0 years (Fresher / Student)",
		Location:            []string{"India"},
		WorkMode:            "Not Specified",
		StipendOrSalary:     "Not Disclosed",
		SkillsRequired:      []string{},
		EligibilityCriteria: "Not Specified",
		JobDescription:      desc,
		PostedDate:          postedDate,
		ApplicationDeadline: "Not Specified",
		ApplyURL:            applyURL,
		SourceURL:           sourceURL,
		SourcePortal:        sourcePortal,
		RelevanceScore:      relevanceScore,
		ScrapedAt:           time.Now().UTC().Format(time.RFC3339),
	}
	return rec
}

// SaveJobsJSON appends deduplicated JobRecords into jobs.json.
func SaveJobsJSON(path string, records []JobRecord) (int, error) {
	var existing []JobRecord
	if data, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(data, &existing)
	}
	seenIDs := map[string]bool{}
	for _, r := range existing {
		seenIDs[r.JobID] = true
	}
	added := 0
	for _, r := range records {
		if r.Title == "" || seenIDs[r.JobID] {
			continue
		}
		existing = append(existing, r)
		seenIDs[r.JobID] = true
		added++
	}
	b, err := json.MarshalIndent(existing, "", "  ")
	if err != nil {
		return 0, err
	}
	return added, os.WriteFile(path, b, 0o644)
}
