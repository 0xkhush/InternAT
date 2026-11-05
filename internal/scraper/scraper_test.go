package scraper

import (
	"path/filepath"
	"strings"
	"testing"
)

const sampleCareersAtHTML = `
<html>
<body>
  <header><nav><a href="/jobs/honeywell-aerospace-internship-2027">Featured Job</a></nav></header>
  <main>
    <div class="feed">
      <article class="card">
        <h2><a href="/jobs/honeywell-aerospace-internship-2027?utm_source=feed">Honeywell Aerospace Internship 2027 – Software Intern</a></h2>
        <p>Honeywell Aerospace • Bangalore, Hyderabad • 1d ago • Internship • 0–1 yrs • Batch 2026–27</p>
      </article>
      <article class="card">
        <h2><a href="/jobs/rubrik-software-intern-2028">Rubrik Software Engineering Intern 2028</a></h2>
        <p>Rubrik • Bengaluru • Internship • Batch 2028 • ₹1,00,000/month</p>
      </article>
      <div class="promo">
        <a href="https://whatsapp.com/channel/0029VaUJgMW2">Join channel</a>
        <p>WhatsApp channel 52,000+ followers Verified fresher jobs</p>
      </div>
    </div>
    <div class="pagination">
      <a href="/jobs?page=2">Next Page</a>
    </div>
  </main>
  <footer>
    <nav aria-label="Footer">
      <a href="/privacy">Privacy Policy</a>
      <a href="/jobs/honeywell-aerospace-internship-2027">Honeywell Footer Link</a>
    </nav>
  </footer>
</body>
</html>
`

func TestGoOuterFirstDOMExtractorAnd16Sites(t *testing.T) {
	portals := GetTargetPortals()
	if len(portals) != 17 {
		t.Fatalf("expected 17 target portals, got %d", len(portals))
	}

	candidates := ExtractDOMCandidates(sampleCareersAtHTML, "https://careersat.tech/jobs", "careersat")
	byPath := map[string]DOMCandidate{}
	for _, c := range candidates {
		byPath[c.URLPath] = c
	}

	priv, ok := byPath["/privacy"]
	if !ok {
		t.Fatalf("expected /privacy candidate")
	}
	if priv.Region != "footer" {
		t.Fatalf("expected nested <nav> inside <footer> to resolve to footer, got %s", priv.Region)
	}

	hw, ok := byPath["/jobs/honeywell-aerospace-internship-2027"]
	if !ok {
		t.Fatalf("expected honeywell candidate")
	}
	if hw.Region != "article" {
		t.Fatalf("expected article region priority over header/footer, got %s", hw.Region)
	}
	if !strings.Contains(hw.Context, "Batch 2026–27") {
		t.Fatalf("expected rich card context, got %s", hw.Context)
	}
	if hw.SiblingCount < 2 {
		t.Fatalf("expected sibling_count >= 2, got %d", hw.SiblingCount)
	}
}

func TestGoDiscoveryQueriesAndSeenStore(t *testing.T) {
	req := ParseQueryHeuristic("2028 internship software engineering bangalore")
	queries := Build16SiteSearchQueries(req)
	joined := strings.Join(queries, " ")
	for _, p := range GetTargetPortals() {
		if !strings.Contains(joined, p.Domain) && !strings.Contains(joined, p.ID) {
			t.Fatalf("expected portal %s (%s) in queries", p.ID, p.Domain)
		}
	}

	tmpFile := filepath.Join(t.TempDir(), "seen_jobs.json")
	store := NewSeenJobsStore(tmpFile)
	rawURL := "https://careersat.tech/jobs/rubrik-swe-intern-2028/?utm_source=ddg&ref=1"
	if store.IsURLSeen(rawURL) {
		t.Fatalf("expected URL not seen initially")
	}
	store.MarkURLSeen("https://careersat.tech/jobs")
	if store.IsURLSeen("https://careersat.tech/jobs") {
		t.Fatalf("portal listing root URL must never be blocked as a seen detail URL")
	}
	store.MarkURLSeen(rawURL)
	if err := store.Save(); err != nil {
		t.Fatalf("failed to save store: %v", err)
	}

	reloaded := NewSeenJobsStore(tmpFile)
	if !reloaded.IsURLSeen("https://careersat.tech/jobs/rubrik-swe-intern-2028") {
		t.Fatalf("expected normalized detail URL to be marked seen after reload")
	}
}

func TestGoArticleInnerHeaderAndDetailExtraction(t *testing.T) {
	nestedHTML := `
	<html><body>
	  <main>
	    <article data-info="a > b">
	      <header><h2><a href="/jobs/atlassian-intern-2028" title="Click > Apply">Atlassian SWE Intern 2028</a></h2></header>
	      <footer><p>Bengaluru • Batch 2028 • ₹1,10,000/month</p></footer>
	    </article>
	  </main>
	  <footer><nav><a href="/privacy">Privacy</a></nav></footer>
	</body></html>`

	cands := ExtractDOMCandidates(nestedHTML, "https://careersat.tech/jobs", "careersat")
	byPath := map[string]DOMCandidate{}
	for _, c := range cands {
		byPath[c.URLPath] = c
	}
	if byPath["/jobs/atlassian-intern-2028"].Region != "article" {
		t.Fatalf("expected article region for link inside <article><header>, got %s", byPath["/jobs/atlassian-intern-2028"].Region)
	}
	if byPath["/privacy"].Region != "footer" {
		t.Fatalf("expected footer region for <footer ><nav>, got %s", byPath["/privacy"].Region)
	}

	detailHTML := `
	<html><head><script type="application/ld+json">
	{"@type":"JobPosting","title":"Atlassian SWE Intern 2028","hiringOrganization":{"name":"Atlassian Hiring"},"description":"<p>Build cloud backend in Go and Python.</p><p>Join our WhatsApp group!</p>"}
	</script></head><body><article><a href="https://atlassian.com/careers/1">Apply Now</a></article></body></html>`
	rec := ExtractJobRecordFromHTML(detailHTML, "https://careersat.tech/jobs/atlassian-intern-2028", "careersat", 0.96)
	if rec.Company != "Atlassian" {
		t.Fatalf("expected clean company Atlassian, got %s", rec.Company)
	}
	if strings.Contains(rec.JobDescription, "WhatsApp") {
		t.Fatalf("expected WhatsApp promo stripped from description, got %s", rec.JobDescription)
	}
	if rec.ApplyURL != "https://atlassian.com/careers/1" {
		t.Fatalf("expected outbound apply URL, got %s", rec.ApplyURL)
	}
}


