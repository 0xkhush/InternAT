package scraper

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
)

// PortalConfig defines one of the 16 target job portals.
type PortalConfig struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Domain   string `json:"domain"`
	EntryURL string `json:"entry_url"`
	Render   string `json:"render"`
}

// GetTargetPortals returns all 16 configured Indian job and internship portals.
func GetTargetPortals() []PortalConfig {
	return []PortalConfig{
		{ID: "thejobcompany", Name: "The Job Company", Domain: "thejobcompany.co.in", EntryURL: "https://thejobcompany.co.in/", Render: "auto"},
		{ID: "careersat", Name: "CareersAt.Tech", Domain: "careersat.tech", EntryURL: "https://careersat.tech/jobs", Render: "always"},
		{ID: "hiredoor", Name: "Hiredoor", Domain: "hiredoor.in", EntryURL: "https://hiredoor.in/", Render: "always"},
		{ID: "freshersworld", Name: "Freshersworld", Domain: "freshersworld.com", EntryURL: "https://www.freshersworld.com/jobs/category/internship-job-vacancies", Render: "always"},
		{ID: "offcampus", Name: "OffCampusJobs4u", Domain: "offcampusjobs4u.com", EntryURL: "https://offcampusjobs4u.com/", Render: "auto"},
		{ID: "talentbattle", Name: "Talent Battle", Domain: "talentbattle.in", EntryURL: "https://talentbattle.in/", Render: "auto"},
		{ID: "internshala", Name: "Internshala", Domain: "internshala.com", EntryURL: "https://internshala.com/internships/", Render: "always"},
		{ID: "unstop", Name: "Unstop", Domain: "unstop.com", EntryURL: "https://unstop.com/internship", Render: "always"},
		{ID: "wellfound", Name: "Wellfound", Domain: "wellfound.com", EntryURL: "https://wellfound.com/jobs", Render: "auto"},
		{ID: "instahyre", Name: "Instahyre", Domain: "instahyre.com", EntryURL: "https://www.instahyre.com/search-jobs/", Render: "always"},
		{ID: "cutshort", Name: "Cutshort", Domain: "cutshort.io", EntryURL: "https://cutshort.io/jobs", Render: "auto"},
		{ID: "hasjob", Name: "Hasjob", Domain: "hasjob.co", EntryURL: "https://hasjob.co/", Render: "always"},
		{ID: "indeed", Name: "Indeed India", Domain: "in.indeed.com", EntryURL: "https://in.indeed.com/jobs", Render: "always"},
		{ID: "linkedin", Name: "LinkedIn Jobs", Domain: "linkedin.com", EntryURL: "https://www.linkedin.com/jobs/search/", Render: "never"},
		{ID: "cuvette", Name: "Cuvette", Domain: "cuvette.tech", EntryURL: "https://cuvette.tech/", Render: "auto"},
		{ID: "prosple", Name: "Prosple India", Domain: "in.prosple.com", EntryURL: "https://in.prosple.com/", Render: "auto"},
		{ID: "interndoor", Name: "InternDoor", Domain: "interndoor.com", EntryURL: "https://interndoor.com/", Render: "auto"},
	}
}

// JobSearchRequirement holds structured user criteria gathered by the CLI agent.
type JobSearchRequirement struct {
	RawQuery           string   `json:"raw_query"`
	RoleKeywords       []string `json:"role_keywords"`
	JobType            string   `json:"job_type"`
	TargetBatches      []string `json:"target_batches"`
	ExperienceLevel    string   `json:"experience_level"`
	Locations          []string `json:"locations"`
	WorkMode           string   `json:"work_mode"`
	MinStipendOrSalary string   `json:"min_stipend_or_salary,omitempty"`
	Skills             []string `json:"skills"`
}

// DOMCandidate represents an extracted link/card candidate from a web page.
type DOMCandidate struct {
	URL            string  `json:"url"`
	URLPath        string  `json:"url_path"`
	Text           string  `json:"text"`
	Context        string  `json:"context"`
	Region         string  `json:"region"`
	TagPath        string  `json:"tag_path"`
	SiblingCount   int     `json:"sibling_count"`
	SourcePortal   string  `json:"source_portal"`
	PredictedLabel string  `json:"predicted_label,omitempty"`
	Confidence     float64 `json:"confidence"`
	RelevanceScore float64 `json:"relevance_score"`
}

// ToFeatureString formats the candidate into the Bi-Encoder input representation.
func (c DOMCandidate) ToFeatureString() string {
	return fmt.Sprintf("[REGION] %s | [SIBLINGS] %d | [TEXT] %s | [CONTEXT] %s | [URL] %s",
		c.Region, c.SiblingCount, c.Text, c.Context, c.URLPath)
}

// JobRecord represents a clean, validated job or internship entry in jobs.json.
type JobRecord struct {
	JobID               string   `json:"job_id"`
	Title               string   `json:"title"`
	Company             string   `json:"company"`
	JobType             string   `json:"job_type"`
	BatchEligible       []string `json:"batch_eligible"`
	ExperienceRequired  string   `json:"experience_required"`
	Location            []string `json:"location"`
	WorkMode            string   `json:"work_mode"`
	StipendOrSalary     string   `json:"stipend_or_salary"`
	Duration            string   `json:"duration,omitempty"`
	SkillsRequired      []string `json:"skills_required"`
	EligibilityCriteria string   `json:"eligibility_criteria"`
	JobDescription      string   `json:"job_description"`
	PostedDate          string   `json:"posted_date"`
	ApplicationDeadline string   `json:"application_deadline"`
	ApplyURL            string   `json:"apply_url"`
	SourceURL           string   `json:"source_url"`
	SourcePortal        string   `json:"source_portal"`
	RelevanceScore      float64  `json:"relevance_score"`
	ScrapedAt           string   `json:"scraped_at"`
}

// ComputeJobID returns a deterministic 16-char SHA-256 hash ID for a job record.
func ComputeJobID(company, title, applyURL string) string {
	payload := fmt.Sprintf("%s|%s|%s",
		strings.ToLower(strings.TrimSpace(company)),
		strings.ToLower(strings.TrimSpace(title)),
		strings.ToLower(strings.TrimRight(strings.TrimSpace(applyURL), "/")),
	)
	sum := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(sum[:])[:16]
}

