package scraper

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

var trackingPrefixes = []string{"utm_", "fbclid", "gclid", "ref", "trk", "source", "mc_", "_ga", "igshid"}
var multiSlashRe = regexp.MustCompile(`/+`)
var nonAlphaNumRe = regexp.MustCompile(`[^a-z0-9]+`)

// NormalizeURL strips tracking query parameters, fragments, and trailing slashes.
func NormalizeURL(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}
	u, err := url.Parse(trimmed)
	if err != nil {
		return trimmed
	}
	scheme := strings.ToLower(u.Scheme)
	if scheme == "" {
		scheme = "https"
	}
	host := strings.ToLower(u.Host)
	path := multiSlashRe.ReplaceAllString(u.Path, "/")
	if path == "" {
		path = "/"
	} else if path != "/" && strings.HasSuffix(path, "/") {
		path = strings.TrimRight(path, "/")
	}

	q := u.Query()
	cleanQ := url.Values{}
	for k, vals := range q {
		kLower := strings.ToLower(k)
		isTracking := false
		for _, pref := range trackingPrefixes {
			if kLower == pref || strings.HasPrefix(kLower, pref) {
				isTracking = true
				break
			}
		}
		if !isTracking {
			for _, v := range vals {
				if v != "" {
					cleanQ.Add(k, v)
				}
			}
		}
	}

	out := url.URL{
		Scheme:   scheme,
		Host:     host,
		Path:     path,
		RawQuery: cleanQ.Encode(),
	}
	return out.String()
}

func getPortalListingRoots() map[string]bool {
	roots := map[string]bool{}
	for _, p := range GetTargetPortals() {
		roots[NormalizeURL(p.EntryURL)] = true
		roots[NormalizeURL("https://"+p.Domain+"/")] = true
		roots[NormalizeURL("https://www."+p.Domain+"/")] = true
	}
	return roots
}

func fingerprint(company, title, portal string) string {
	c := strings.TrimSpace(nonAlphaNumRe.ReplaceAllString(strings.ToLower(company), " "))
	t := strings.TrimSpace(nonAlphaNumRe.ReplaceAllString(strings.ToLower(title), " "))
	p := strings.TrimSpace(strings.ToLower(portal))
	sum := sha256.Sum256([]byte(c + "|" + t + "|" + p))
	return hex.EncodeToString(sum[:])[:20]
}

// SeenJobsStore persists already-scraped URLs and job fingerprints so jobs are never listed twice.
type SeenJobsStore struct {
	Path             string
	SeenURLs         map[string]bool
	SeenFingerprints map[string]bool
	SeenJobIDs       map[string]bool
	listingRoots     map[string]bool
}

type seenStoreDisk struct {
	SeenURLs         []string `json:"seen_urls"`
	SeenFingerprints []string `json:"seen_fingerprints"`
	SeenJobIDs       []string `json:"seen_job_ids"`
}

// NewSeenJobsStore loads or initializes a persistent SeenJobsStore at path.
func NewSeenJobsStore(path string) *SeenJobsStore {
	s := &SeenJobsStore{
		Path:             path,
		SeenURLs:         map[string]bool{},
		SeenFingerprints: map[string]bool{},
		SeenJobIDs:       map[string]bool{},
		listingRoots:     getPortalListingRoots(),
	}
	s.load()
	return s
}

func (s *SeenJobsStore) load() {
	data, err := os.ReadFile(s.Path)
	if err != nil {
		return
	}
	var disk seenStoreDisk
	if err := json.Unmarshal(data, &disk); err != nil {
		return
	}
	for _, u := range disk.SeenURLs {
		s.SeenURLs[u] = true
	}
	for _, fp := range disk.SeenFingerprints {
		s.SeenFingerprints[fp] = true
	}
	for _, id := range disk.SeenJobIDs {
		s.SeenJobIDs[id] = true
	}
}

// IsURLSeen returns true if the detail URL was already scraped in an earlier run.
func (s *SeenJobsStore) IsURLSeen(rawURL string) bool {
	norm := NormalizeURL(rawURL)
	if norm == "" || s.listingRoots[norm] {
		return false
	}
	return s.SeenURLs[norm]
}

// MarkURLSeen marks a detail URL as seen (ignoring portal listing roots).
func (s *SeenJobsStore) MarkURLSeen(rawURL string) {
	norm := NormalizeURL(rawURL)
	if norm != "" && !s.listingRoots[norm] {
		s.SeenURLs[norm] = true
	}
}

// MarkRecordSeen records a full JobRecord as seen.
func (s *SeenJobsStore) MarkRecordSeen(rec JobRecord) {
	s.MarkURLSeen(rec.SourceURL)
	if rec.ApplyURL != "" {
		s.MarkURLSeen(rec.ApplyURL)
	}
	s.SeenFingerprints[fingerprint(rec.Company, rec.Title, rec.SourcePortal)] = true
	if rec.JobID != "" {
		s.SeenJobIDs[rec.JobID] = true
	}
}

// Save writes the SeenJobsStore to disk.
func (s *SeenJobsStore) Save() error {
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o755); err != nil {
		return err
	}
	disk := seenStoreDisk{
		SeenURLs:         sortedKeys(s.SeenURLs),
		SeenFingerprints: sortedKeys(s.SeenFingerprints),
		SeenJobIDs:       sortedKeys(s.SeenJobIDs),
	}
	b, err := json.MarshalIndent(disk, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.Path, b, 0o644)
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

