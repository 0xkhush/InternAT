package scraper

import (
	"html"
	"net/url"
	"regexp"
	"strings"
)

var (
	stripBlocksRe = regexp.MustCompile(`(?is)<script\b[^>]*>.*?</script>|<style\b[^>]*>.*?</style>|<noscript\b[^>]*>.*?</noscript>|<svg\b[^>]*>.*?</svg>|<template\b[^>]*>.*?</template>|<canvas\b[^>]*>.*?</canvas>|<iframe\b[^>]*>.*?</iframe>`)
	commentRe     = regexp.MustCompile(`(?s)<!--.*?-->`)
	tagTokenRe    = regexp.MustCompile(`(?is)<(/?)([a-z0-9]+)((?:[^>"']|"[^"]*"|'[^']*')*)>`)
	hrefAttrRe    = regexp.MustCompile(`(?i)\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>"']+))`)
	hiddenAttrRe  = regexp.MustCompile(`(?i)(\bhidden\b|aria-hidden\s*=\s*["']true["'])`)
	htmlTagsRe    = regexp.MustCompile(`(?s)<[^>]+>`)
	wsRe          = regexp.MustCompile(`\s+`)
)

var regionPriority = map[string]int{
	"article": 6,
	"main":    5,
	"body":    4,
	"aside":   3,
	"nav":     2,
	"header":  1,
	"footer":  0,
}

type domNode struct {
	tag      string
	attrs    string
	href     string
	hidden   bool
	parent   *domNode
	children []*domNode
	textBuf  strings.Builder
}

func (n *domNode) fullText() string {
	if n == nil || n.hidden {
		return ""
	}
	var sb strings.Builder
	sb.WriteString(n.textBuf.String())
	for _, ch := range n.children {
		t := ch.fullText()
		if t != "" {
			sb.WriteString(" ")
			sb.WriteString(t)
		}
	}
	return cleanWhitespace(sb.String())
}

func cleanWhitespace(s string) string {
	unesc := html.UnescapeString(s)
	unesc = strings.ReplaceAll(unesc, "\u200b", "")
	unesc = strings.ReplaceAll(unesc, "\ufeff", "")
	return strings.TrimSpace(wsRe.ReplaceAllString(unesc, " "))
}

func isVoidElement(tag string) bool {
	switch tag {
	case "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr":
		return true
	}
	return false
}

func resolveNodeRegion(node *domNode) string {
	var chain []string
	for curr := node; curr != nil; curr = curr.parent {
		if curr.tag != "" {
			chain = append(chain, curr.tag)
		}
	}
	inMain := false
	// Walk from outermost (root/body) to innermost (the node itself)
	for i := len(chain) - 1; i >= 0; i-- {
		switch chain[i] {
		case "footer", "header", "aside", "nav":
			return chain[i]
		case "article":
			return "article"
		case "main":
			inMain = true
		}
	}
	if inMain {
		return "main"
	}
	return "body"
}

func buildNodeTagPath(node *domNode) string {
	var tags []string
	for curr := node; curr != nil; curr = curr.parent {
		if curr.tag != "" && curr.tag != "root" && curr.tag != "html" {
			tags = append(tags, curr.tag)
		}
	}
	for i, j := 0, len(tags)-1; i < j; i, j = i+1, j-1 {
		tags[i], tags[j] = tags[j], tags[i]
	}
	if len(tags) > 5 {
		tags = tags[len(tags)-5:]
	}
	return strings.Join(tags, " > ")
}

func findCardContainer(anchor *domNode) *domNode {
	for curr := anchor.parent; curr != nil; curr = curr.parent {
		switch curr.tag {
		case "article", "li", "tr":
			return curr
		case "main", "body", "footer", "header", "nav", "aside", "root":
			goto fallback
		}
	}
fallback:
	best := anchor
	for curr := anchor.parent; curr != nil; curr = curr.parent {
		switch curr.tag {
		case "main", "body", "footer", "header", "nav", "aside", "root":
			return best
		}
		txt := curr.fullText()
		if len(txt) >= 20 && len(txt) <= 600 {
			best = curr
			if countSameTagSiblings(curr) >= 2 {
				return curr
			}
		}
	}
	return best
}

func countSameTagSiblings(container *domNode) int {
	if container == nil || container.parent == nil {
		return 1
	}
	count := 0
	for _, ch := range container.parent.children {
		if ch.tag == container.tag && !ch.hidden {
			count++
		}
	}
	if count < 1 {
		return 1
	}
	return count
}

func resolveURL(baseURL, href string) (string, string) {
	base, err := url.Parse(baseURL)
	if err != nil {
		return "", ""
	}
	rel, err := url.Parse(strings.TrimSpace(href))
	if err != nil {
		return "", ""
	}
	resolved := base.ResolveReference(rel).String()
	norm := NormalizeURL(resolved)
	if norm == "" {
		return "", ""
	}
	parsedNorm, err := url.Parse(norm)
	if err != nil {
		return norm, "/"
	}
	path := parsedNorm.Path
	if path == "" {
		path = "/"
	}
	if parsedNorm.RawQuery != "" {
		path = path + "?" + parsedNorm.RawQuery
	}
	return norm, path
}

// ExtractDOMCandidates parses HTML in Go, resolves outer-first semantic regions, computes sibling_count, and deduplicates per page.
func ExtractDOMCandidates(rawHTML, baseURL, sourcePortal string) []DOMCandidate {
	cleaned := commentRe.ReplaceAllString(rawHTML, " ")
	cleaned = stripBlocksRe.ReplaceAllString(cleaned, " ")

	root := &domNode{tag: "root"}
	stack := []*domNode{root}
	var anchors []*domNode

	matches := tagTokenRe.FindAllStringSubmatchIndex(cleaned, -1)
	lastIdx := 0

	for _, m := range matches {
		start, end := m[0], m[1]
		if start > lastIdx {
			txt := cleanWhitespace(htmlTagsRe.ReplaceAllString(cleaned[lastIdx:start], " "))
			if txt != "" {
				top := stack[len(stack)-1]
				if top.textBuf.Len() > 0 {
					top.textBuf.WriteString(" ")
				}
				top.textBuf.WriteString(txt)
			}
		}
		lastIdx = end

		isClosing := cleaned[m[2]:m[3]] == "/"
		tagName := strings.ToLower(cleaned[m[4]:m[5]])
		attrs := ""
		if m[6] != -1 && m[7] != -1 {
			attrs = cleaned[m[6]:m[7]]
		}

		if isClosing {
			for i := len(stack) - 1; i >= 1; i-- {
				if stack[i].tag == tagName {
					stack = stack[:i]
					break
				}
			}
			continue
		}

		parent := stack[len(stack)-1]
		isNextStreamingDiv := tagName == "div" && (strings.Contains(attrs, `id="S:`) || strings.Contains(attrs, `id="B:`))
		node := &domNode{
			tag:    tagName,
			attrs:  attrs,
			hidden: parent.hidden || (!isNextStreamingDiv && hiddenAttrRe.MatchString(attrs)),
			parent: parent,
		}
		if tagName == "a" {
			if hm := hrefAttrRe.FindStringSubmatch(attrs); len(hm) > 0 {
				href := hm[1]
				if href == "" {
					href = hm[2]
				}
				if href == "" {
					href = hm[3]
				}
				node.href = strings.TrimSpace(html.UnescapeString(href))
			}
			anchors = append(anchors, node)
		}
		parent.children = append(parent.children, node)

		if !isVoidElement(tagName) && !strings.HasSuffix(strings.TrimSpace(attrs), "/") {
			stack = append(stack, node)
		}
	}

	dedup := map[string]DOMCandidate{}
	var order []string

	for _, a := range anchors {
		if a.hidden || a.href == "" {
			continue
		}
		hrefLower := strings.ToLower(a.href)
		if strings.HasPrefix(hrefLower, "#") || strings.HasPrefix(hrefLower, "javascript:") ||
			strings.HasPrefix(hrefLower, "mailto:") || strings.HasPrefix(hrefLower, "tel:") {
			continue
		}

		normURL, urlPath := resolveURL(baseURL, a.href)
		if normURL == "" {
			continue
		}

		text := a.fullText()
		if text == "" {
			continue
		}
		if len(text) > 200 {
			text = text[:200]
		}

		region := resolveNodeRegion(a)
		container := findCardContainer(a)
		sibCount := countSameTagSiblings(container)
		ctx := container.fullText()
		if len(ctx) > 400 {
			ctx = ctx[:400]
		}
		tagPath := buildNodeTagPath(a)

		cand := DOMCandidate{
			URL:          normURL,
			URLPath:      urlPath,
			Text:         text,
			Context:      ctx,
			Region:       region,
			TagPath:      tagPath,
			SiblingCount: sibCount,
			SourcePortal: sourcePortal,
		}

		existing, exists := dedup[normURL]
		if !exists {
			dedup[normURL] = cand
			order = append(order, normURL)
		} else {
			currPrio := regionPriority[cand.Region]*10000 + len(cand.Context)*10 + len(cand.Text)
			existPrio := regionPriority[existing.Region]*10000 + len(existing.Context)*10 + len(existing.Text)
			if currPrio > existPrio {
				dedup[normURL] = cand
			}
		}
	}

	out := make([]DOMCandidate, 0, len(order))
	for _, u := range order {
		out = append(out, dedup[u])
	}
	return out
}
