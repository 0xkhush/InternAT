from scraper.extractor import extract_dom_candidates
from scraper.fetcher import needs_js_rendering


SAMPLE_CAREERSAT_HTML = """
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
"""


def test_outer_first_region_and_deduplication():
    candidates = extract_dom_candidates(SAMPLE_CAREERSAT_HTML, "https://careersat.tech/jobs", "careersat")
    by_path = {c.url_path: c for c in candidates}

    # 1. Nested <nav> inside <footer> MUST be tagged region="footer", not "nav"
    assert by_path["/privacy"].region == "footer"

    # 2. Duplicate URL in <header><nav>, <footer><nav>, and <main><article> must keep <article> with richest context
    honeywell = by_path["/jobs/honeywell-aerospace-internship-2027"]
    assert honeywell.region == "article"
    assert "Batch 2026–27" in honeywell.context
    assert honeywell.sibling_count >= 2


def test_needs_js_rendering_detects_empty_spa_shell():
    empty_react = "<html><body><div id='root'></div><script src='app.js'></script></body></html>"
    rich_ssr = SAMPLE_CAREERSAT_HTML
    assert needs_js_rendering(empty_react) is True
    assert needs_js_rendering(rich_ssr) is False
