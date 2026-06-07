"""
Low-level library for comparing citations and body-text overlap between two markdown reports.

Provides three core primitives:
  - extract_citations()    — parse the references section of a report into structured dicts
  - match_citations()      — match citations across two reports (exact URL, then fuzzy title)
  - compute_ngram_overlap() — measure body-text similarity via n-gram overlap

Can be run standalone as a CLI tool to inspect any two report files:
  python ablations/scripts/compare_reports.py report_v1.md report_v2.md
  python ablations/scripts/compare_reports.py report_v1.md report_v2.md --title-threshold 0.7
  python ablations/scripts/compare_reports.py report_v1.md report_v2.md --ngram 3

###---- IMPORTANT ----###
To run citation analysis across all tasks for a given model (gpt4.1, gpt4.1mini,
deepseekv4flash) and produce a full summary report, use analyse_citations.py instead:
  python ablations/scripts/analyse_citations.py gpt4.1
"""

import re
import sys
import difflib


def extract_citations(text):
    """
    Extract citations from markdown. Handles common formats:
      - [1] Title. Author. https://...
      - 1. Author (Year). Title. [url](url)
      - [Author (Year)] Title. url
    Returns list of dicts: {raw, title, url, index}
    """
    citations = []
    lines = text.split('\n')

    in_sources = False
    for line in lines:
        # Detect start of sources/references section
        if re.match(r'^#{1,4}\s*(?:\d+[\.\)]\s*)?(Sources|References|Bibliography|Works Cited|Citations)', line, re.I):
            in_sources = True
            continue

        # Detect next section header (end of sources)
        if in_sources and re.match(r'^#{1,4}\s+\S', line) and not re.match(
                r'^#{1,4}\s*(?:\d+[\.\)]\s*)?(Sources|References|Bibliography|Works Cited|Citations)', line, re.I):
            in_sources = False
            continue

        if not in_sources:
            continue

        line = line.strip()
        if not line or line.startswith('---'):
            continue

        # Extract URL(s)
        urls = re.findall(r'https?://[^\s\)>\]]+', line)
        url = urls[0].rstrip('.,;)') if urls else ''

        # Extract title: strip leading number/bracket, strip URL, strip markdown links
        title = line
        title = re.sub(r'^\[?\d+\]?[\.\)\s]+', '', title)  # leading [1]. or 1.
        title = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', title)  # [text](url) -> text
        title = re.sub(r'https?://\S+', '', title)  # remove bare URLs
        title = re.sub(r'[*_`]', '', title)  # remove formatting
        title = re.sub(r'\s+', ' ', title).strip().rstrip('.')

        if len(title) < 5 and not url:
            continue

        citations.append({
            'raw': line,
            'title': title,
            'url': url,
            'index': len(citations) + 1
        })

    return citations


def normalize_url(url):
    """Normalize URL for comparison: strip trailing slashes, www, protocol."""
    url = url.lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    url = url.rstrip('/')
    return url


def match_citations(cites_a, cites_b, title_threshold=0.65):
    """
    Match citations between two lists.
    Returns: url_matches, title_matches, unmatched_a, unmatched_b
    """
    # Normalize URLs
    urls_a = {normalize_url(c['url']): c for c in cites_a if c['url']}
    urls_b = {normalize_url(c['url']): c for c in cites_b if c['url']}

    # Phase 1: Exact URL matches
    shared_urls = set(urls_a.keys()) & set(urls_b.keys())
    url_matches = []
    matched_a_indices = set()
    matched_b_indices = set()

    for url in shared_urls:
        ca = urls_a[url]
        cb = urls_b[url]
        url_matches.append((ca, cb, 1.0, 'url'))
        matched_a_indices.add(ca['index'])
        matched_b_indices.add(cb['index'])

    # Phase 2: Fuzzy title matching on remaining
    remaining_a = [c for c in cites_a if c['index'] not in matched_a_indices]
    remaining_b = [c for c in cites_b if c['index'] not in matched_b_indices]

    title_matches = []
    title_matched_b = set()

    for ca in remaining_a:
        if not ca['title']:
            continue
        best_ratio, best_cb = 0, None
        for cb in remaining_b:
            if cb['index'] in title_matched_b or not cb['title']:
                continue
            ratio = difflib.SequenceMatcher(None,
                                           ca['title'].lower(),
                                           cb['title'].lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_cb = cb

        if best_ratio >= title_threshold and best_cb is not None:
            title_matches.append((ca, best_cb, best_ratio, 'title'))
            matched_a_indices.add(ca['index'])
            matched_b_indices.add(best_cb['index'])
            title_matched_b.add(best_cb['index'])

    unmatched_a = [c for c in cites_a if c['index'] not in matched_a_indices]
    unmatched_b = [c for c in cites_b if c['index'] not in matched_b_indices]

    return url_matches, title_matches, unmatched_a, unmatched_b


# ---------------------------------------------------------------------------
# N-gram overlap
# ---------------------------------------------------------------------------

_MD_STRIP_RE = re.compile(r'\[([^\]]*)\]\([^\)]*\)')   # [text](url) -> text
_URL_RE      = re.compile(r'https?://\S+')
_NONWORD_RE  = re.compile(r'[^a-z0-9\s]')
_WS_RE       = re.compile(r'\s+')


def _body_text(text: str) -> str:
    """Strip markdown formatting and the references section from report text."""
    # Drop everything from the first references/sources heading onward
    text = re.split(
        r'\n#{1,4}\s*(?:\d+[\.\)]\s*)?(Sources|References|Bibliography|Works Cited|Citations)\b',
        text, maxsplit=1, flags=re.I
    )[0]
    text = _MD_STRIP_RE.sub(r'\1', text)   # [text](url) → text
    text = _URL_RE.sub(' ', text)          # drop bare URLs
    text = text.lower()
    text = _NONWORD_RE.sub(' ', text)
    return _WS_RE.sub(' ', text).strip()


def _ngrams(text: str, n: int) -> list[tuple]:
    tokens = text.split()
    return list(zip(*[tokens[i:] for i in range(n)]))


def compute_ngram_overlap(text_a: str, text_b: str, n: int = 5) -> dict:
    """Compute n-gram overlap between two report texts.

    Returns precision (B→A), recall (A→B), F1, and raw counts.
    Here 'recall' answers: what fraction of A's n-grams appear in B?
    """
    body_a = _body_text(text_a)
    body_b = _body_text(text_b)
    grams_a = set(_ngrams(body_a, n))
    grams_b = set(_ngrams(body_b, n))
    shared  = grams_a & grams_b

    recall    = len(shared) / len(grams_a) if grams_a else None   # A retained in B
    precision = len(shared) / len(grams_b) if grams_b else None   # B sourced from A
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None
          and (precision + recall) > 0 else None)

    return {
        "n":          n,
        "ngrams_a":   len(grams_a),
        "ngrams_b":   len(grams_b),
        "shared":     len(shared),
        "recall":     round(recall    * 100, 2) if recall    is not None else None,
        "precision":  round(precision * 100, 2) if precision is not None else None,
        "f1":         round(f1        * 100, 2) if f1        is not None else None,
    }


def main():
    title_threshold = 0.65
    ngram_n = 5

    if '--title-threshold' in sys.argv:
        i = sys.argv.index('--title-threshold')
        title_threshold = float(sys.argv[i + 1])
        del sys.argv[i:i + 2]

    if '--ngram' in sys.argv:
        i = sys.argv.index('--ngram')
        ngram_n = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]

    path_a = sys.argv[1]
    path_b = sys.argv[2]

    text_a = open(path_a, encoding='utf-8').read()
    text_b = open(path_b, encoding='utf-8').read()

    cites_a = extract_citations(text_a)
    cites_b = extract_citations(text_b)

    print(f"Report A: {path_a}")
    print(f"Report B: {path_b}")
    print(f"Citations in A: {len(cites_a)}")
    print(f"Citations in B: {len(cites_b)}")
    print()

    url_matches, title_matches, unmatched_a, unmatched_b = match_citations(
        cites_a, cites_b, title_threshold)

    total_matches = len(url_matches) + len(title_matches)

    # Summary
    print(f"=== MATCH SUMMARY ===")
    print(f"URL matches:     {len(url_matches)}")
    print(f"Title matches:   {len(title_matches)} (threshold >= {title_threshold:.0%})")
    print(f"Total matched:   {total_matches}")
    print(f"A retained in B: {total_matches}/{len(cites_a)} ({total_matches / len(cites_a) * 100:.1f}%)" if cites_a else "")
    print(f"B from A:        {total_matches}/{len(cites_b)} ({total_matches / len(cites_b) * 100:.1f}%)" if cites_b else "")
    print(f"Only in A:       {len(unmatched_a)}")
    print(f"Only in B:       {len(unmatched_b)}")
    print()

    # N-gram overlap
    ng = compute_ngram_overlap(text_a, text_b, n=ngram_n)
    print(f"=== {ngram_n}-GRAM BODY OVERLAP ===")
    print(f"Unique {ngram_n}-grams in A: {ng['ngrams_a']}")
    print(f"Unique {ngram_n}-grams in B: {ng['ngrams_b']}")
    print(f"Shared:          {ng['shared']}")
    print(f"Recall  (A→B):   {ng['recall']}%   (fraction of A's {ngram_n}-grams present in B)")
    print(f"Precision (B→A): {ng['precision']}%   (fraction of B's {ngram_n}-grams sourced from A)")
    print(f"F1:              {ng['f1']}%")
    print()


if __name__ == '__main__':
    main()