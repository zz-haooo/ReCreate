#!/usr/bin/env python3
"""
Web Search Tool - ReCreate-Agent web search tool.

Features:
1. search  - General web search (DuckDuckGo)
2. github  - GitHub search (code, repos, issues)
3. fetch   - Fetch webpage content (plain text)
4. stackoverflow - Stack Overflow search

Usage:
    # General search
    python3 web_search.py search "Django URLValidator fix"
    
    # GitHub search
    python3 web_search.py github code "URLValidator" --language python
    python3 web_search.py github issues "URLValidator bug" --repo django/django
    python3 web_search.py github repos "swe-agent"
    
    # Fetch webpage
    python3 web_search.py fetch "https://docs.python.org/3/library/re.html"
    
    # Stack Overflow search
    python3 web_search.py stackoverflow "python async context manager"
"""

import argparse
import html
import json
import os
import re
import sys
from urllib.request import Request, urlopen
from urllib.parse import quote, urlencode, quote_plus
from urllib.error import HTTPError, URLError
import ssl

# Disable SSL verification (needed in some environments)
ssl._create_default_https_context = ssl._create_unverified_context

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    import warnings
    warnings.warn(
        "GITHUB_TOKEN not set. GitHub search functionality will be limited. "
        "Set GITHUB_TOKEN environment variable to enable full GitHub API access."
    )

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def http_get(url: str, headers: dict = None, timeout: int = 30) -> str:
    """Send HTTP GET request."""
    default_headers = {"User-Agent": USER_AGENT}
    if headers:
        default_headers.update(headers)
    
    req = Request(url, headers=default_headers)
    try:
        with urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError) as e:
        return f"Error: {e}"


def html_to_text(html_content: str, max_length: int = 8000) -> str:
    """Convert HTML to plain text."""
    # Remove script and style
    text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Handle common tags
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?p[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?div[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?li[^>]*>', '\n• ', text, flags=re.IGNORECASE)
    text = re.sub(r'<h[1-6][^>]*>', '\n## ', text, flags=re.IGNORECASE)
    text = re.sub(r'</h[1-6]>', '\n', text, flags=re.IGNORECASE)
    
    # Preserve code block content
    text = re.sub(r'<pre[^>]*>(.*?)</pre>', r'\n```\n\1\n```\n', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove all other tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Clean whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    
    # Truncate
    if len(text) > max_length:
        text = text[:max_length] + "\n... (truncated)"
    
    return text


# ============== DuckDuckGo Search ==============

def search_duckduckgo(query: str, limit: int = 10) -> str:
    """Search using DuckDuckGo."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    content = http_get(url)
    
    if content.startswith("Error:"):
        return content
    
    # Parse results
    results = []
    pattern = r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
    matches = re.findall(pattern, content, re.IGNORECASE)
    
    snippet_pattern = r'<a[^>]+class="result__snippet"[^>]*>([^<]+(?:<[^>]+>[^<]*)*)</a>'
    snippets = re.findall(snippet_pattern, content, re.IGNORECASE)
    
    lines = [f"Search results for: {query}\n"]
    
    for i, (url, title) in enumerate(matches[:limit]):
        # Clean DuckDuckGo redirect URL
        if "uddg=" in url:
            url = re.search(r'uddg=([^&]+)', url)
            if url:
                from urllib.parse import unquote
                url = unquote(url.group(1))
            else:
                continue
        
        title = html.unescape(re.sub(r'<[^>]+>', '', title)).strip()
        snippet = ""
        if i < len(snippets):
            snippet = html.unescape(re.sub(r'<[^>]+>', '', snippets[i])).strip()[:150]
        
        lines.append(f"{i+1}. {title}")
        lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    
    if len(lines) <= 1:
        return "No results found. Try different keywords."
    
    return "\n".join(lines)


# ============== GitHub Search ==============

def github_api(endpoint: str, params: dict = None) -> dict:
    """GitHub API request."""
    url = f"https://api.github.com{endpoint}"
    if params:
        url += "?" + urlencode(params)
    
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    }
    
    content = http_get(url, headers)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": content}


def github_search_code(query: str, language: str = None, repo: str = None, limit: int = 10) -> str:
    """GitHub code search."""
    q = query
    if language:
        q += f" language:{language}"
    if repo:
        q += f" repo:{repo}"
    
    result = github_api("/search/code", {"q": q, "per_page": limit})
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    lines = [f"GitHub Code: {result.get('total_count', 0)} results for '{query}'\n"]
    
    for item in result.get("items", [])[:limit]:
        repo_name = item.get("repository", {}).get("full_name", "")
        path = item.get("path", "")
        lines.append(f"📄 {repo_name}/{path}")
        lines.append(f"   {item.get('html_url', '')}")
        lines.append("")
    
    return "\n".join(lines) if len(lines) > 1 else "No code found."


def github_search_repos(query: str, sort: str = "stars", limit: int = 10) -> str:
    """GitHub repository search."""
    params = {"q": query, "per_page": limit}
    if sort in ["stars", "forks", "updated"]:
        params["sort"] = sort
    
    result = github_api("/search/repositories", params)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    lines = [f"GitHub Repos: {result.get('total_count', 0)} results for '{query}'\n"]
    
    for item in result.get("items", [])[:limit]:
        name = item.get("full_name", "")
        desc = (item.get("description") or "")[:80]
        stars = item.get("stargazers_count", 0)
        lang = item.get("language") or ""
        lines.append(f"⭐ {stars:,} | {name} [{lang}]")
        if desc:
            lines.append(f"   {desc}")
        lines.append(f"   {item.get('html_url', '')}")
        lines.append("")
    
    return "\n".join(lines) if len(lines) > 1 else "No repos found."


def github_search_issues(query: str, repo: str = None, state: str = "all", limit: int = 10) -> str:
    """GitHub issue search."""
    q = query + " is:issue"
    if repo:
        q += f" repo:{repo}"
    if state in ["open", "closed"]:
        q += f" is:{state}"
    
    url = f"https://api.github.com/search/issues?q={quote(q)}&per_page={limit}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    }
    content = http_get(url, headers)
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"error": content}
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    lines = [f"GitHub Issues: {result.get('total_count', 0)} results for '{query}'\n"]
    
    for item in result.get("items", [])[:limit]:
        number = item.get("number", "")
        title = item.get("title", "")[:60]
        state = item.get("state", "")
        repo_url = item.get("repository_url", "").replace("https://api.github.com/repos/", "")
        icon = "🟢" if state == "open" else "🔴"
        lines.append(f"{icon} #{number} [{repo_url}]")
        lines.append(f"   {title}")
        lines.append(f"   {item.get('html_url', '')}")
        lines.append("")
    
    return "\n".join(lines) if len(lines) > 1 else "No issues found."


def github_get_file(repo: str, path: str, ref: str = "main") -> str:
    """Get GitHub file content."""
    result = github_api(f"/repos/{repo}/contents/{path}", {"ref": ref})
    
    if "error" in result:
        if ref == "main":
            return github_get_file(repo, path, "master")
        return f"Error: {result['error']}"
    
    if result.get("encoding") == "base64":
        import base64
        content = base64.b64decode(result.get("content", "")).decode("utf-8", errors="replace")
        lines = [
            f"📄 {repo}/{path} (ref: {ref})",
            f"Size: {result.get('size', 0)} bytes",
            "-" * 60,
            content[:6000] + ("\n... (truncated)" if len(content) > 6000 else "")
        ]
        return "\n".join(lines)
    
    return f"Cannot decode: {result.get('type', 'unknown')}"


# ============== Stack Overflow Search ==============

def search_stackoverflow(query: str, limit: int = 10) -> str:
    """Stack Overflow search."""
    params = {
        "order": "desc",
        "sort": "relevance",
        "tagged": "python",
        "intitle": query,
        "site": "stackoverflow",
        "pagesize": limit,
    }
    url = "https://api.stackexchange.com/2.3/search?" + urlencode(params)
    
    content = http_get(url)
    try:
        import gzip
        import io
        req = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
        with urlopen(req, timeout=30) as response:
            if response.info().get('Content-Encoding') == 'gzip':
                content = gzip.GzipFile(fileobj=io.BytesIO(response.read())).read().decode('utf-8')
            else:
                content = response.read().decode('utf-8')
        result = json.loads(content)
    except Exception as e:
        return f"Error: {e}"
    
    if "error_message" in result:
        return f"Error: {result['error_message']}"
    
    lines = [f"Stack Overflow: {result.get('quota_remaining', '?')} API calls remaining\n"]
    lines.append(f"Results for: {query}\n")
    
    for item in result.get("items", [])[:limit]:
        title = html.unescape(item.get("title", ""))
        score = item.get("score", 0)
        answered = "✓" if item.get("is_answered") else "○"
        answers = item.get("answer_count", 0)
        link = item.get("link", "")
        
        lines.append(f"{answered} [{score:+d}] {title}")
        lines.append(f"   {answers} answers | {link}")
        lines.append("")
    
    return "\n".join(lines) if len(lines) > 2 else "No results found."


# ============== Webpage Fetch ==============

def fetch_webpage(url: str, max_length: int = 8000) -> str:
    """Fetch webpage content and convert to plain text."""
    content = http_get(url)
    
    if content.startswith("Error:"):
        return content
    
    # Extract title
    title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
    title = html.unescape(title_match.group(1)).strip() if title_match else "No title"
    
    # Convert to plain text
    text = html_to_text(content, max_length)
    
    lines = [
        f"📄 {url}",
        f"Title: {title}",
        "-" * 60,
        text
    ]
    
    return "\n".join(lines)


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(
        description="Web Search Tool for ReCreate-Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # search - General search
    p_search = subparsers.add_parser("search", help="General web search (DuckDuckGo)")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", "-n", type=int, default=10)
    
    # github - GitHub search
    p_github = subparsers.add_parser("github", help="GitHub search")
    p_github_sub = p_github.add_subparsers(dest="github_type", required=True)
    
    # github code
    p_gh_code = p_github_sub.add_parser("code", help="Search code")
    p_gh_code.add_argument("query")
    p_gh_code.add_argument("--language", "-l")
    p_gh_code.add_argument("--repo", "-r")
    p_gh_code.add_argument("--limit", "-n", type=int, default=10)
    
    # github repos
    p_gh_repos = p_github_sub.add_parser("repos", help="Search repositories")
    p_gh_repos.add_argument("query")
    p_gh_repos.add_argument("--sort", "-s", choices=["stars", "forks", "updated"], default="stars")
    p_gh_repos.add_argument("--limit", "-n", type=int, default=10)
    
    # github issues
    p_gh_issues = p_github_sub.add_parser("issues", help="Search issues")
    p_gh_issues.add_argument("query")
    p_gh_issues.add_argument("--repo", "-r")
    p_gh_issues.add_argument("--state", "-s", choices=["open", "closed", "all"], default="all")
    p_gh_issues.add_argument("--limit", "-n", type=int, default=10)
    
    # github file
    p_gh_file = p_github_sub.add_parser("file", help="Get file content")
    p_gh_file.add_argument("repo", help="owner/repo")
    p_gh_file.add_argument("path", help="file path")
    p_gh_file.add_argument("--ref", "-r", default="main")
    
    # fetch - Fetch webpage
    p_fetch = subparsers.add_parser("fetch", help="Fetch webpage content")
    p_fetch.add_argument("url", help="URL to fetch")
    p_fetch.add_argument("--max-length", "-m", type=int, default=8000)
    
    # stackoverflow
    p_so = subparsers.add_parser("stackoverflow", help="Search Stack Overflow")
    p_so.add_argument("query")
    p_so.add_argument("--limit", "-n", type=int, default=10)
    
    args = parser.parse_args()
    
    if args.command == "search":
        print(search_duckduckgo(args.query, args.limit))
    
    elif args.command == "github":
        if args.github_type == "code":
            print(github_search_code(args.query, args.language, args.repo, args.limit))
        elif args.github_type == "repos":
            print(github_search_repos(args.query, args.sort, args.limit))
        elif args.github_type == "issues":
            print(github_search_issues(args.query, args.repo, args.state, args.limit))
        elif args.github_type == "file":
            print(github_get_file(args.repo, args.path, args.ref))
    
    elif args.command == "fetch":
        print(fetch_webpage(args.url, args.max_length))
    
    elif args.command == "stackoverflow":
        print(search_stackoverflow(args.query, args.limit))


if __name__ == "__main__":
    main()
