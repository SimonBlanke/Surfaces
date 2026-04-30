#!/usr/bin/env python3
"""Sync the [Unreleased] section of CHANGELOG.md to a GitHub Release draft.

Only manages a single draft release named "Unreleased". Published releases
are never touched. When the [Unreleased] section has content, the draft is
created or updated. When it is empty, the draft is deleted.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

UNRELEASED_HEADER = re.compile(r"^## \[Unreleased\]")
NEXT_HEADER = re.compile(r"^## \[")
LINK_REFERENCE = re.compile(r"^\[.+\]:\s*https?://")


def extract_unreleased(text):
    """Extract the body text of the [Unreleased] section."""
    lines = []
    in_unreleased = False

    for line in text.splitlines():
        if UNRELEASED_HEADER.match(line):
            in_unreleased = True
            continue
        if in_unreleased:
            if NEXT_HEADER.match(line):
                break
            if LINK_REFERENCE.match(line):
                continue
            lines.append(line)

    return "\n".join(lines).strip()


def github_api(method, endpoint, token, repo, payload=None):
    """Make a GitHub REST API request. Returns (status_code, parsed_body)."""
    url = f"https://api.github.com/repos/{repo}/{endpoint}"
    data = json.dumps(payload).encode("utf-8") if payload else None

    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data:
        req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8") if e.fp else ""
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        return e.code, raw


def find_unreleased_draft(token, repo):
    """Find the existing 'Unreleased' draft release, if any."""
    page = 1
    while True:
        status, releases = github_api("GET", f"releases?per_page=100&page={page}", token, repo)
        if status != 200:
            print(f"Failed to list releases: {status} {releases}", file=sys.stderr)
            sys.exit(1)
        for r in releases:
            if r.get("draft") and r.get("tag_name") == "unreleased":
                return r
        if len(releases) < 100:
            break
        page += 1
    return None


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN environment variable required", file=sys.stderr)
        sys.exit(1)

    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        print("GITHUB_REPOSITORY environment variable required", file=sys.stderr)
        sys.exit(1)

    with open("CHANGELOG.md", encoding="utf-8") as f:
        body = extract_unreleased(f.read())

    draft = find_unreleased_draft(token, repo)

    if body:
        if draft is None:
            status, resp = github_api(
                "POST",
                "releases",
                token,
                repo,
                {
                    "tag_name": "unreleased",
                    "name": "Unreleased",
                    "body": body,
                    "draft": True,
                },
            )
            if status != 201:
                print(f"Failed to create draft: {status} {resp}", file=sys.stderr)
                sys.exit(1)
            print("Created draft: Unreleased")

        elif (draft.get("body") or "").strip() != body:
            status, resp = github_api(
                "PATCH", f"releases/{draft['id']}", token, repo, {"body": body}
            )
            if status != 200:
                print(f"Failed to update draft: {status} {resp}", file=sys.stderr)
                sys.exit(1)
            print("Updated draft: Unreleased")

        else:
            print("Draft already up to date")

    else:
        print("No unreleased content, nothing to do")


if __name__ == "__main__":
    main()
