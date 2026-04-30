#!/usr/bin/env python3
"""Sync CHANGELOG.md entries to GitHub Release drafts.

Parses CHANGELOG.md, detects which sections changed since the last push,
and creates/updates GitHub Release drafts only for those sections.
Uses --full for complete reconciliation (workflow_dispatch / bootstrapping).
"""

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

VERSION_HEADER = re.compile(r"^## \[(\d+\.\d+\.\d+)\]")
UNRELEASED_HEADER = re.compile(r"^## \[Unreleased\]")
LINK_REFERENCE = re.compile(r"^\[.+\]:\s*https?://")


def parse_changelog(text):
    """Parse Keep-a-Changelog text into {version_string: body} blocks."""
    blocks = {}
    current_version = None
    current_lines = []

    for line in text.splitlines():
        version_match = VERSION_HEADER.match(line)
        unreleased_match = UNRELEASED_HEADER.match(line)

        if version_match or unreleased_match:
            if current_version is not None:
                blocks[current_version] = "\n".join(current_lines).strip()
            current_version = version_match.group(1) if version_match else "unreleased"
            current_lines = []
        elif current_version is not None:
            if LINK_REFERENCE.match(line):
                continue
            current_lines.append(line)

    if current_version is not None:
        blocks[current_version] = "\n".join(current_lines).strip()

    return blocks


def get_old_changelog():
    """Retrieve CHANGELOG.md content from before the current push."""
    before_sha = os.environ.get("BEFORE_SHA", "")
    ref = (
        f"{before_sha}:CHANGELOG.md"
        if before_sha and before_sha != "0" * 40
        else "HEAD~1:CHANGELOG.md"
    )
    result = subprocess.run(["git", "show", ref], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout


def find_changed_versions(old_blocks, new_blocks):
    """Return {version: body} for blocks that are new or whose body changed."""
    changed = {}
    for version, body in new_blocks.items():
        if version not in old_blocks or old_blocks[version] != body:
            changed[version] = body
    # Detect removal of the [Unreleased] header so sync_unreleased
    # can clean up an orphaned draft on GitHub.
    if "unreleased" in old_blocks and "unreleased" not in new_blocks:
        changed["unreleased"] = ""
    return changed


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


def fetch_all_releases(token, repo):
    """Fetch all releases (including drafts) into a {tag_name: release} dict.

    Uses setdefault so the first occurrence (newest) wins if there are
    duplicates with the same tag_name.
    """
    by_tag = {}
    page = 1
    while True:
        status, releases = github_api("GET", f"releases?per_page=100&page={page}", token, repo)
        if status != 200:
            print(f"Failed to list releases: {status} {releases}", file=sys.stderr)
            sys.exit(1)
        for r in releases:
            by_tag.setdefault(r["tag_name"], r)
        if len(releases) < 100:
            break
        page += 1
    return by_tag


def create_draft(token, repo, tag_name, name, body):
    status, resp = github_api(
        "POST",
        "releases",
        token,
        repo,
        {
            "tag_name": tag_name,
            "name": name,
            "body": body,
            "draft": True,
        },
    )
    if status != 201:
        print(f"Failed to create draft '{name}': {status} {resp}", file=sys.stderr)
        sys.exit(1)
    print(f"  Created draft: {name}")


def update_release_body(token, repo, release_id, name, body):
    status, resp = github_api("PATCH", f"releases/{release_id}", token, repo, {"body": body})
    if status != 200:
        print(
            f"Failed to update release '{name}': {status} {resp}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Updated: {name}")


def delete_release(token, repo, release_id, name):
    status, _ = github_api("DELETE", f"releases/{release_id}", token, repo)
    if status != 204:
        print(f"Failed to delete release '{name}': {status}", file=sys.stderr)
        sys.exit(1)
    print(f"  Deleted: {name}")


def sync_versioned(releases_by_tag, token, repo, version, body):
    tag = f"v{version}"
    release = releases_by_tag.get(tag)

    if release is None:
        create_draft(token, repo, tag, tag, body)
        return

    if (release.get("body") or "").strip() == body:
        print(f"  {tag} already up to date")
        return

    update_release_body(token, repo, release["id"], tag, body)


def sync_unreleased(releases_by_tag, token, repo, body):
    release = releases_by_tag.get("unreleased")

    if release and not release.get("draft"):
        print(
            "WARNING: A published release with tag 'unreleased' exists. "
            "This was likely published by accident. Skipping unreleased sync.",
            file=sys.stderr,
        )
        return

    if body:
        if release is None:
            create_draft(token, repo, "unreleased", "Unreleased", body)
        elif (release.get("body") or "").strip() != body:
            update_release_body(token, repo, release["id"], "Unreleased", body)
        else:
            print("  Unreleased draft already up to date")
    elif release is not None:
        delete_release(token, repo, release["id"], "Unreleased")
    else:
        print("  No unreleased content, nothing to do")


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN environment variable required", file=sys.stderr)
        sys.exit(1)

    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        print("GITHUB_REPOSITORY environment variable required", file=sys.stderr)
        sys.exit(1)

    full = "--full" in sys.argv

    with open("CHANGELOG.md", encoding="utf-8") as f:
        new_blocks = parse_changelog(f.read())

    if full:
        changed = new_blocks
        print(f"Full reconciliation: {len(changed)} version(s)")
    else:
        old_content = get_old_changelog()
        if old_content is None:
            changed = new_blocks
            print("No previous CHANGELOG.md found, full reconciliation")
        else:
            old_blocks = parse_changelog(old_content)
            changed = find_changed_versions(old_blocks, new_blocks)
            print(f"Changed version(s): {len(changed)}")

    if not changed:
        print("Nothing to sync")
        return

    releases_by_tag = fetch_all_releases(token, repo)
    print(f"Existing releases on GitHub: {len(releases_by_tag)}")

    for version, body in changed.items():
        if version == "unreleased":
            sync_unreleased(releases_by_tag, token, repo, body)
        else:
            sync_versioned(releases_by_tag, token, repo, version, body)

    print("Done")


if __name__ == "__main__":
    main()
