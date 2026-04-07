"""
scrape_tabs.py
--------------
Multithreaded two-pass GP5 tab scraper.

  Pass 1 (threaded): Fetch /en/tabs/{artist_slug} per artist in parallel,
                     collect all tab links from <ul class="tabs">
  Pass 2 (threaded): For each tab URL, fetch the tab page, find the download
                     link, download the file. Skips already-downloaded files.

Each thread has its own requests.Session. A threading.Semaphore caps
concurrent requests to avoid hammering the server.

Usage:
    python scrape_tabs.py --base-url "https://example.com" \\
                          --artists artists_rock_metal.txt \\
                          --out-dir ./assets/data/scraped_tabs \\
                          --workers 8 --delay 1.0
"""

import argparse
import os
import re
import signal
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup


# ── Helpers ───────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+",    "-", slug)
    return slug


def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (research scraper)"})
    return s


# Thread-local sessions — one per worker thread, avoids lock contention
_local = threading.local()

def get_session():
    if not hasattr(_local, "session"):
        _local.session = make_session()
    return _local.session


# ── Pass 1: collect tab links for one artist ──────────────────────────────────

def fetch_artist_tabs(base_url, artist_slug, sem, delay):
    """
    Fetch /en/tabs/{artist_slug}, return list of (artist_slug, tab_href) tuples.
    Semaphore limits concurrent requests globally.
    """
    url = f"{base_url}/en/tabs/{artist_slug}"
    with sem:
        try:
            r = get_session().get(url, timeout=15)
            time.sleep(delay)
            if r.status_code == 404:
                return []
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [WARN] {url}: {e}")
            return []

    soup     = BeautifulSoup(r.text, "html.parser")
    tab_list = soup.find("ul", class_="tabs")
    if not tab_list:
        return []

    links = []
    for li in tab_list.find_all("li"):
        a = li.find("a", href=True)
        if a and a["href"].startswith("/en/tabs/"):
            links.append((artist_slug, urllib.parse.urljoin(base_url, a["href"])))
    return links


# ── Pass 2: fetch one tab page, get download URL, save file ───────────────────

def fetch_and_download(base_url, artist_slug, tab_url, out_dir, sem, delay):
    """
    Fetch tab page → find ?download link → download file.
    Skips if output file already exists.
    Returns one of: 'downloaded', 'skipped', 'no_link', 'error'
    """
    tab_slug  = tab_url.rstrip("/").split("/")[-1]
    artist_dir = os.path.join(out_dir, artist_slug)
    out_path   = os.path.join(artist_dir, f"{tab_slug}.gp5")

    # Skip already-downloaded files without any network request
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return "skipped", tab_slug

    os.makedirs(artist_dir, exist_ok=True)

    # Fetch tab page
    with sem:
        try:
            r = get_session().get(tab_url, timeout=15)
            time.sleep(delay)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [WARN] {tab_url}: {e}")
            return "error", tab_slug

    soup   = BeautifulSoup(r.text, "html.parser")
    dl_tag = soup.find("a", class_="button", href=re.compile(r"\?download$"))
    if not dl_tag:
        return "no_link", tab_slug

    dl_url = urllib.parse.urljoin(base_url, dl_tag["href"])

    # Download file
    with sem:
        try:
            r = get_session().get(dl_url, timeout=30, stream=True)
            time.sleep(delay)
            r.raise_for_status()
            tmp = out_path + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            os.replace(tmp, out_path)   # atomic rename — no partial files
        except requests.RequestException as e:
            print(f"  [ERROR] {dl_url}: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            return "error", tab_slug

    return "downloaded", tab_slug


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url",    required=True)
    parser.add_argument("--artists",     default="artists_rock_metal.txt")
    parser.add_argument("--out-dir",     default="./assets/data/scraped_tabs")
    parser.add_argument("--workers",     type=int,   default=8,
                        help="Number of parallel download threads (default 8)")
    parser.add_argument("--delay",       type=float, default=1.0,
                        help="Per-thread sleep after each request (default 1.0s)")
    parser.add_argument("--max-artists", type=int,   default=None,
                        help="Stop after N artists (for testing)")
    parser.add_argument("--max-files",   type=int,   default=None,
                        help="Stop after N files downloaded (press Ctrl+C anytime too)")
    args = parser.parse_args()

    with open(args.artists, encoding="utf-8") as f:
        artists = [l.strip() for l in f if l.strip()]
    if args.max_artists:
        artists = artists[:args.max_artists]
    print(f"Loaded {len(artists)} artists  |  workers={args.workers}  "
          f"delay={args.delay}s  max_files={args.max_files or 'unlimited'}")

    os.makedirs(args.out_dir, exist_ok=True)

    # stop_event: set by Ctrl+C or when max_files is reached
    # All threads check it before making new requests.
    stop_event = threading.Event()

    def _handle_sigint(sig, frame):
        if not stop_event.is_set():
            print("\n[!] Ctrl+C received — finishing in-flight requests then stopping...")
            stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    sem    = threading.Semaphore(args.workers)
    counts = {"downloaded": 0, "skipped": 0, "no_link": 0, "error": 0}
    lock   = threading.Lock()

    # ── Pass 1: collect all tab links ────────────────────────────────────────
    print(f"\nPass 1: collecting tab links...")
    all_tabs = []
    no_tabs  = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_artist_tabs,
                        args.base_url, slugify(a), sem, args.delay): a
            for a in artists
        }
        for i, fut in enumerate(as_completed(futures), 1):
            if stop_event.is_set():
                break
            artist = futures[fut]
            links  = fut.result()
            if links:
                all_tabs.extend(links)
                print(f"  [{i}/{len(artists)}] {artist}: {len(links)} tab(s)")
            else:
                no_tabs += 1

    if stop_event.is_set():
        print("Stopped during Pass 1.")
        return

    print(f"\nPass 1 done: {len(all_tabs)} tabs across {len(artists)-no_tabs} artists "
          f"({no_tabs} artists had no tabs).")

    # ── Pass 2: download files ────────────────────────────────────────────────
    print(f"\nPass 2: downloading {len(all_tabs)} files...")

    def _download_task(artist_slug, tab_url):
        """Wrapper that checks stop_event before doing any work."""
        if stop_event.is_set():
            return "stopped", tab_url.split("/")[-1]
        status, slug = fetch_and_download(
            args.base_url, artist_slug, tab_url,
            args.out_dir, sem, args.delay)
        # If max_files reached after this download, signal stop
        if status == "downloaded" and args.max_files:
            with lock:
                if counts["downloaded"] + 1 >= args.max_files:
                    stop_event.set()
        return status, slug

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_task, slug, url): (slug, url)
            for slug, url in all_tabs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            status, tab_slug = fut.result()
            if status == "stopped":
                continue
            with lock:
                counts[status] += 1
            if status == "downloaded":
                n = counts["downloaded"]
                limit_str = f"/{args.max_files}" if args.max_files else ""
                print(f"  [{i}/{len(all_tabs)}] [OK]   {tab_slug}  "
                      f"(total: {n}{limit_str})")
                if stop_event.is_set():
                    print(f"[!] Reached {n} downloaded files — stopping.")
                    break
            elif status == "error":
                print(f"  [{i}/{len(all_tabs)}] [ERR]  {tab_slug}")

    print(f"\n{'Stopped early' if stop_event.is_set() else 'Done'}.")
    print(f"  Downloaded : {counts['downloaded']}")
    print(f"  Skipped    : {counts['skipped']} (already on disk)")
    print(f"  No link    : {counts['no_link']}")
    print(f"  Errors     : {counts['error']}")


if __name__ == "__main__":
    main()
