from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup


CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_FILE = Path("data/assessments_scraped.json")


def scrape_catalog() -> List[Dict[str, Any]]:
    response = requests.get(CATALOG_URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    cards = soup.select("a[href*='/product-catalog/view/']")
    results: List[Dict[str, Any]] = []
    seen = set()

    for card in cards:
        href = card.get("href")
        if not href:
            continue
        url = href if href.startswith("http") else f"https://www.shl.com{href}"
        if url in seen:
            continue
        seen.add(url)
        name = card.get_text(" ", strip=True)
        if not name:
            continue
        results.append(
            {
                "name": name,
                "url": url,
                "category": "Unknown",
                "test_type": "U",
                "skills": [],
                "duration_minutes": None,
                "description": "",
            }
        )
    return results


def main() -> None:
    data = scrape_catalog()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} assessments to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
