"""Probe Lever boards with different URL patterns."""
import httpx
import asyncio

LEVER_SLUGS = [
    "spotify", "stripe", "databricks", "cloudflare",
    "reddit", "ramp", "deel", "samsara", "relativity",
]

async def main():
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        # Pattern 1: api.lever.co/v0/postings/{slug}?mode=json
        print("=== Lever v0 API ===")
        for slug in LEVER_SLUGS:
            try:
                r = await client.get(
                    f"https://api.lever.co/v0/postings/{slug}",
                    params={"mode": "json", "limit": 1},
                )
                if r.status_code == 200:
                    data = r.json()
                    count = len(data) if isinstance(data, list) else "?"
                    print(f"  {slug}: {r.status_code} -> {count} jobs")
                else:
                    print(f"  {slug}: {r.status_code}")
            except Exception as e:
                print(f"  {slug}: ERROR {type(e).__name__}: {e}")

asyncio.run(main())
