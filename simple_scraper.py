"""
Simple web scraper - No Docker, No Database Required
Works standalone for quick testing and development

Usage:
    python simple_scraper.py https://example.com
    python simple_scraper.py https://example.com --output results.json
"""

import asyncio
import argparse
import json
from datetime import datetime
from pathlib import Path
import hashlib

# Optional imports (will fallback if not available)
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("⚠️  BeautifulSoup not installed. Install with: pip install beautifulsoup4")

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    import requests
    print("⚠️  httpx not installed, using requests instead")

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("⚠️  sentence-transformers not installed (embeddings disabled)")


class SimpleScraper:
    """Standalone scraper with no external dependencies"""

    def __init__(self, output_dir: str = "./scraped_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize embedding model if available
        self.embedding_model = None
        if HAS_EMBEDDINGS:
            try:
                print("Loading embedding model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Embedding model loaded")
            except Exception as e:
                print(f"⚠️  Could not load embedding model: {e}")

    def generate_id(self, url: str) -> str:
        """Generate unique ID from URL"""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    async def fetch_url(self, url: str) -> tuple[str, int]:
        """
        Fetch URL content
        Returns: (html_content, status_code)
        """
        try:
            if HAS_HTTPX:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    response = await client.get(
                        url,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                        timeout=30.0
                    )
                    return response.text, response.status_code
            else:
                # Fallback to requests (blocking)
                response = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                    timeout=30
                )
                return response.text, response.status_code
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
            return "", 0

    def parse_content(self, html: str) -> dict:
        """Extract content from HTML"""
        if not HAS_BS4:
            return {
                "title": "N/A (BeautifulSoup not installed)",
                "content": html[:500],
                "error": "Install beautifulsoup4 for proper parsing"
            }

        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Get title
        title = soup.title.string if soup.title else "No title"

        # Get main content
        content = soup.get_text(separator='\n', strip=True)

        # Get links
        links = [a.get('href') for a in soup.find_all('a', href=True)]

        return {
            "title": title,
            "content": content,
            "links": links[:50],  # First 50 links
            "word_count": len(content.split())
        }

    def generate_embedding(self, text: str) -> list:
        """Generate embedding vector"""
        if not self.embedding_model:
            return []

        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"⚠️  Error generating embedding: {e}")
            return []

    async def scrape(self, url: str) -> dict:
        """
        Scrape a single URL and return results
        """
        print(f"🔍 Scraping: {url}")

        # Fetch HTML
        start_time = datetime.now()
        html, status_code = await self.fetch_url(url)
        fetch_time = (datetime.now() - start_time).total_seconds()

        if not html:
            return {
                "url": url,
                "status": "failed",
                "error": f"HTTP {status_code}",
                "timestamp": datetime.now().isoformat()
            }

        # Parse content
        parsed = self.parse_content(html)

        # Generate embedding (optional)
        embedding = []
        if self.embedding_model and parsed["content"]:
            print("🧠 Generating embedding...")
            embedding = self.generate_embedding(parsed["content"][:1000])

        # Build result
        result = {
            "id": self.generate_id(url),
            "url": url,
            "status": "success",
            "status_code": status_code,
            "title": parsed["title"],
            "content": parsed["content"][:2000],  # First 2000 chars
            "content_preview": parsed["content"][:200] + "...",
            "word_count": parsed.get("word_count", 0),
            "links_found": len(parsed.get("links", [])),
            "has_embedding": len(embedding) > 0,
            "embedding_dim": len(embedding) if embedding else 0,
            "fetch_time_seconds": fetch_time,
            "timestamp": datetime.now().isoformat(),
        }

        # Optionally include full data
        result["links"] = parsed.get("links", [])
        if embedding:
            result["embedding"] = embedding

        print(f"✅ Scraped successfully!")
        print(f"   Title: {result['title']}")
        print(f"   Words: {result['word_count']}")
        print(f"   Links: {result['links_found']}")
        print(f"   Time: {fetch_time:.2f}s")

        return result

    def save_result(self, result: dict, output_file: str = None):
        """Save result to JSON file"""
        if output_file is None:
            # Auto-generate filename
            safe_url = result['id']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"scraped_{safe_url}_{timestamp}.json"
        else:
            output_file = Path(output_file)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved to: {output_file}")
        return str(output_file)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Simple web scraper")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--no-save", action="store_true", help="Don't save to file")
    parser.add_argument("--print", "-p", action="store_true", help="Print result to console")

    args = parser.parse_args()

    # Create scraper
    scraper = SimpleScraper()

    # Scrape URL
    result = await scraper.scrape(args.url)

    # Save result
    if not args.no_save:
        scraper.save_result(result, args.output)

    # Print to console
    if args.print:
        print("\n" + "="*60)
        print("RESULT:")
        print("="*60)
        print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
