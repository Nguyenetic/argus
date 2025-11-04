"""
Base scraper class with common functionality
All scrapers inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
import random
import hashlib
from pathlib import Path


class BaseScraper(ABC):
    """Abstract base class for all scrapers"""

    def __init__(
        self,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        headless: bool = True,
        timeout: int = 30000,
        screenshots_dir: str = "./screenshots",
        **kwargs
    ):
        self.proxy = proxy
        self.user_agent = user_agent or self._get_random_user_agent()
        self.headless = headless
        self.timeout = timeout
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Stats tracking
        self.stats = {
            "pages_scraped": 0,
            "screenshots_taken": 0,
            "errors": 0,
            "bypasses_successful": 0,
            "total_time": 0.0
        }

    @abstractmethod
    async def scrape(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Scrape a URL and return structured data
        Must be implemented by subclasses
        """
        pass

    @abstractmethod
    async def screenshot(self, url: str, output_path: Optional[str] = None) -> str:
        """
        Take a screenshot of the page
        Must be implemented by subclasses
        """
        pass

    def _get_random_user_agent(self) -> str:
        """Get a random realistic user agent"""
        user_agents = [
            # Chrome on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            # Chrome on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Firefox on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            # Firefox on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            # Edge on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            # Safari on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        ]
        return random.choice(user_agents)

    async def _human_delay(self, min_seconds: float = 0.5, max_seconds: float = 2.0):
        """Simulate human-like delay"""
        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)

    def _generate_hash(self, content: str) -> str:
        """Generate SHA256 hash of content"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _generate_screenshot_path(self, url: str, suffix: str = "") -> str:
        """Generate unique screenshot path"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{url_hash}_{timestamp}{suffix}.png"
        return str(self.screenshots_dir / filename)

    def get_stats(self) -> Dict[str, Any]:
        """Get scraper statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset scraper statistics"""
        self.stats = {
            "pages_scraped": 0,
            "screenshots_taken": 0,
            "errors": 0,
            "bypasses_successful": 0,
            "total_time": 0.0
        }
