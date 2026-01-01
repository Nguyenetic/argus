import type { VercelRequest, VercelResponse } from '@vercel/node';
import { chromium } from 'playwright-core';

interface Review {
  author: string;
  rating: number;
  date: string;
  text: string;
}

interface BusinessInfo {
  name: string;
  rating: number;
  total_reviews: number;
  address: string;
  phone: string;
}

interface ScrapeResult {
  business: BusinessInfo;
  reviews: Review[];
  count: number;
  scraped_at: string;
}

// Vercel serverless function for cron job
export default async function handler(req: VercelRequest, res: VercelResponse) {
  const authHeader = req.headers.authorization;
  if (process.env.CRON_SECRET && authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const url = (req.query.url as string) || process.env.GOOGLE_MAPS_URL;
  const maxReviews = parseInt((req.query.max as string) || '100');

  if (!url) {
    return res.status(400).json({ error: 'Missing url parameter or GOOGLE_MAPS_URL env var' });
  }

  try {
    console.log(`Starting scrape for: ${url}`);

    const browserWSEndpoint = process.env.BROWSERLESS_URL;

    let browser;
    if (browserWSEndpoint) {
      browser = await chromium.connect(browserWSEndpoint);
    } else {
      browser = await chromium.launch({
        headless: true,
        args: [
          '--disable-blink-features=AutomationControlled',
          '--no-sandbox',
          '--disable-setuid-sandbox',
          '--disable-dev-shm-usage',
        ]
      });
    }

    const context = await browser.newContext({
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      viewport: { width: 1920, height: 1080 },
      locale: 'en-US',
    });

    await context.addInitScript(() => {
      Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
      Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
    });

    const page = await context.newPage();
    await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(3000);

    const reviewsButton = await page.$('button[aria-label*="Reviews"], [data-tab="reviews"]');
    if (reviewsButton) {
      await reviewsButton.click();
      await page.waitForTimeout(2000);
    }

    await page.waitForSelector('.bwb7ce', { timeout: 30000 }).catch(() => {});

    await page.evaluate(async (max: number) => {
      const scrollable = document.querySelector('.RVCQse') as HTMLElement;
      if (!scrollable) return;
      let prevCount = 0;
      let currentCount = document.querySelectorAll('.bwb7ce').length;
      while (currentCount > prevCount && currentCount < max) {
        prevCount = currentCount;
        scrollable.scrollTop = scrollable.scrollHeight;
        await new Promise(r => setTimeout(r, 800));
        currentCount = document.querySelectorAll('.bwb7ce').length;
      }
    }, maxReviews);

    const result = await page.evaluate(() => {
      const reviews: Review[] = [];
      const reviewContainers = document.querySelectorAll('.bwb7ce');

      reviewContainers.forEach(container => {
        const authorLink = container.querySelector('a[href*="/maps/contrib/"]');
        const rawAuthor = authorLink?.textContent?.trim() || '';
        const authorMatch = rawAuthor.match(/^([A-Za-zs.'-]+?)(?:Local Guide|[d]|·|$)/);
        const author = authorMatch ? authorMatch[1].trim() : rawAuthor.split(/Local Guide|d/)[0].trim();

        const ratingEl = container.querySelector('[aria-label*="star"], [aria-label*="Rated"]');
        let rating = 0;
        if (ratingEl) {
          const label = ratingEl.getAttribute('aria-label') || '';
          const match = label.match(/(d)/);
          if (match) rating = parseInt(match[1]);
        }

        const fullText = container.textContent || '';
        const dateMatch = fullText.match(/(d+s+(?:days?|weeks?|months?|years?)s+ago|as+(?:day|week|month|year)s+ago)/i);
        const date = dateMatch ? dateMatch[0] : '';

        const textMatch = fullText.match(/Report review.*?ago(?:New)?(.+?)(?:Fast-Fix Jewelry.*?(Owner)|Hover to react|More_|(function()|Likes*Share|$)/s);
        let text = textMatch ? textMatch[1].trim() : '';
        text = text.replace(/…s*Mores*$/, '…').trim();
        text = text.replace(/(function()[sS]*$/, '').trim();
        text = text.replace(/_❤️d+$/, '').trim();

        if (author) {
          reviews.push({ author, rating, date, text });
        }
      });

      // Extract total review count from page
      const allText = document.body.innerText;
      const reviewCountMatch = allText.match(/([d,]+)s*reviews?/i);
      const totalReviews = reviewCountMatch ? parseInt(reviewCountMatch[1].replace(/,/g, '')) : reviews.length;

      const businessName = document.querySelector('h1')?.textContent?.trim() || 'Unknown Business';

      return {
        business: {
          name: businessName,
          rating: 0,
          total_reviews: totalReviews,
          address: '',
          phone: ''
        },
        reviews
      };
    });

    await browser.close();

    const scrapeResult: ScrapeResult = {
      ...result,
      count: result.reviews.length,
      scraped_at: new Date().toISOString()
    };

    if (process.env.SANITY_PROJECT_ID && process.env.SANITY_TOKEN) {
      await saveToSanity(scrapeResult);
    }

    return res.status(200).json(scrapeResult);

  } catch (error) {
    console.error('Scraping failed:', error);
    return res.status(500).json({
      error: 'Scraping failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

async function saveToSanity(result: ScrapeResult): Promise<void> {
  const projectId = process.env.SANITY_PROJECT_ID;
  const dataset = process.env.SANITY_DATASET || 'production';
  const token = process.env.SANITY_TOKEN;

  if (!projectId || !token) return;

  // Save reviews as testimonials (matching Fast-Fix schema)
  const testimonialMutations = result.reviews
    .filter(review => review.text && review.text.length > 10)
    .map(review => ({
      createOrReplace: {
        _type: 'testimonial',
        _id: `testimonial-${review.author.toLowerCase().replace(/[^a-z0-9]/g, '-')}-${review.date.toLowerCase().replace(/[^a-z0-9]/g, '-')}`,
        customerName: review.author,
        rating: review.rating || 5,
        review: review.text,
        date: review.date,
        featured: false,
        source: 'google'
      }
    }));

  // Update the review count in About Page stats
  const totalReviews = result.business.total_reviews || result.reviews.length;
  const statsUpdateMutation = {
    patch: {
      id: 'aboutPage',
      set: {
        'statsItems': [
          { _key: 'stat-experience', value: '40+', label: 'Years Experience', icon: 'Award' },
          { _key: 'stat-repairs', value: '50+', label: 'Repairs Completed', icon: 'Zap' },
          { _key: 'stat-rating', value: '47', label: 'Star Rating', icon: 'Star' },
          { _key: 'stat-customers', value: `${totalReviews}+`, label: 'Happy Customers', icon: 'Users' },
        ],
      }
    }
  };

  const allMutations = [...testimonialMutations, statsUpdateMutation];

  const response = await fetch(
    `https://${projectId}.api.sanity.io/v2024-01-01/data/mutate/${dataset}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify({ mutations: allMutations })
    }
  );

  if (!response.ok) {
    console.error('Failed to save to Sanity:', await response.text());
  } else {
    console.log(`Saved ${testimonialMutations.length} testimonials and updated stats (${totalReviews}+ reviews) to Sanity`);
  }
}
