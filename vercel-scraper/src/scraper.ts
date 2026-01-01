import { chromium, type Browser, type Page } from 'playwright';

export interface Review {
  author: string;
  rating: number;
  date: string;
  text: string;
}

export interface BusinessInfo {
  name: string;
  rating: number;
  total_reviews: number;
  address: string;
  phone: string;
}

export interface ScrapeResult {
  business: BusinessInfo;
  reviews: Review[];
  count: number;
  scraped_at: string;
}

export interface ScrapeOptions {
  maxReviews?: number;
  headless?: boolean;
}

/**
 * Convert a Google Maps short URL to a share.google URL for scraping
 * Google Search results are less protected than direct Maps pages
 */
function getSearchUrl(mapsUrl: string): string {
  // If already a share.google URL, use as-is
  if (mapsUrl.includes('share.google')) {
    return mapsUrl;
  }
  // For maps.app.goo.gl URLs, use the share.google redirect
  // These redirect to Google Search which has accessible review modals
  return mapsUrl;
}

/**
 * Extract reviews from the Google Maps review modal
 */
async function extractReviews(page: Page): Promise<{ business: BusinessInfo; reviews: Review[] }> {
  return await page.evaluate(() => {
    const reviews: Review[] = [];
    const reviewContainers = document.querySelectorAll('.bwb7ce');

    reviewContainers.forEach(container => {
      // Extract author name
      const authorLink = container.querySelector('a[href*="/maps/contrib/"]');
      const rawAuthor = authorLink?.textContent?.trim() || '';
      const authorMatch = rawAuthor.match(/^([A-Za-z\s.'-]+?)(?:Local Guide|[\d]|·|$)/);
      const author = authorMatch ? authorMatch[1].trim() : rawAuthor.split(/Local Guide|\d/)[0].trim();

      // Extract rating
      const ratingEl = container.querySelector('[aria-label*="star"], [aria-label*="Rated"]');
      let rating = 0;
      if (ratingEl) {
        const label = ratingEl.getAttribute('aria-label') || '';
        const match = label.match(/(\d)/);
        if (match) rating = parseInt(match[1]);
      }

      // Extract date
      const fullText = container.textContent || '';
      const dateMatch = fullText.match(/(\d+\s+(?:days?|weeks?|months?|years?)\s+ago|a\s+(?:day|week|month|year)\s+ago)/i);
      const date = dateMatch ? dateMatch[0] : '';

      // Extract review text (clean JS garbage)
      const textMatch = fullText.match(/Report review.*?ago(?:New)?(.+?)(?:Fast-Fix Jewelry.*?\(Owner\)|Hover to react|More_|\(function\(\)|Like\s*Share|$)/s);
      let text = textMatch ? textMatch[1].trim() : '';
      text = text.replace(/…\s*More\s*$/, '…').trim();
      text = text.replace(/\(function\(\)[\s\S]*$/, '').trim();
      text = text.replace(/_❤️\d+$/, '').trim();

      if (author) {
        reviews.push({ author, rating, date, text });
      }
    });

    // Extract business info from the page
    // Try multiple selectors for business name
    let businessName = 'Unknown Business';
    const titleSelectors = [
      '[data-attrid="title"]',
      '.SPZz6b span',
      '.qrShPb span',
      '[data-item-id] .fontHeadlineLarge',
      '.DUwDvf',
    ];
    for (const sel of titleSelectors) {
      const el = document.querySelector(sel);
      if (el?.textContent && !el.textContent.includes('Accessibility')) {
        businessName = el.textContent.trim();
        break;
      }
    }
    // Fallback: look for the business name in the review dialog header
    if (businessName === 'Unknown Business') {
      const dialogHeader = document.querySelector('.review-dialog-body, .Yr7JMd, .P5Bobd');
      if (dialogHeader) {
        const headerText = dialogHeader.textContent || '';
        const nameMatch = headerText.match(/^([A-Za-z\s\-&']+?)(?:\d|reviews?|Google)/i);
        if (nameMatch) businessName = nameMatch[1].trim();
      }
    }

    const ratingText = document.querySelector('[data-attrid="kc:/local:lu attribute list"] span, .Aq14fc')?.textContent || '';
    const ratingMatch = ratingText.match(/([\d.]+)/);
    const businessRating = ratingMatch ? parseFloat(ratingMatch[1]) : 0;

    const reviewCountText = document.querySelector('[data-attrid="kc:/local:lu attribute list"], .z5jxId')?.textContent || '';
    const countMatch = reviewCountText.match(/([\d,]+)\s*reviews?/i);
    const totalReviews = countMatch ? parseInt(countMatch[1].replace(',', '')) : reviews.length;

    return {
      business: {
        name: businessName,
        rating: businessRating,
        total_reviews: totalReviews,
        address: '',
        phone: ''
      },
      reviews
    };
  });
}

/**
 * Scroll the review modal to load more reviews
 */
async function scrollToLoadReviews(page: Page, maxReviews: number): Promise<number> {
  return await page.evaluate(async (max: number) => {
    const scrollable = document.querySelector('.RVCQse') as HTMLElement;
    if (!scrollable) return document.querySelectorAll('.bwb7ce').length;

    let prevCount = 0;
    let currentCount = document.querySelectorAll('.bwb7ce').length;

    while (currentCount > prevCount && currentCount < max) {
      prevCount = currentCount;
      scrollable.scrollTop = scrollable.scrollHeight;
      await new Promise(r => setTimeout(r, 800));
      currentCount = document.querySelectorAll('.bwb7ce').length;
    }

    return currentCount;
  }, maxReviews);
}

/**
 * Main scraper function
 */
export async function scrapeGoogleMapsReviews(
  url: string,
  options: ScrapeOptions = {}
): Promise<ScrapeResult> {
  const { maxReviews = 100, headless = true } = options;

  // Launch browser with stealth settings
  const browser: Browser = await chromium.launch({
    headless,
    args: [
      '--disable-blink-features=AutomationControlled',
      '--disable-features=IsolateOrigins,site-per-process',
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--no-first-run',
      '--no-zygote',
      '--disable-gpu'
    ]
  });

  try {
    const context = await browser.newContext({
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      viewport: { width: 1920, height: 1080 },
      locale: 'en-US',
    });

    // Add stealth scripts
    await context.addInitScript(() => {
      Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
      Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
      Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    });

    const page = await context.newPage();

    // Navigate to the URL
    console.log(`Navigating to: ${url}`);
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });

    // Wait for page to fully load
    await page.waitForTimeout(5000);

    // Try multiple selectors for the reviews button/link
    const reviewSelectors = [
      'a[href*="reviews"]',
      'button[aria-label*="Reviews"]',
      '[data-tab="reviews"]',
      'a:has-text("reviews")',
      '[data-async-trigger="reviewDialog"]',
      '.hqzQac', // Google search reviews link
    ];

    let clicked = false;
    for (const selector of reviewSelectors) {
      try {
        const btn = await page.$(selector);
        if (btn) {
          console.log(`Found reviews element with selector: ${selector}`);
          await btn.click();
          clicked = true;
          break;
        }
      } catch (e) {
        // Continue to next selector
      }
    }

    if (clicked) {
      console.log('Clicked reviews, waiting for dialog...');
      await page.waitForTimeout(3000);
    }

    // Wait for review content to load with multiple possible selectors
    const reviewContainerSelectors = ['.bwb7ce', '.jftiEf', '[data-review-id]', '.gws-localreviews__google-review'];
    let foundReviews = false;

    for (const selector of reviewContainerSelectors) {
      try {
        await page.waitForSelector(selector, { timeout: 10000 });
        console.log(`Found reviews with selector: ${selector}`);
        foundReviews = true;
        break;
      } catch (e) {
        // Try next selector
      }
    }

    if (!foundReviews) {
      console.log('Review containers not found, taking screenshot for debugging...');
      // Continue anyway, might still find some content
    }

    // Scroll to load more reviews
    console.log(`Scrolling to load up to ${maxReviews} reviews...`);
    const loadedCount = await scrollToLoadReviews(page, maxReviews);
    console.log(`Loaded ${loadedCount} reviews`);

    // Extract all reviews
    const { business, reviews } = await extractReviews(page);

    await browser.close();

    return {
      business,
      reviews: reviews.slice(0, maxReviews),
      count: Math.min(reviews.length, maxReviews),
      scraped_at: new Date().toISOString()
    };

  } catch (error) {
    await browser.close();
    throw error;
  }
}

// CLI runner
if (process.argv[1]?.includes('scraper')) {
  const url = process.argv[2] || 'https://maps.app.goo.gl/dqphbjQEoKY3TuSUA';
  const maxReviews = parseInt(process.argv[3] || '100');

  console.log(`Scraping reviews from: ${url}`);
  console.log(`Max reviews: ${maxReviews}`);

  scrapeGoogleMapsReviews(url, { maxReviews, headless: false })
    .then(result => {
      console.log(`\nScraped ${result.count} reviews for ${result.business.name}`);
      console.log(JSON.stringify(result, null, 2));
    })
    .catch(err => {
      console.error('Scraping failed:', err);
      process.exit(1);
    });
}
