# Google Maps Review Scraper for Vercel

A Playwright-based Google Maps review scraper designed to run as a Vercel cron job.

## Important: Browser Service Required

**Google actively blocks headless browsers.** For production use on Vercel, you MUST use a remote browser service:

### Recommended: Browserless.io
1. Sign up at https://browserless.io (free tier available)
2. Get your API token
3. Set environment variable: `BROWSERLESS_URL=wss://chrome.browserless.io?token=YOUR_TOKEN`

Browserless has built-in stealth features that bypass Google's bot detection.

### Alternative Services
- [Bright Data](https://brightdata.com/products/scraping-browser)
- [ScrapingBee](https://www.scrapingbee.com/)
- Self-hosted browserless Docker container

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env.local`:

```bash
cp .env.example .env.local
```

Required:
- `BROWSERLESS_URL` - Remote browser WebSocket URL (required for Vercel)
- `GOOGLE_MAPS_URL` - Default Google Maps URL to scrape

Optional (for Sanity CMS integration):
- `SANITY_PROJECT_ID`
- `SANITY_DATASET`
- `SANITY_TOKEN`
- `CRON_SECRET` - For authenticating cron requests

### 3. Deploy to Vercel

```bash
vercel
```

Add environment variables in Vercel dashboard:
1. Go to Project Settings → Environment Variables
2. Add `BROWSERLESS_URL` with your browserless token
3. Add `GOOGLE_MAPS_URL` with your target Maps URL
4. Add other optional variables as needed

### 4. Cron Configuration

The scraper runs daily at 9 AM UTC by default (configured in `vercel.json`):

```json
{
  "crons": [
    {
      "path": "/api/scrape",
      "schedule": "0 9 * * *"
    }
  ]
}
```

## Local Development

For local testing, you can run with a visible browser:

```bash
# With browserless (recommended)
BROWSERLESS_URL=wss://chrome.browserless.io?token=YOUR_TOKEN npx ts-node --esm src/scraper.ts

# Without browserless (may get blocked)
npx ts-node --esm src/scraper.ts "https://maps.app.goo.gl/YOUR_URL" 50
```

## API Usage

### GET /api/scrape

Query parameters:
- `url` - Google Maps URL (optional if `GOOGLE_MAPS_URL` is set)
- `max` - Maximum reviews to scrape (default: 100)

Headers:
- `Authorization: Bearer <CRON_SECRET>` (required if `CRON_SECRET` is set)

Response:
```json
{
  "business": {
    "name": "Business Name",
    "rating": 4.7,
    "total_reviews": 486
  },
  "reviews": [
    {
      "author": "John Doe",
      "rating": 5,
      "date": "2 weeks ago",
      "text": "Great service..."
    }
  ],
  "count": 100,
  "scraped_at": "2024-01-01T00:00:00.000Z"
}
```

## Why Remote Browser is Required

Google's bot detection blocks:
- ✗ Headless Chrome (even with stealth flags)
- ✗ chromiumoxide (Rust)
- ✗ Puppeteer headless
- ✗ Playwright headless

Google allows:
- ✓ Browserless.io (stealth mode)
- ✓ Real browser with user profile
- ✓ Playwright MCP (uses real browser)

## Sanity CMS Integration

If configured, reviews are automatically saved to Sanity. Create this schema:

```typescript
// schemas/review.ts
export default {
  name: 'review',
  type: 'document',
  title: 'Review',
  fields: [
    { name: 'author', type: 'string', title: 'Author' },
    { name: 'rating', type: 'number', title: 'Rating' },
    { name: 'date', type: 'string', title: 'Date' },
    { name: 'text', type: 'text', title: 'Review Text' },
    { name: 'business', type: 'string', title: 'Business' },
    { name: 'scrapedAt', type: 'datetime', title: 'Scraped At' }
  ]
}
```

## Cost Estimate

- **Browserless.io**: Free tier = 1000 sessions/month
- **Vercel Pro**: Cron jobs included, 60s function timeout
- **Sanity**: Free tier = 100K API requests/month

For daily scraping of 100 reviews, free tiers should be sufficient.
