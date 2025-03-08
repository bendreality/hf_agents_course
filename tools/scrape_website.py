import asyncio
import json
from pathlib import Path
from typing import List
from datetime import datetime
import requests
import time
import random
from xml.etree import ElementTree
import re
from smolagents import tool

# Import your crawler components
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


def get_site_urls(input_sitemap_url):
    """
    Fetches all URLs from a sitemap.
    Uses more robust request headers to avoid 403 Forbidden errors.

    Returns:
        List[str]: List of URLs
    """
    sitemap_url = input_sitemap_url

    # More realistic browser headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': '/'.join(sitemap_url.split('/')[:3]),  # Base domain as referer
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }

    try:
        session = requests.Session()

        # First visit the main site to get cookies
        base_url = '/'.join(sitemap_url.split('/')[:3])
        session.get(base_url, headers=headers, timeout=15)

        # Add a small delay to mimic human behavior
        time.sleep(random.uniform(1, 2))

        # Then request the sitemap
        response = session.get(sitemap_url, headers=headers, timeout=15)
        response.raise_for_status()

        print(f"Request status code: {response.status_code}")

        # Check if it's an XML file
        if 'xml' in response.headers.get('Content-Type', '').lower() or response.text.strip().startswith('<?xml'):
            # Parse the XML
            try:
                root = ElementTree.fromstring(response.content)

                # Try with namespace
                namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]

                # If no URLs found with namespace, try without namespace
                if not urls:
                    print("Trying without namespace...")
                    urls = [loc.text for loc in root.findall('.//loc')]

                # Check if this is a sitemap index
                if not urls and ("<sitemapindex" in response.text or "sitemap" in response.text.lower()):
                    print("This appears to be a sitemap index. Fetching linked sitemaps...")
                    sitemap_urls = [loc.text for loc in root.findall('.//ns:sitemap/ns:loc', namespace)]

                    if not sitemap_urls:
                        sitemap_urls = [loc.text for loc in root.findall('.//sitemap/loc')]

                    all_urls = []
                    for sub_sitemap in sitemap_urls[:3]:  # Limit to first 3 sitemaps
                        print(f"Fetching sub-sitemap: {sub_sitemap}")
                        time.sleep(1)  # Be polite
                        sub_urls = get_site_urls(sub_sitemap)
                        all_urls.extend(sub_urls)

                    return all_urls
            except ElementTree.ParseError:
                print("XML parsing error. The response may not be a valid XML sitemap.")
                urls = []
        else:
            print("Response doesn't appear to be XML. Looking for URLs directly...")
            # If not XML, try to extract URLs directly using regex
            urls = re.findall(r'https?://[^\s<>"\']+', response.text)
            # Filter to only include URLs from the same domain
            domain = '/'.join(sitemap_url.split('/')[:3])
            urls = [url for url in urls if url.startswith(domain)]

        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []


def get_domain_name(url):
    """
    Extracts the domain name from a URL.

    Args:
        url: The URL to extract the domain from

    Returns:
        str: The domain name
    """
    # Extract the domain from the URL
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if match:
        return match.group(1)
    return "unknown-domain"


def create_safe_filename(url):
    """
    Creates a safe filename from a URL.

    Args:
        url: The URL to convert to a filename

    Returns:
        str: A safe filename
    """
    # Create safe filename from URL
    safe_filename = "".join(c if c.isalnum() else "_" for c in url.split("//")[-1])
    if len(safe_filename) > 100:
        safe_filename = safe_filename[:100]
    return safe_filename


def is_already_scraped(url, results_dir):
    """
    Checks if a URL has already been scraped.

    Args:
        url: The URL to check
        results_dir: The directory to check in

    Returns:
        bool: True if the URL has already been scraped, False otherwise
    """
    safe_filename = create_safe_filename(url)

    # Check if the markdown file exists
    markdown_path = results_dir / f"{safe_filename}.md"
    if markdown_path.exists():
        print(f"Skipping already scraped URL: {url}")
        return True

    return False


async def crawl_sequential(urls: List[str], sitemap_url: str):
    """
    Sequentially crawls a list of URLs and saves results.

    Args:
        urls: List of URLs to crawl
        sitemap_url: Original sitemap URL (used to determine the domain folder)
    """
    if not urls:
        print("No URLs to crawl")
        return "No URLs to crawl, Skip this page and continue with the next."

    # Get domain name for the folder
    domain = get_domain_name(sitemap_url)

    # Create results directory based on domain
    results_dir = Path("../DATA/crawl_results") / domain
    results_dir.mkdir(parents=True, exist_ok=True)

    browser_config = BrowserConfig(
        headless=True,
        extra_args=[
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process"
        ],
    )

    crawl_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator()
    )

    # Create the crawler (opens the browser)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        session_id = "session1"
        successful_crawls = 0
        skipped_urls = 0
        max_urls = 30
        urls_to_crawl = urls[:max_urls]

        # Filter out already scraped URLs
        filtered_urls = []
        for url in urls_to_crawl:
            if not is_already_scraped(url, results_dir):
                filtered_urls.append(url)
            else:
                skipped_urls += 1

        print(f"Skipped {skipped_urls} already scraped URLs")
        urls_to_crawl = filtered_urls

        for i, url in enumerate(urls_to_crawl):
            # Add a random delay between requests
            if i > 0:
                delay = random.uniform(2, 4)
                print(f"Waiting {delay:.1f} seconds before next request...")
                await asyncio.sleep(delay)

            result = await crawler.arun(
                url=url,
                config=crawl_config,
                session_id=session_id
            )

            # Create safe filename from URL
            safe_filename = create_safe_filename(url)

            if result.success:
                print(f"Successfully crawled: {url}")
                successful_crawls += 1

                # Save markdown
                markdown_path = results_dir / f"{safe_filename}.md"
                markdown_path.write_text(result.markdown.raw_markdown, encoding="utf-8")

                # Save metadata
                metadata = {
                    "url": url,
                    "crawl_time": datetime.now().isoformat(),
                    "success": True,
                    "markdown_length": len(result.markdown.raw_markdown)
                }

                metadata_path = results_dir / f"{safe_filename}_meta.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

                print(f"Saved results to: {markdown_path}")
            else:
                print(f"Failed: {url} - Error: {result.error_message}")

                # Save error information
                error_metadata = {
                    "url": url,
                    "crawl_time": datetime.now().isoformat(),
                    "success": False,
                    "error_message": result.error_message
                }

                error_path = results_dir / f"{safe_filename}_error.json"
                with open(error_path, 'w', encoding='utf-8') as f:
                    json.dump(error_metadata, f, indent=2)
    finally:
        # Clean up
        await crawler.close()

        # Save summary
        summary = {
            "crawl_time": datetime.now().isoformat(),
            "total_urls_attempted": len(urls_to_crawl),
            "successful_crawls": successful_crawls,
            "skipped_urls": skipped_urls,
            "results_directory": str(results_dir)
        }

        summary_path = results_dir / "crawl_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"\nCrawling complete. Results saved in: {results_dir}")

    if len(urls_to_crawl) != skipped_urls:
        return f"Crawling complete. Processed {len(urls_to_crawl)} URLs with {successful_crawls} successful crawls. Skipped {skipped_urls} previously scraped URLs."
    else:
        return "No sites were scraped since everything is up to date"


@tool
def scrape_website_using_sitemap_url(sitemap_url: str) -> str:
    """
    A tool that scrapes websites by using the web address to the sitemap.xml

    This function is designed to be used by an AI agent. It handles the asyncio
    event loop internally so the agent doesn't need to worry about async/await.

    Args:
        sitemap_url: the full web address to the sitemap.xml

    Returns:
        str: A message indicating the result of the scraping operation
    """
    # If sitemap_url doesn't end with .xml, we'll append sitemap.xml
    if not sitemap_url.endswith('.xml'):
        if not sitemap_url.endswith('/'):
            sitemap_url += '/'
        sitemap_url += 'sitemap.xml'
        print(f"Adjusted URL to: {sitemap_url}")

    # Get URLs from the sitemap
    print(f"Fetching URLs from sitemap: {sitemap_url}")
    urls = get_site_urls(sitemap_url)

    if not urls:
        print("No URLs found to crawl")
        return "No URLs found to crawl. The site may be blocking access or using a non-standard format."

    print(f"Found {len(urls)} URLs to crawl")

    for url in urls[:11]:
        print(url)

    # Run the crawler in an asyncio event loop
    result = asyncio.run(crawl_sequential(urls[:11], sitemap_url))

    # Return a descriptive message
    return f"Scraping complete: the 11 latest URLs and processed them.\n{result}"


if __name__ == "__main__":
    # Example usage
    test_page = "https://towardsdatascience.com/post-sitemap.xml"
    result = scrape_website_using_sitemap_url(test_page)
    print(result)