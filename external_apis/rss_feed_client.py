"""
RSS Feed Client for aggregating movie articles from popular film blogs.
Fetches and parses RSS feeds from sources like IndieWire, Collider, etc.
"""

import feedparser
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Curated list of quality movie/film RSS feeds
MOVIE_BLOG_FEEDS = {
    "indiewire": {
        "url": "https://www.indiewire.com/feed/",
        "name": "IndieWire",
        "category": "Industry News"
    },
    "collider": {
        "url": "https://collider.com/feed/",
        "name": "Collider",
        "category": "Entertainment"
    },
    "slashfilm": {
        "url": "https://www.slashfilm.com/feed/",
        "name": "SlashFilm",
        "category": "Film News"
    },
    "screenrant": {
        "url": "https://screenrant.com/feed/",
        "name": "Screen Rant",
        "category": "Movies & TV"
    },
    "variety_film": {
        "url": "https://variety.com/v/film/feed/",
        "name": "Variety - Film",
        "category": "Industry"
    },
    "hollywood_reporter": {
        "url": "https://www.hollywoodreporter.com/feed/",
        "name": "The Hollywood Reporter",
        "category": "Industry News"
    },
    "filmschoolrejects": {
        "url": "https://filmschoolrejects.com/feed/",
        "name": "Film School Rejects",
        "category": "Film Analysis"
    },
    "tasteofcinema": {
        "url": "https://www.tasteofcinema.com/feed/",
        "name": "Taste of Cinema",
        "category": "Film Lists & Essays"
    },
}


class RSSFeedClient:
    """Client for fetching and parsing movie blog RSS feeds"""
    
    def __init__(self):
        self.feeds = MOVIE_BLOG_FEEDS
        
    def fetch_feed(self, feed_key: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single RSS feed by key
        
        Args:
            feed_key: Key from MOVIE_BLOG_FEEDS
            
        Returns:
            Parsed feed data or None if error
        """
        if feed_key not in self.feeds:
            logger.error(f"Unknown feed key: {feed_key}")
            return None
            
        feed_info = self.feeds[feed_key]
        
        try:
            logger.info(f"Fetching RSS feed: {feed_info['name']}")
            feed = feedparser.parse(feed_info['url'])
            
            if feed.bozo:
                logger.warning(f"Feed parsing warning for {feed_info['name']}: {feed.bozo_exception}")
            
            return feed
            
        except Exception as e:
            logger.error(f"Error fetching feed {feed_info['name']}: {e}")
            return None
    
    def fetch_all_feeds(self) -> List[Dict[str, Any]]:
        """
        Fetch all configured RSS feeds
        
        Returns:
            List of parsed feed objects
        """
        feeds = []
        for feed_key in self.feeds.keys():
            feed = self.fetch_feed(feed_key)
            if feed:
                feeds.append({
                    "key": feed_key,
                    "feed": feed,
                    "info": self.feeds[feed_key]
                })
        return feeds
    
    def parse_article(self, entry: Any, source_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Parse a single feed entry into a standardized article format
        
        Args:
            entry: Feed entry object
            source_info: Source metadata (name, category, etc.)
            
        Returns:
            Standardized article dictionary
        """
        # Extract publish date
        pub_date = None
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                pub_date = datetime(*entry.published_parsed[:6]).isoformat()
            except:
                pass
        
        # Extract image/thumbnail
        image_url = None
        if hasattr(entry, 'media_content') and entry.media_content:
            image_url = entry.media_content[0].get('url')
        elif hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
            image_url = entry.media_thumbnail[0].get('url')
        elif hasattr(entry, 'enclosures') and entry.enclosures:
            for enclosure in entry.enclosures:
                if enclosure.get('type', '').startswith('image/'):
                    image_url = enclosure.get('href')
                    break
        
        # If no image found, try to extract from content/description HTML
        if not image_url:
            content = entry.get('content', [{}])[0].get('value', '') if hasattr(entry, 'content') else ''
            if not content and hasattr(entry, 'summary'):
                content = entry.summary
            elif not content and hasattr(entry, 'description'):
                content = entry.description
            
            if content:
                image_url = self._extract_first_image(content)
        
        # Clean summary/description
        summary = ""
        if hasattr(entry, 'summary'):
            summary = self._clean_html(entry.summary)
        elif hasattr(entry, 'description'):
            summary = self._clean_html(entry.description)
        
        # Extract categories/tags
        tags = []
        if hasattr(entry, 'tags'):
            tags = [tag.term for tag in entry.tags[:5]]  # Limit to 5 tags
        
        # Calculate read time (rough estimate)
        word_count = len(summary.split())
        read_time = max(1, word_count // 200)  # Assume 200 words per minute
        
        return {
            "id": entry.get('id', entry.get('link', '')),
            "title": entry.get('title', 'Untitled'),
            "summary": summary[:500] + "..." if len(summary) > 500 else summary,
            "content": summary,
            "url": entry.get('link', ''),
            "author": entry.get('author', source_info['name']),
            "publishDate": pub_date,
            "source": source_info['name'],
            "category": source_info['category'],
            "imageUrl": image_url,
            "tags": tags,
            "readTime": read_time
        }
    
    def get_articles(self, limit: int = 20, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get aggregated articles from all feeds or a specific source
        
        Args:
            limit: Maximum number of articles to return
            source: Optional source filter (feed key)
            
        Returns:
            List of parsed articles, sorted by date (newest first)
        """
        articles = []
        
        if source and source in self.feeds:
            # Fetch single source
            feed_data = self.fetch_feed(source)
            if feed_data:
                feed_list = [{"key": source, "feed": feed_data, "info": self.feeds[source]}]
            else:
                return []
        else:
            # Fetch all sources
            feed_list = self.fetch_all_feeds()
        
        # Parse articles from all feeds
        for feed_data in feed_list:
            feed = feed_data['feed']
            source_info = feed_data['info']
            
            for entry in feed.entries[:10]:  # Take up to 10 from each source
                try:
                    article = self.parse_article(entry, source_info)
                    articles.append(article)
                except Exception as e:
                    logger.error(f"Error parsing article: {e}")
                    continue
        
        # Sort by date (newest first)
        articles.sort(key=lambda x: x.get('publishDate', ''), reverse=True)
        
        return articles[:limit]
    
    def get_featured_articles(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get featured articles (most recent from top-tier sources)
        
        Args:
            count: Number of featured articles
            
        Returns:
            List of featured articles
        """
        # Priority sources for featured content
        priority_sources = ['variety_film', 'hollywood_reporter', 'indiewire']
        
        articles = []
        for source in priority_sources:
            feed_data = self.fetch_feed(source)
            if feed_data:
                source_info = self.feeds[source]
                for entry in feed_data.entries[:2]:  # Take top 2 from each
                    try:
                        article = self.parse_article(entry, source_info)
                        articles.append(article)
                    except Exception as e:
                        logger.error(f"Error parsing featured article: {e}")
                        continue
        
        # Sort and return
        articles.sort(key=lambda x: x.get('publishDate', ''), reverse=True)
        return articles[:count]
    
    def search_articles(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search articles by keyword in title or summary
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            Matching articles
        """
        all_articles = self.get_articles(limit=100)  # Search in recent 100
        query_lower = query.lower()
        
        matches = []
        for article in all_articles:
            title = article.get('title', '').lower()
            summary = article.get('summary', '').lower()
            
            if query_lower in title or query_lower in summary:
                matches.append(article)
        
        return matches[:limit]
    
    def get_sources(self) -> List[Dict[str, str]]:
        """
        Get list of available article sources
        
        Returns:
            List of source information
        """
        return [
            {
                "key": key,
                "name": info['name'],
                "category": info['category'],
                "url": info['url']
            }
            for key, info in self.feeds.items()
        ]
    
    @staticmethod
    def _extract_first_image(html_text: str) -> str:
        """Extract the first image URL from HTML content"""
        if not html_text:
            return None
        
        # Look for img tags with src attribute
        img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html_text, re.IGNORECASE)
        if img_match:
            img_url = img_match.group(1)
            # Filter out common placeholder/tracking pixels
            if img_url and not any(x in img_url.lower() for x in ['1x1', 'pixel', 'tracker', 'blank.gif']):
                # Make sure it's a valid HTTP/HTTPS URL
                if img_url.startswith('http'):
                    return img_url
        return None
    
    @staticmethod
    def _clean_html(html_text: str) -> str:
        """Remove HTML tags and clean text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_text)
        # Decode HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# Singleton instance
_rss_client = None

def get_rss_client() -> RSSFeedClient:
    """Get singleton RSS feed client"""
    global _rss_client
    if _rss_client is None:
        _rss_client = RSSFeedClient()
    return _rss_client
