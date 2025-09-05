"""
Lightweight news gate helper.

- By default uses NewsAPI (https://newsapi.org/) if NEWS_API_KEY provided.
- You can modify fetch_headlines() to use RSS feeds or other news sources.
- Sentiment is computed with a tiny lexicon-based approach (no heavy deps).
- Returns a sentiment score in [-1.0, +1.0], higher -> more positive.
"""

import os
import requests
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("news-gate")

# tiny lexicon
_POS_WORDS = {
    "gain", "up", "positive", "beat", "strong", "rise", "bull", "surge", "growth", "improve", "better",
}
_NEG_WORDS = {
    "drop", "down", "negative", "miss", "weak", "fall", "bear", "plunge", "decline", "risk", "worse", "crash",
}


class NewsGate:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("NEWS_API_KEY", "")
        self.cache = []
        self.last_fetch = datetime.min
        self.cache_ttl = timedelta(seconds=int(os.getenv("NEWS_CACHE_TTL", "60")))

    def fetch_headlines(self):
        """
        Returns list of headline strings.
        If NEWS_API_KEY set, uses newsapi.org top headlines. Otherwise returns empty list.
        You should replace or extend this method to use additional sources relevant to your markets.
        """
        headlines = []
        if not self.api_key:
            logger.debug("No NEWS_API_KEY provided; skipping external news fetch.")
            return headlines

        now = datetime.utcnow()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {"language": "en", "pageSize": 20}
            resp = requests.get(url, params=params, headers={"Authorization": self.api_key}, timeout=8)
            data = resp.json()
            for art in data.get("articles", []):
                title = art.get("title")
                desc = art.get("description")
                if title:
                    headlines.append(title)
                if desc:
                    headlines.append(desc)
            self.cache = headlines
            self.last_fetch = now
        except Exception as e:
            logger.warning("Failed to fetch headlines: %s", e)
        return headlines

    def score_text(self, text: str) -> float:
        """
        basic lexicon-based sentiment: (+1 for pos word, -1 for neg word) / sqrt(wordcount)
        """
        if not text:
            return 0.0
        txt = text.lower()
        tokens = [w.strip(".,!?:;()[]\"'") for w in txt.split()]
        score = 0
        for t in tokens:
            if t in _POS_WORDS:
                score += 1
            elif t in _NEG_WORDS:
                score -= 1
        denom = max(1.0, len(tokens) ** 0.5)
        return score / denom

    def aggregate_headlines_score(self):
        headlines = self.fetch_headlines()
        if not headlines:
            return 0.0
        scores = [self.score_text(h) for h in headlines]
        # average and clamp
        avg = sum(scores) / len(scores)
        return max(-1.0, min(1.0, avg))

    def get_sentiment(self) -> float:
        """
        public method to get current aggregated sentiment.
        """
        try:
            s = self.aggregate_headlines_score()
            logger.debug("Aggregated news sentiment: %.4f", s)
            return s
        except Exception as e:
            logger.exception("News scoring failed: %s", e)
            return 0.0
