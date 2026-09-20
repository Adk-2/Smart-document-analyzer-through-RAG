import requests
from bs4 import BeautifulSoup
from readability import Document


class WebLoader:
    @staticmethod
    def load_url(url: str):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        with requests.Session() as session:
            try:
                response = session.get(
                    url,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                html = response.text
            except requests.exceptions.RequestException as e:
                raise Exception(f"Failed to fetch URL: {str(e)}")

        # Extract readable content
        doc = Document(html)

        clean_html = doc.summary()

        soup = BeautifulSoup(
            clean_html,
            "html.parser"
        )

        text = soup.get_text(
            separator="\n",
            strip=True
        )

        MAX_WEBPAGE_CHARS = 15000

        text = text[:MAX_WEBPAGE_CHARS]

        return {
            "url": url,
            "title": doc.title() or "Untitled",
            "text": text
        }
