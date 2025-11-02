import requests
from crawlers.thegioididong.tggd_config import TGGDConfig

class TGGDApiClient:
    """
    Gọi API của thegioididong để lấy danh sách bài viết HTML.
    """

    def __init__(self, config: TGGDConfig = TGGDConfig()):
        self.config = config

    def fetch_articles_html(self, index: int) -> str:
        """
        Gửi POST request để lấy danh sách bài viết.
        """
        payload = self.config.PAYLOAD_TEMPLATE.copy()
        payload["Index"] = str(index)
        response = requests.post(
            self.config.BASE_URL,
            headers=self.config.HEADERS,
            data=payload,
            timeout=10,
        )
        response.raise_for_status()
        return response.text