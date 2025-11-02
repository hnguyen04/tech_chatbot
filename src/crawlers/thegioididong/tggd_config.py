class TGGDConfig:
    BASE_URL = "https://www.thegioididong.com/tin-tuc/aj/Home/Box"
    HEADERS = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Accept": "*/*",
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
    }
    PAYLOAD_TEMPLATE = {
        "ID": "1169",
        "Size": "1000",   # số bài / trang
    }

    DETAILED_API_HEADERS = {"User-Agent": "Mozilla/5.0"}

    DOMAIN = "thegioididong.com"
