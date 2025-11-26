class TGGDConfig:
    ARTICLE_BASE_URL = "https://www.thegioididong.com/tin-tuc/aj/Home/Box"
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
    BASE_ARTICLE_URL = "https://www.thegioididong.com/tin-tuc"
    DOMAIN = "thegioididong.com"
    BASE_URL = "https://www.thegioididong.com"
    PRODUCT_CATEGORY_LIST = [
        # "dtdd", 
        # "laptop"
        # "sac-dtdd",
        # "op-lung-flipcover",
        # "op-lung-may-tinh-bang",
        # "mieng-dan-man-hinh",
        # "mieng-dan-camera",
        # "tui-dung-airpods",
        # "airtag",
        # "phu-kien-thong-minh",
        # "gia-do-dien-thoai",
        # "phu-kien-op-lung",
        # "day-dong-ho",
        # "tai-nghe-bluetooth",
        # "tai-nghe-co-day",
        # "tai-nghe-chup-tai",
        # "micro-cac-loai",
        # "loa-laptop",
        # "tai-nghe-the-thao",
        # "pin",
        # "quat-mini",
        # "chuot-may-tinh-chuot-gaming",
        # "ban-phim-gaming",
        # "tai-nghe-gaming",
        # "mieng-lot-chuot",
        # "may-choi-game-cam-tay-tay-cam-choi-game",
        # "may-tinh-bang",
        # "hub-chuyen-doi",
        # "chuot-may-tinh",
        # "ban-phim",
        # "thiet-bi-mang",
        # "tui-chong-soc",
        # "tui-dung-phu-kien",
        # "mieng-phu-ban-phim",
        # "phan-mem",
        # "gia-treo-man-hinh",
        # "den-dien-den-sac",
        # "bang-ve-dien-tu",
        # "o-cung-di-dong",
        # "the-nho-dien-thoai",
        # "usb",
        "camera-giam-sat",
        "flycam",
        "camera-hanh-trinh-hanh-dong",
        "gay-tu-suong",
        "may-chieu",
        "muc-in",
        "may-in",
        "may-tinh-de-ban",
        "man-hinh-may-tinh",
        "bo-luu-dien",
    ]
