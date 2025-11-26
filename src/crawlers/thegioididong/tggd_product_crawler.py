from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from crawlers.thegioididong.tggd_config import TGGDConfig
from crawlers.core.article_document import ArticleDocument, ArticleSource, ArticleMetadata, ArticleContent, ArticleStatus
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
import json
import time
import os
import sys
import uuid
from datetime import datetime, timezone
import pandas as pd


class TGDDProductCrawler:
    def __init__(self, category: str = "dtdd", config: TGGDConfig = TGGDConfig(), crawl_limit: int = 100000):
        self.config = config
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options,
        )
        self.wait = WebDriverWait(self.driver, 10)
        self.url = self.config.BASE_URL + f"/{category}"
        self.category = category
        self.max_worker = max(1, os.cpu_count() - 1)
        self.crawl_limit = crawl_limit

    def load_all_products(self):
        self.driver.get(self.url)
        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.listproduct li.item")))

        last_count = 0
        while True:
            try:
                # Lấy lại nút xem thêm mỗi vòng
                see_more_btn = self.driver.find_element(By.CSS_SELECTOR, ".view-more .see-more-btn")
            except NoSuchElementException:
                break  # hết nút xem thêm

            # Scroll tới button
            self.driver.execute_script("arguments[0].scrollIntoView(true);", see_more_btn)
            time.sleep(0.3)

            # Click bằng JS (không cần tương tác trực tiếp)
            self.driver.execute_script("arguments[0].click();", see_more_btn)

            # Chờ số lượng li tăng
            try:
                self.wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "ul.listproduct li.item")) > last_count)
                last_count = len(self.driver.find_elements(By.CSS_SELECTOR, "ul.listproduct li.item"))
            except:
                break

            time.sleep(0.3)  # chờ load sản phẩm mới

    def get_products(self):
        """Lấy danh sách sản phẩm, luôn lấy bản gốc, bỏ qua merge items"""
        products = []
        base_site = self.config.BASE_URL

        # Load toàn bộ sản phẩm
        self.load_all_products()

        # Lấy lại danh sách item sau khi load xong (DOM mới nhất)
        li_elements = self.driver.find_elements(By.CSS_SELECTOR, "ul.listproduct li.item")
        print(f"🛒 Tìm thấy {len(li_elements)} sản phẩm trong danh mục {self.category}")

        for index, li in enumerate(li_elements):
            if index > self.crawl_limit:  
                break

            try:
                a_tag = li.find_element(By.TAG_NAME, "a")
                base_title = a_tag.get_attribute("data-name")
                base_url = urljoin(base_site, a_tag.get_attribute("href"))

                base_price = li.find_element(By.CSS_SELECTOR, "strong.price").text.strip()

                print(f"🛒 [{index+1}/{len(li_elements)}] Lấy sản phẩm: {base_title} - {base_url}")

                # Append kết quả
                products.append({
                    "url": base_url,
                    "title": base_title,
                    "price": base_price,
                })

            except NoSuchElementException:
                continue
            except Exception as e:
                print(f"Error processing product at index {index}: {e}")
                continue

        return products


    def get_articles(self) -> list[ArticleDocument]:
        """Chuyển sản phẩm đã crawl sang list ArticleDocument"""
        products = self.get_products()
        now = datetime.now(timezone.utc).isoformat()
        articles = []


        for p in products:
            detail = self.get_product_detail(p["url"])
            doc_id = f"doc_{now[:10].replace('-', '')}_{uuid.uuid4().hex}"
            article = ArticleDocument(
                id=doc_id,
                source=ArticleSource(
                    url=p.get("url", ""),
                    domain= self.config.DOMAIN,
                    crawl_date=now
                ),
                metadata=ArticleMetadata(
                    title=p.get("title", ""),
                    author="Không rõ",
                    category=self.category,
                    price=p.get("price", None),
                ),
                content=ArticleContent(
                    raw_html=None,
                    text=detail.get("info_text", ""),
                    images=detail.get("images", []),
                    summary="",
                    product=detail.get("specs", {})
                ),
                status=ArticleStatus(
                    parsed=True,
                    cleaned=False,
                    embedded=False,
                    last_updated=now
                )
            )
            articles.append(article)

        return articles
    
    def get_product_detail(self, product_url: str) -> dict:

        print(f"🛒 Lấy chi tiết sản phẩm: {product_url}")

        def retry_on_stale(func, retries=3, sleep=0.2):
            for _ in range(retries):
                try:
                    return func()
                except StaleElementReferenceException:
                    time.sleep(sleep)
                    continue
            return None

        def exists(css):
            try:
                return len(self.driver.find_elements(By.CSS_SELECTOR, css)) > 0
            except Exception:
                return False
        
        result = {
            "specs": {},
            "info_text": "",
            "images": []
        }
        
        try: 
            self.driver.get(product_url)
        except TimeoutException:
            print(f"Timeout while loading {product_url}")
            return result

        wait = self.wait
        base_site = self.config.DOMAIN

        # ========== 1. Specs ==========
        def get_box_specs():
            return self.driver.find_elements(By.CSS_SELECTOR, "div.box-specifi")

        box_divs = retry_on_stale(lambda: get_box_specs()) or []

        if box_divs:
            for b in range(len(box_divs)):
                def get_box_at():
                    return self.driver.find_elements(By.CSS_SELECTOR, "div.box-specifi")[b]

                box_div = retry_on_stale(lambda: get_box_at())
                if box_div is None:
                    continue

                try:
                    tab_a_list = retry_on_stale(lambda: box_div.find_elements(By.TAG_NAME, "a"))
                    tab_a = tab_a_list[0] if tab_a_list else None
                    if tab_a is None:
                        continue
                    ul_elem = retry_on_stale(lambda: box_div.find_element(By.CSS_SELECTOR, "ul.text-specifi"))
                    if ul_elem is None:
                        continue

                    ul_class = ""
                    try:
                        ul_class = ul_elem.get_attribute("class") or ""
                    except Exception:
                        ul_class = ""

                    is_active = "active" in ul_class

                    if not is_active and tab_a is not None:
                        try:
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", tab_a)
                            self.driver.execute_script("arguments[0].click();", tab_a)
                            time.sleep(0.4)
                            ul_elem = retry_on_stale(lambda: box_div.find_element(By.CSS_SELECTOR, "ul.text-specifi")) or ul_elem
                        except Exception:
                            pass

                    li_items = retry_on_stale(lambda: ul_elem.find_elements(By.TAG_NAME, "li")) or []

                    for li in li_items:
                        asides = retry_on_stale(lambda: li.find_elements(By.TAG_NAME, "aside")) or []
                        if len(asides) < 2:
                            continue

                        # key = nối tất cả text con của aside[0]
                        try:
                            key_text = " ".join([el.text.strip() for el in asides[0].find_elements(By.XPATH, ".//*") if el.text.strip()])
                            if not key_text:
                                key_text = asides[0].text.strip()
                        except Exception:
                            key_text = ""

                        # value = nối tất cả text con của aside[1]
                        try:
                            val_text = " ".join([el.text.strip() for el in asides[1].find_elements(By.XPATH, ".//*") if el.text.strip()])
                            if not val_text:
                                val_text = asides[1].text.strip()
                        except Exception:
                            val_text = ""

                        if key_text:
                            if key_text in result["specs"]:
                                i = 2
                                new_key = f"{key_text} ({i})"
                                while new_key in result["specs"]:
                                    i += 1
                                    new_key = f"{key_text} ({i})"
                                key_text = new_key

                            result["specs"][key_text] = val_text

                except Exception as e:
                    print(product_url, "Error processing box:", e)
                    continue

        else:
            # ========== Template 2 ==========
            try:
                ul_list = retry_on_stale(lambda: self.driver.find_elements(By.CSS_SELECTOR, "ul.parameter")) or []
                for ul in ul_list:
                    li_list = retry_on_stale(lambda: ul.find_elements(By.TAG_NAME, "li")) or []

                    for li in li_list:
                        key_text = ""
                        val_text = ""
                        try:
                            span_elem = retry_on_stale(lambda: li.find_element(By.TAG_NAME, "span"))
                            if span_elem is not None:
                                try:
                                    # try a inside span first
                                    a_in_span = retry_on_stale(lambda: span_elem.find_element(By.TAG_NAME, "a"))
                                    if a_in_span is not None:
                                        key_text = a_in_span.text.strip()
                                    else:
                                        key_text = span_elem.text.strip()
                                except Exception:
                                    key_text = span_elem.text.strip()
                        except Exception:
                            key_text = ""

                        try:
                            div_elem = retry_on_stale(lambda: li.find_element(By.TAG_NAME, "div"))
                            if div_elem is not None:
                                p_list = retry_on_stale(lambda: div_elem.find_elements(By.TAG_NAME, "p")) or []
                                val_text = ", ".join([p.text.strip() for p in p_list if p.text and p.text.strip()])
                        except Exception:
                            val_text = ""

                        if key_text:
                            if key_text in result["specs"]:
                                i = 2
                                new_key = f"{key_text} ({i})"
                                while new_key in result["specs"]:
                                    i += 1
                                    new_key = f"{key_text} ({i})"
                                key_text = new_key
                            result["specs"][key_text] = val_text
            except Exception as e:
                print(product_url, "Error processing template 2 specs:", e)

        # ========== 2. Info text ==========
        # chỉ click nếu selector tồn tại
        try:
            if exists("h2.tab-link[data-tab='tab-2']"):
                def click_info_tab():
                    tab_info = self.driver.find_element(By.CSS_SELECTOR, "h2.tab-link[data-tab='tab-2']")
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", tab_info)
                    self.driver.execute_script("arguments[0].click();", tab_info)

                retry_on_stale(lambda: click_info_tab())

                # sau khi click, cố gắng lấy div.text-detail nếu có
                if exists("div.text-detail"):
                    text_detail = retry_on_stale(lambda: self.driver.find_element(By.CSS_SELECTOR, "div.text-detail").text.strip())
                    result["info_text"] = text_detail or ""
            else:
                # fallback: một số template có section.feature hoặc article#product-main
                if exists("section.feature"):
                    try:
                        sec = self.driver.find_element(By.CSS_SELECTOR, "section.feature")
                        result["info_text"] = sec.text.strip()
                    except Exception:
                        result["info_text"] = result["info_text"]
                elif exists("article#product-main"):
                    try:
                        art = self.driver.find_element(By.CSS_SELECTOR, "article#product-main")
                        result["info_text"] = art.text.strip()
                    except Exception:
                        result["info_text"] = result["info_text"]
        except Exception as e:
            # bảo vệ chung, KHÔNG throw ra ngoài
            print("Error processing info text:", product_url, e)

        # ========== 3. Images ==========
        try:
            if exists("div.gallery-img img"):
                imgs = retry_on_stale(lambda: self.driver.find_elements(By.CSS_SELECTOR, "div.gallery-img img")) or []
                result["images"] = [urljoin(base_site, img.get_attribute("src")) for img in imgs if img.get_attribute("src")]
            elif exists("img.gallery-slide-item.loaded"):
                imgs2 = retry_on_stale(lambda: self.driver.find_elements(By.CSS_SELECTOR, "img.gallery-slide-item.loaded")) or []
                result["images"] = [urljoin(base_site, img.get_attribute("src")) for img in imgs2 if img.get_attribute("src")]
            else:
                # fallback: tìm bất kỳ img trong product area
                if exists("article#product-main img"):
                    imgs3 = retry_on_stale(lambda: self.driver.find_elements(By.CSS_SELECTOR, "article#product-main img")) or []
                    result["images"] = [urljoin(base_site, img.get_attribute("src")) for img in imgs3 if img.get_attribute("src")]
        except Exception:
            pass

        return result
        
    def close(self):
        self.driver.quit()

    def run(self) -> list[ArticleDocument]:
        """Chạy crawler, trả về list ArticleDocument"""
        print("🛒 Lấy danh sách sản phẩm ...")
        articles = self.get_articles()
        self.close()
        print(f"🛒 Hoàn thành lấy {len(articles)} sản phẩm.")
        return articles

