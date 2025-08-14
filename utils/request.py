import logging
import time

import requests

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
    "accept-encoding": "gzip",
}


class RequestHandler:
    def __init__(self, retries=3, delay=3, timeout=10, proxy=None, headers=None):
        """
        :param retries: 最大重试次数
        :param delay: 每次重试的间隔时间（秒）
        :param timeout: 请求超时时间（秒）
        """
        if proxy is None:
            proxy = {}
        self.proxy = proxy
        self.headers = headers
        self.retries = retries
        if headers is None:
            self.headers = HEADERS
        self.retries = retries
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()

    def request(self, method, full_url, **kwargs):
        for attempt in range(1, self.retries + 1):
            try:
                self.session.headers.update(self.headers)
                response = self.session.request(method, full_url, timeout=self.timeout, proxies=self.proxy, **kwargs)

                if response.status_code in (200, 201, 202):
                    return response
                if response.status_code == 210:
                    log.error(
                        f"[{method}] 请求失败 (状态码: {response.status_code})，URL: {full_url}，信息为：{response.json().get('message')}\n 对于210只能更换代理或者等待1小时后重试，我们无能为力")
                    exit(1)
                log.warning(
                    f"[{method}] 请求失败 (状态码: {response.status_code})，URL: {full_url}，尝试第 {attempt}/{self.retries} 次...")

            except requests.RequestException as e:
                log.warning(f"[{method}] 请求异常: {e}，URL: {full_url}，尝试第 {attempt}/{self.retries} 次...")

            time.sleep(self.delay)

        log.error(f"[{method}] 请求失败: 超过最大重试次数 ({self.retries})，URL: {full_url}")
        return None

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)
