"""页面状态检测器 — 特征库 + LLaVA比对"""
import base64, logging, requests, json as _json, re, os

logger = logging.getLogger(__name__)

# 淘宝页面特征库
PAGE_FEATURES = {
    "桌面": "手机系统桌面/主屏幕。特征：底部有固定的电话/短信/浏览器等Dock栏，上方排列着各种App的圆角图标（不是商品卡片），顶部有状态栏（时间/电量）。没有搜索框、没有商品、没有价格。",
    "淘宝首页": "搜索栏在顶部，有'关注''推荐''直播'Tab，有商品推荐瀑布流",
    "搜索结果页": "搜索栏里有搜索词，下方有'综合''销量''全部''天猫'筛选Tab，显示商品列表",
    "店铺首页": "顶部显示店铺名称和'关注'按钮，有'首页''全部''新品''价格'Tab，有评分和粉丝数",
    "商品列表页": "网格排列的商品卡片，每张卡片有主图、标题、价格、付款人数",
    "商品详情页": "顶部是大图轮播，有价格和'加入购物车'按钮，有商品详情和评价Tab",
    "弹窗-优惠券": "半透明遮罩+居中弹窗，文字包含'领券''优惠''红包'，通常右上角有X",
    "弹窗-验证码": "滑块拼图验证，底部有滑块轨道",
}


class PageStateDetector:
    """用LLaVA比对页面特征，识别当前状态。

    用法:
        d = PageStateDetector()
        page, has_popup = d.identify("captures/screen.png")
    """

    def __init__(self, model: str = "llava:7b", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def identify(self, image_path: str) -> tuple[str, bool]:
        """返回 (页面类型, 是否有弹窗)。"""
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        # 先问：这是什么页面
        features_text = "\n".join(f"- {k}: {v}" for k, v in PAGE_FEATURES.items())
        prompt = (
            f"以下是淘宝App的各种页面特征：\n{features_text}\n\n"
            "当前截图最符合哪种页面？只回答页面类型名称，不要解释。"
        )

        page = self._ask(prompt, img_b64, 32).strip().rstrip("。.")
        if page not in PAGE_FEATURES:
            page = "其他"

        # 再问：有没有弹窗遮挡
        popup_prompt = "这张截图有弹窗遮挡吗？只回答'有'或'没有'。"
        ans = self._ask(popup_prompt, img_b64, 16).strip()
        has_popup = "有" in ans

        logger.info("页面: %s | 弹窗: %s", page, has_popup)
        return page, has_popup

    def find_popup_close(self, image_path: str) -> str | None:
        """用OCR找到弹窗关闭按钮位置。返回如'右上角'或None。"""
        try:
            import cv2, easyocr
            img = cv2.imread(image_path)
            if img is None:
                return None
            r = easyocr.Reader(['ch_sim', 'en'], gpu=False)
            h = img.shape[0]

            # 找关闭相关文字
            for t in r.readtext(img):
                text, conf, bbox = t[1].strip(), t[2], t[0]
                if conf < 0.4:
                    continue
                cy = int((bbox[0][1] + bbox[2][1]) / 2)
                # X / 关闭 / 跳过 通常在弹窗上方
                if text.lower() in ['x', '×', '✕'] or text in ['关闭', '跳过', '取消', '暂不']:
                    x_rel = "右上" if cy < h / 3 else ("中央" if cy < 2 * h / 3 else "底部")
                    logger.info("弹窗关闭按钮: '%s' @ %s", text, x_rel)
                    return x_rel

            # 找券/红包关键词（说明弹窗存在）
            for t in r.readtext(img):
                text = t[1]
                if t[2] > 0.5 and any(kw in text for kw in ['券', '红包', '优惠', '福利', '领']):
                    return "右上角"
        except Exception:
            pass
        return None

    def _ask(self, prompt: str, img_b64: str, num_predict: int = 128) -> str:
        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model, "prompt": prompt, "images": [img_b64],
                    "stream": False, "options": {"temperature": 0, "num_predict": num_predict}
                },
                timeout=60,
            )
            return resp.json().get("response", "")
        except Exception:
            return ""
