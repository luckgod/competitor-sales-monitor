"""页面特征库 — 模型和代码共用的页面识别标准"""
import base64, logging, os, requests

logger = logging.getLogger(__name__)

# 每种页面的特征定义
PAGE_FEATURES = {
    "桌面": {
        "ocr_keywords": ["电话", "短信", "相机", "设置", "时钟"],
        "visual": "手机系统桌面，底部固定Dock栏，上方App图标网格",
        "action": "打开淘宝App",
    },
    "淘宝首页": {
        "ocr_keywords": ["关注", "推荐", "直播", "618", "天猫"],
        "visual": "搜索栏在顶部，关注/推荐/直播Tab，商品推荐瀑布流",
        "action": "点击搜索框",
    },
    "搜索输入页": {
        "ocr_keywords": ["历史记录", "搜索发现", "热搜"],
        "visual": "搜索框有光标，下方搜索历史/热榜，键盘弹出",
        "action": "输入店铺名",
    },
    "搜索结果页": {
        "ocr_keywords": ["综合", "销量", "筛选", "天猫"],
        "visual": "顶部搜索框有搜索词，下方商品列表，有筛选Tab",
        "action": "找店铺入口卡片并点击",
    },
    "店铺首页": {
        "ocr_keywords": ["关注", "粉丝", "全部", "新品", "首页"],
        "visual": "顶部店铺名+关注按钮，评分和粉丝数，商品Tab",
        "action": "下滑浏览商品或点击全部商品",
    },
    "商品列表": {
        "ocr_keywords": ["人付款", "¥", "已售"],
        "visual": "网格排列的商品卡片，含主图/标题/价格/付款人数",
        "action": "切分卡片提取商品信息",
    },
    "商品详情页": {
        "ocr_keywords": ["加入购物车", "立即购买", "优惠券", "店铺"],
        "visual": "大图轮播，价格，加购按钮，商品详情Tab",
        "action": "找实时销量弹窗入口或月销数据",
    },
    "弹窗-优惠券": {
        "ocr_keywords": ["领券", "优惠", "红包", "福利", "限时"],
        "visual": "半透明遮罩+居中弹窗，有X关闭按钮",
        "action": "点X或按返回键关闭",
    },
    "弹窗-验证码": {
        "ocr_keywords": ["滑块", "验证", "拼图"],
        "visual": "滑块拼图验证，底部滑块轨道",
        "action": "蜂鸣通知人工处理",
    },
}


class PageLibrary:
    """页面特征库 — 为模型和代码提供统一的页面识别标准。

    用法:
        lib = PageLibrary()
        page_name = lib.identify_by_ocr(["关注","推荐","直播","618"])
        # -> "淘宝首页"

        page_name = lib.ask_model("captures/screen.png")
        # -> "搜索结果页"
    """

    def __init__(self, model: str = "minicpm-v:latest",
                 base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._ref_dir = "data/pages"

    # ── OCR文字匹配 ────────────────────────────────────────

    def identify_by_ocr(self, ocr_texts: list[str]) -> tuple[str, float]:
        """根据OCR文字列表匹配页面类型，返回(页面名, 置信度)。"""
        best_page = "未知"
        best_score = 0.0

        all_text = " ".join(ocr_texts)
        for page_name, features in PAGE_FEATURES.items():
            keywords = features["ocr_keywords"]
            hits = sum(1 for kw in keywords if kw in all_text)
            score = hits / max(len(keywords), 1)
            if score > best_score:
                best_score = score
                best_page = page_name

        return best_page, best_score

    # ── 模型视觉识别 ──────────────────────────────────────

    def ask_model(self, image_path: str) -> str:
        """让VLM根据特征库识别页面类型。"""
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        # 构建特征描述
        features_text = "\n".join(
            f"- {k}: {v['visual']}" for k, v in PAGE_FEATURES.items()
        )

        prompt = (
            f"以下是各种手机页面的特征：\n{features_text}\n\n"
            "当前截图最符合哪种页面？只回答页面类型名称，不要解释。"
        )

        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model, "prompt": prompt, "images": [img_b64],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 32},
                },
                timeout=120,
            )
            ans = resp.json().get("response", "").strip().rstrip("。.")
            if ans in PAGE_FEATURES:
                return ans
        except Exception:
            logger.exception("模型识别失败")

        return "未知"

    def ask_yesno(self, image_path: str, question: str) -> bool:
        """让模型看图回答是/否。"""
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model, "prompt": question + " 只回答'是'或'否'。",
                    "images": [img_b64], "stream": False,
                    "options": {"temperature": 0, "num_predict": 16},
                },
                timeout=120,
            )
            ans = resp.json().get("response", "").strip()
            return "是" in ans
        except Exception:
            return False

    def get_action(self, page_name: str) -> str:
        """获取当前页面应该执行的下一步动作。"""
        return PAGE_FEATURES.get(page_name, {}).get("action", "未知动作")

    def list_pages(self) -> list[str]:
        return list(PAGE_FEATURES.keys())
