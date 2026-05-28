"""SKU 文本模糊对齐层 — Levenshtein + Trie 前缀树 + 子串匹配。

设计文档 5.5 / 5.5.1：
- 电商 UI 省略号截断导致 SKU 名与数据库标题不一致
- Trie 前缀树 O(K) 快速过滤候选集 → 缩小范围后才算编辑距离
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Levenshtein 编辑距离 ─────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


# ── Trie 前缀树 ──────────────────────────────────────────────

class _TrieNode:
    __slots__ = ('children', 'titles')

    def __init__(self):
        self.children: dict[str, "_TrieNode"] = {}
        self.titles: list[str] = []


class FastSKUIndex:
    """Trie 前缀树索引 — 微秒级候选集过滤。

    用法:
        idx = FastSKUIndex()
        idx.insert("综合知识笔试课")
        candidates = idx.search("综合知识笔试")  # → ["综合知识笔试课"]
    """

    def __init__(self):
        self._root = _TrieNode()
        self._size = 0

    def insert(self, title: str) -> None:
        """插入完整标题到前缀树。"""
        node = self._root
        for ch in title:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        node.titles.append(title)
        self._size += 1

    def batch_insert(self, titles: list[str]) -> None:
        for t in titles:
            self.insert(t)

    def search(self, prefix: str) -> list[str]:
        """前缀匹配候选集（O(K)，K = 前缀长度）。

        Returns:
            匹配到的完整标题列表（通常 0~5 条）
        """
        node = self._root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]

        # BFS 收集当前子树下所有标题
        result = []
        stack = [node]
        while stack:
            n = stack.pop()
            result.extend(n.titles)
            stack.extend(n.children.values())
        return result

    @property
    def size(self) -> int:
        return self._size


class SKUResolver:
    """SKU 模糊对齐器 — Trie 候选集 + Levenshtein 精准对齐。"""

    def __init__(self, title_tree: list[str] | None = None,
                 max_distance: int = 3):
        self._trie = FastSKUIndex()
        self._tree: list[str] = []
        self._max_distance = max_distance
        self._cache: dict[str, str] = {}
        self._hit_count = 0
        self._miss_count = 0
        self._trie_hit_count = 0
        self._lev_hit_count = 0

        if title_tree:
            self.load_tree(title_tree)

    def load_tree(self, titles: list[str]) -> None:
        self._tree = list(set(titles))
        self._trie = FastSKUIndex()
        self._trie.batch_insert(self._tree)
        self._cache.clear()
        logger.info("标题树加载: %d 个唯一标题", len(self._tree))

    def resolve(self, raw_sku: str) -> str:
        if not raw_sku or not self._tree:
            return raw_sku or ""

        if raw_sku in self._cache:
            return self._cache[raw_sku]

        # 去掉尾部省略号
        cleaned = raw_sku.rstrip("…...。. ")

        # 策略 1：子串包含匹配（快速路径）
        for full_title in self._tree:
            if cleaned in full_title or full_title in cleaned:
                self._cache[raw_sku] = full_title
                self._hit_count += 1
                return full_title

        # 策略 2：Trie 前缀匹配 → 候选集过滤
        candidates = self._trie.search(cleaned)
        if candidates:
            self._trie_hit_count += 1
            for full_title in candidates:
                if cleaned in full_title:
                    self._cache[raw_sku] = full_title
                    self._hit_count += 1
                    return full_title

        # 策略 3：候选集上 Levenshtein 精准对齐
        # 强前缀断言：对齐后的完整标题必须包含截断文本的所有字符
        search_set = candidates if candidates else self._tree
        best, best_dist = raw_sku, self._max_distance + 1
        for full_title in search_set[:20]:
            # 强前缀断言 — 防止"2026冲刺课"错位对齐到"2026冲刺课（升级版）"
            if cleaned not in full_title and full_title not in cleaned:
                continue
            dist = levenshtein_distance(cleaned, full_title)
            if dist < best_dist:
                best, best_dist = full_title, dist

        if best_dist <= self._max_distance:
            self._cache[raw_sku] = best
            self._lev_hit_count += 1
            self._hit_count += 1
            return best

        self._miss_count += 1
        logger.warning("SKU 无法匹配: '%s'", raw_sku)
        return raw_sku

    def resolve_batch(self, skus: list[str]) -> list[str]:
        return [self.resolve(s) for s in skus]

    def add_title(self, title: str) -> None:
        if title and title not in self._tree:
            self._tree.append(title)
            self._trie.insert(title)

    @property
    def stats(self) -> dict:
        total = self._hit_count + self._miss_count
        return {
            "tree_size": len(self._tree),
            "trie_size": self._trie.size,
            "cache_size": len(self._cache),
            "hit_count": self._hit_count,
            "trie_hits": self._trie_hit_count,
            "lev_hits": self._lev_hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self._hit_count / max(total, 1),
        }
