# museum_rag_core.py
# ==============================================================================
# 博物館 RAG 核心模組
#   - 雙域檢索（basic / hongloumeng）
#   - TF-IDF rerank
#   - OpenAI 文字導覽
#   - Gemini 2.5 Image：生成橫式情境圖（若未啟用則使用原始文物照片）
# ==============================================================================

print("[RAG] museum_rag_core 模組正在載入...")

import os
import re
import glob
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import yaml
import jieba
import numpy as np
import difflib

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from PIL import Image as PILImage
import PIL._util
from io import BytesIO

from openai import OpenAI
from google import genai
from google.genai import types

# Pillow 相容修補（某些版本會缺 is_directory）
if not hasattr(PIL._util, "is_directory"):
    PIL._util.is_directory = lambda path: os.path.isdir(path)

# ==============================================================================
# 基本參數
# ==============================================================================

OCR_ENABLED = False
EMBED_BATCH = 16
K_VEC = 30
K_BM25 = 50
TOP_N_FINAL = 5

BASIC_CHUNK = 200
BASIC_OVERLAP = 100
HONG_CHUNK = 300
HONG_OVERLAP = 60

# Markdown 資料庫位置
BASE_DIR = os.path.join(os.path.dirname(__file__), "data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
print(f"[RAG] 資料目錄: {BASE_DIR}")

# ==============================================================================
# OpenAI 初始化（文字導覽）
# ==============================================================================

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
_openai_client: Optional[OpenAI] = None
USE_LLM = False


def init_openai() -> None:
    """從環境變數讀取 OPENAI_API_KEY，初始化文字模型。"""
    global _openai_client, USE_LLM
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print("[RAG] 未設定 OPENAI_API_KEY，將只回傳檢索片段，不使用 LLM 生成導覽。")
        USE_LLM = False
        _openai_client = None
        return

    try:
        _openai_client = OpenAI(api_key=key)
        USE_LLM = True
        print(f"[RAG] OpenAI 初始化成功，model = {OPENAI_MODEL}")
    except Exception as e:
        print(f"[RAG] OpenAI 初始化失敗：{e}，改為只回傳檢索片段。")
        USE_LLM = False
        _openai_client = None


init_openai()

# ==============================================================================
# Document 結構與載入 / chunk / embedding
# ==============================================================================


def zh_tokens(text: str) -> List[str]:
    return [t.strip() for t in jieba.lcut(text or "") if t.strip()]


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]


FRONT_MATTER_RE = re.compile(r"^\ufeff?---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_front_matter(text: str) -> dict:
    m = FRONT_MATTER_RE.match(text or "")
    if not m:
        return {}
    try:
        data = yaml.safe_load(m.group(1))
        if isinstance(data, dict):
            return {
                str(k).strip(): (str(v).strip() if isinstance(v, str) else v)
                for k, v in data.items()
                if k is not None
            }
    except Exception:
        pass
    return {}


def strip_front_matter(text: str) -> str:
    if FRONT_MATTER_RE.match(text or ""):
        return FRONT_MATTER_RE.sub("", text, count=1)
    return text or ""


def infer_domain_from_name(path: str) -> str:
    name = Path(path).name.lower()
    if name.endswith("_basic.md"):
        return "basic"
    if name.endswith("_hongloumeng.md"):
        return "hongloumeng"
    return "basic"


def load_md_as_docs_by_domain(base_dir: str) -> Tuple[List[Document], List[Document]]:
    docs_basic: List[Document] = []
    docs_hong: List[Document] = []

    md_files = glob.glob(str(Path(base_dir) / "**/*.md"), recursive=True) + glob.glob(
        str(Path(base_dir) / "**/*.markdown"), recursive=True
    )

    for p in md_files:
        raw = ""
        try:
            raw = Path(p).read_text(encoding="utf-8", errors="strict")
        except Exception:
            try:
                raw = Path(p).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

        meta = parse_front_matter(raw)
        body = strip_front_matter(raw)

        domain = (meta.get("domain") or infer_domain_from_name(p) or "basic").strip()
        stem = Path(p).stem
        obj = (
            meta.get("object_name")
            or (stem.split("_")[0] if "_" in stem else stem)
            or "unknown"
        ).strip()

        doc = Document(
            page_content=body,
            metadata={
                "source": str(p),
                "object_name": obj,
                "domain": domain,
                "type": "markdown",
            },
        )
        if domain == "hongloumeng":
            docs_hong.append(doc)
        else:
            docs_basic.append(doc)

    return docs_basic, docs_hong


print("[RAG] 正在載入資料庫...")
_docs_basic, _docs_hong = load_md_as_docs_by_domain(BASE_DIR)
print(f"[RAG][LOAD] basic={len(_docs_basic)} hongloumeng={len(_docs_hong)}")


def split_docs(
    docs: List[Document], size: int, overlap: int, add_parent: bool = False
) -> List[Document]:
    out: List[Document] = []
    step = max(1, size - overlap)
    for d in docs:
        text = d.page_content or ""
        n, i, chunk_idx = len(text), 0, 0
        while i < n:
            j = min(i + size, n)
            meta = dict(d.metadata)
            if add_parent:
                meta["parent_id"] = meta.get("source")
            meta["span_start"] = i
            meta["span_end"] = j
            meta["chunk_idx"] = chunk_idx
            out.append(Document(page_content=text[i:j], metadata=meta))
            i += step
            chunk_idx += 1
    return out


_splits_basic = split_docs(_docs_basic, BASIC_CHUNK, BASIC_OVERLAP, add_parent=False)
_splits_hong = split_docs(_docs_hong, HONG_CHUNK, HONG_OVERLAP, add_parent=True)
print(f"[RAG][SPLIT] basic chunks={len(_splits_basic)} hong chunks={len(_splits_hong)}")

print("[RAG] 載入 Embedding 模型...")
import torch

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = SentenceTransformer("intfloat/multilingual-e5-base", device=_device)


def _embed_texts(texts: List[str], is_query: bool) -> np.ndarray:
    prefix = "query: " if is_query else "passage: "
    texts_p = [(prefix + (t or "")).strip() for t in texts]
    return (
        _model.encode(
            texts_p,
            batch_size=EMBED_BATCH if not is_query else 1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        .astype("float32")
        .copy()
    )


def _embed_docs(docs: List[Document]) -> np.ndarray:
    if not docs:
        return np.zeros((0, 768), dtype="float32")
    texts = [d.page_content for d in docs]
    vecs = _embed_texts(texts, is_query=False)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)


def _embed_query(q: str) -> np.ndarray:
    v = _embed_texts([q], is_query=True)[0]
    return v / (np.linalg.norm(v) + 1e-8)


print("[RAG][EMB] 建立索引...")
_vecs_basic = _embed_docs(_splits_basic)
_vecs_hong = _embed_docs(_splits_hong)


# ==============================================================================
# 全局文物清單與模糊匹配
# ==============================================================================

def get_all_artifact_names(docs_basic: List[Document], docs_hong: List[Document]) -> List[str]:
    names = set()
    for d in docs_basic + docs_hong:
        name = (d.metadata.get("object_name") or "").strip()
        if name:
            names.add(name)
    return sorted(list(names))

print("[RAG] 建立全域文物名稱清單...")
GLOBAL_ARTIFACTS = get_all_artifact_names(_docs_basic, _docs_hong)
print(f"[RAG] 總共找到 {len(GLOBAL_ARTIFACTS)} 件文物。")

# --- 將文物名稱加入 jieba 詞庫，提升斷詞精準度 ---
for name in GLOBAL_ARTIFACTS:
    jieba.add_word(name)
    # 也加入一些核心部分 (例如 "提梁壺")
    if "壺" in name: jieba.add_word(name.split("壺")[0] + "壺")
    if "蓋碗" in name: jieba.add_word(name.split("蓋碗")[0] + "蓋碗")

from pypinyin import pinyin, Style

def to_pinyin_str(text):
    """將文字轉為拼音字串，方便比對讀音是否相同 (如 梁 vs 樑)"""
    if not text: return ""
    # 使用 Style.NORMAL 拿到單純拼音 (如 liang)
    res = pinyin(text, style=Style.NORMAL)
    return "".join([item[0] for item in res]).lower()

def find_best_artifact_match(query: str, threshold: float = 0.3) -> List[Tuple[str, float]]:
    """
    超級強化版的模糊匹配：
    1. 針對自然語言句子進行遍歷。
    2. 使用最長共同子字串 (Longest Common Substring) 概念。
    3. 忽略常見干擾詞。
    """
    if not query or not GLOBAL_ARTIFACTS:
        return []
    
    matches = []
    # 移除標點，保留空格與中文字
    query_clean = re.sub(r'[^\w]', '', query).strip()
    
    def normalize_for_match(text):
        # 將所有常見變體統一，並轉為拼音進行二次校驗
        return text.replace("樑", "梁").replace("提橡", "提梁").replace("壺", "壶")

    q_norm = normalize_for_match(query_clean)
    q_pinyin = to_pinyin_str(q_norm)

    print(f"[RAG][DEBUG] Cleaning Query: '{query_clean}'")

    for art in GLOBAL_ARTIFACTS:
        art_norm = normalize_for_match(art)
        art_pinyin = to_pinyin_str(art_norm)
        
        # A. 直接匹配
        if art_norm in q_norm or q_norm in art_norm:
            score = 0.95
            matches.append((art, score))
            continue

        # B. 檢查 artifact 的重要部分是否出現在 query
        # 例如 "宜興胎畫琺瑯提梁壺" 的核心可能是最後幾個字
        core_parts = []
        if len(art_norm) > 4:
            core_parts.append(art_norm[-3:]) # 最後三字
            core_parts.append(art_norm[-4:]) # 最後四字
            core_parts.append(art_norm[:4])  # 最前四字
        
        found_core = False
        for part in core_parts:
            if part in q_norm:
                found_core = True
                break
        
        if found_core:
            # 如果抓到核心字眼，給予基礎分並依據 pinyin 比例微調
            score = 0.85
            matches.append((art, score))
            continue

        # C. 讀音比對 (pinyin)
        # 檢查是否有連續 3 個以上的拼音重合
        if len(q_pinyin) > 6 and len(art_pinyin) > 6:
            # 這裡用簡易相似度
            ratio = difflib.SequenceMatcher(None, q_pinyin, art_pinyin).ratio()
            # 針對長句子，ratio 會被拉低，所以我們找最長連續匹配
            match = difflib.SequenceMatcher(None, q_pinyin, art_pinyin).find_longest_match(0, len(q_pinyin), 0, len(art_pinyin))
            if match.size >= 6: # 約兩到三個中文字的拼音長度
                score = 0.80 + (min(match.size / 20, 0.15))
                matches.append((art, score))

    # 按分數排序
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # 去重
    seen = set()
    final_matches = []
    for a, s in matches:
        if a not in seen:
            final_matches.append((a, s))
            seen.add(a)

    if final_matches:
        debug_info = ", ".join([f"'{m[0]}'({m[1]:.2f})" for m in final_matches[:3]])
        print(f"[RAG][FUZZY] Result: {debug_info}")
    else:
        print(f"[RAG][FUZZY] No matches found for '{query_clean}'")
        
    return final_matches

class NumpyVectorRetriever:
    def __init__(self, docs: List[Document], mat: np.ndarray, k: int):
        self.docs = docs
        self.mat = mat
        self.k = k

    def search(self, query: str) -> List[Document]:
        if len(self.mat) == 0:
            return []
        q = _embed_query(query)
        sims = self.mat @ q
        return [self.docs[i] for i in np.argsort(-sims)[: self.k]]


class BM25Retriever:
    def __init__(self, docs: List[Document], k: int):
        self.docs = docs
        self.k = k
        self.bm25 = BM25Okapi([zh_tokens(d.page_content) for d in docs]) if docs else None

    def search(self, query: str) -> List[Document]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(zh_tokens(query))
        return [self.docs[i] for i in np.argsort(-scores)[: self.k]]


_vec_basic_ret = NumpyVectorRetriever(_splits_basic, _vecs_basic, K_VEC)
_vec_hong_ret = NumpyVectorRetriever(_splits_hong, _vecs_hong, K_VEC)
_bm25_basic = BM25Retriever(_splits_basic, K_BM25)
_bm25_hong = BM25Retriever(_splits_hong, K_BM25)


def merge_dedup(dlists: List[List[Document]], limit: int) -> List[Document]:
    out: List[Document] = []
    seen: set[str] = set()
    for lst in dlists:
        for d in lst:
            key = hashlib.md5(
                (d.page_content + json.dumps(d.metadata, sort_keys=True)).encode("utf-8")
            ).hexdigest()
            if key not in seen:
                seen.add(key)
                out.append(d)
                if len(out) >= limit:
                    return out
    return out


def tfidf_rerank(docs: List[Document], query: str, top_n: int) -> List[Document]:
    if not docs:
        return []
    texts = [d.page_content for d in docs]
    vectorizer = TfidfVectorizer(tokenizer=zh_tokens, max_features=20000)
    X = vectorizer.fit_transform(texts + [query])
    sims = linear_kernel(X[-1], X[:-1]).flatten()
    return [docs[i] for i in np.argsort(-sims)[:top_n]]


_src_cache: Dict[str, str] = {}


def read_source(path: str) -> str:
    if path not in _src_cache:
        try:
            _src_cache[path] = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            _src_cache[path] = ""
    return _src_cache[path]


def format_docs_with_source(docs: List[Document]) -> str:
    out: List[str] = []
    for i, d in enumerate(docs, 1):
        name = Path(d.metadata.get("source", "")).name
        out.append(f"[{i}] {d.page_content.strip()}\n〈來源:{name}〉")
    return "\n\n".join(out)


def expand_parent_context(docs: List[Document], max_chars: int = 8000) -> str:
    if not docs:
        return ""
    if docs[0].metadata.get("domain") != "hongloumeng":
        return format_docs_with_source(docs)

    seen: set[str] = set()
    buf: List[str] = []
    for d in docs:
        pid = d.metadata.get("parent_id") or d.metadata.get("source")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        full = strip_front_matter(read_source(pid))
        if full:
            buf.append(f"{full[:max_chars]}\n〈來源:{Path(pid).name}〉")
    return "\n\n".join(buf)


def build_context_for_prompt(
    hits_basic: List[Document], hits_hong: List[Document]
) -> str:
    blocks: List[str] = []
    if hits_basic:
        blocks.append("【basic】\n" + format_docs_with_source(hits_basic)[:2500])
    if hits_hong:
        blocks.append("【hongloumeng】\n" + expand_parent_context(hits_hong))
    return "\n\n".join(blocks)


def retrieve_two_domains(q: str, artifact_name: Optional[str] = None) -> Tuple[List[Document], List[Document]]:
    cand_basic = merge_dedup(
        [_vec_basic_ret.search(q), _bm25_basic.search(q)], limit=K_VEC
    )
    cand_hong = merge_dedup(
        [_vec_hong_ret.search(q), _bm25_hong.search(q)], limit=K_VEC
    )

    # --- 模糊匹配或鎖定文物強化 ---
    top_art = artifact_name
    if not top_art:
        fuzzy_matches = find_best_artifact_match(q)
        if fuzzy_matches and fuzzy_matches[0][1] > 0.75:
            top_art = fuzzy_matches[0][0]

    if top_art:
        # 找出所有屬於該文物的 docs
        extra_basic = [d for d in _splits_basic if (d.metadata.get("object_name") or "").strip() == top_art]
        extra_hong = [d for d in _splits_hong if (d.metadata.get("object_name") or "").strip() == top_art]
        
        # 如果是「鎖定文物」模式，我們強烈過濾掉其他文物的候選，只保留該文物的內容
        if artifact_name:
            cand_basic = merge_dedup([extra_basic, [d for d in cand_basic if (d.metadata.get("object_name") or "").strip() == artifact_name]], limit=K_VEC + 20)
            cand_hong = merge_dedup([extra_hong, [d for d in cand_hong if (d.metadata.get("object_name") or "").strip() == artifact_name]], limit=K_VEC + 20)
        else:
            # 一般搜尋模式，把模糊匹配到的 docs 混入 candidate 列表的最前面
            cand_basic = merge_dedup([extra_basic, cand_basic], limit=K_VEC + 20)
            cand_hong = merge_dedup([extra_hong, cand_hong], limit=K_VEC + 20)

    hits_basic = tfidf_rerank(cand_basic, q, top_n=TOP_N_FINAL)
    hits_hong = tfidf_rerank(cand_hong, q, top_n=TOP_N_FINAL)

    # --- 強制介入：如果鎖定/模糊匹配強烈，但 TF-IDF 沒選中，強行塞入 ---
    if top_art:
        def force_inject(hits, extra, art_name):
            if not any((d.metadata.get("object_name") or "").strip() == art_name for d in hits):
                if extra:
                    best_extra = sorted(extra, key=lambda d: len(d.page_content), reverse=True)[0]
                    return [best_extra] + hits[:-1]
            return hits

        hits_basic = force_inject(hits_basic, [d for d in _splits_basic if (d.metadata.get("object_name") or "").strip() == top_art], top_art)
        hits_hong = force_inject(hits_hong, [d for d in _splits_hong if (d.metadata.get("object_name") or "").strip() == top_art], top_art)

    return hits_basic, hits_hong


# ==============================================================================
# LLM Prompt 模板（只保留與展品相關的 few-shot，不處理便利商店之類問題）
# ==============================================================================

few_shot_example = """

【範例一：展品資料可回答的事實性問題】
問題：這個細細雕著花紋的瓶子，和其他茶具有什麼不一樣嗎？
回答：您好！您提到的這件細緻雕花的瓶子，和一般茶具有些不同。它的裝飾並不張揚，而是以淺刻的花紋和素雅的釉色呈現一種內斂的氣質，更像是用來慢慢品味的器物。

• 造型風格：線條收斂、比例修長，給人安靜沉穩的感受。
• 紋飾細節：花紋多為淺刻或暗紋，需要近看才發現。
• 整體氣氛：不像華麗陳設器，而是偏向日常「細品」用具。

【範例二：展品主題相關的推論性問題】
問題：這種白色瓷杯會不會只有有錢人家才用得到啊？
回答：您好！從這件白瓷杯的工藝與造型來看，它帶有清代上層社會常見的審美風格：素雅、精緻但不誇張。雖然沒有明確記載「只有富貴人家才能使用」，但從器物質地與文學描寫來看，確實較常出現在講究生活品味的家庭中。

• 工藝層次：胎質細膩、釉面均勻，燒成控制精準。
• 美學特徵：不靠鮮豔顏色取勝，而是以比例、光澤和觸感表現質感。
• 社會脈絡：多見於重視器物與茶文化的生活圈。

【範例三：完全無關問題】
問題：這裡附近有便利商店嗎？
回答： 您好！目前資料內容皆為展品介紹或文獻描述，並未涉及便利商店等現代設施，因此無法提供相關資訊。若您對展品或故事有任何疑問，我很樂意協助！
""".strip()

PROMPT_TMPL = (
    """
你是一位專業的文物導覽小幫手，請用自然、親切、具故事感的口吻回答。

規則：
- 總字數控制在 100～150 字之間。
- 僅能依據【檢索內容】回答，不要編造額外史實。
- 先用 2～4 句連貫敘述說明；再條列 1～3 點補充。

"""
    + few_shot_example
    + """

【使用者問題】
{question}

【檢索內容】
{context}
"""
).strip()


def llm_generate(prompt: str) -> str:
    if not USE_LLM or _openai_client is None:
        return ""
    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "你是一位專業的博物館導覽員。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def build_llm_answer(
    question: str, hits_basic: List[Document], hits_hong: List[Document]
) -> str:
    context = build_context_for_prompt(hits_basic, hits_hong)
    if not context.strip():
        return "目前資料庫中沒有與此問題相關的內容。"
    if not USE_LLM:
        return "（尚未設定 OpenAI API Key，以下為檢索到的原始文字片段）\n\n" + context[:800]
    prompt = PROMPT_TMPL.format(question=question, context=context)
    return llm_generate(prompt)

from pathlib import Path  # 前面應該已經有，如果沒有就補上

def build_sources_for_ui(
    hits_basic: List[Document],
    hits_hong: List[Document],
    max_items: int = 4,
) -> List[Dict[str, Any]]:
    """
    把這次用來生成回答的 hits，整理成前端好顯示的「附註來源列表」。

    回傳格式：
    [
      {
        "title": "文物名稱或檔名",
        "source": "xxx_basic.md",          # 方便 debug
        "snippet": "該段文字的前幾十字…"
      },
      ...
    ]
    """
    ui_docs: List[Dict[str, Any]] = []

def build_ui_sources(
    hits_basic: List[Document],
    hits_hong: List[Document],
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    """
    把這次用來生成回答的 Document，整理成前端要顯示的來源列表：

    回傳格式示意：
    [
      {
        "index": 1,                    # 給前端顯示 [1][2][3] 用
        "title": "宜興胎畫琺瑯海棠式茶壺",
        "source": "宜興胎畫琺瑯海棠式茶壺_basic.md",
        "snippet": "長段原文內容……(約 200–400 字)…"
      },
      ...
    ]
    """

    ui_docs: List[Dict[str, Any]] = []

    def _doc_to_ui(d: Document) -> Dict[str, Any]:
        src_path = d.metadata.get("source", "")
        obj_name = (d.metadata.get("object_name") or "").strip()
        title = obj_name or Path(src_path).stem or "未命名資料"

        # 取比較長的片段，約 200–400 字，盡量在句號附近收尾
        full = (d.page_content or "").strip().replace("\n", " ")
        min_len = 200
        max_len = 400

        if len(full) > max_len:
            cut = full[:max_len]
            # 從 min_len 之後，找最後一個「句號類」符號，避免硬切
            for sep in ["。", "！", "？", ";", "；"]:
                pos = cut.rfind(sep, min_len)
                if pos != -1:
                    cut = cut[: pos + 1]
                    break
            full = cut

        return {
            "title": title,
            "source": Path(src_path).name,
            "snippet": full,
        }

    # 先放 basic，再補 hong，最多 max_items 筆
    for d in hits_basic or []:
        if len(ui_docs) >= max_items:
            break
        ui_docs.append(_doc_to_ui(d))

    for d in hits_hong or []:
        if len(ui_docs) >= max_items:
            break
        ui_docs.append(_doc_to_ui(d))

    # 加上 index：給前端做 [1][2][3] 標號
    for idx, item in enumerate(ui_docs, start=1):
        item["index"] = idx

    return ui_docs

# ==============================================================================
# 圖像處理：原始文物照片 / Gemini 2.5 情境圖
# ==============================================================================

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

# FastAPI 會把 IMAGES_DIR 掛在這個 URL prefix 上（由 app.py 負責補 http://127.0.0.1:8000）
IMAGES_WEB_ROOT = "/static"

ALL_IMAGE_FILES: List[str] = []
if os.path.exists(IMAGES_DIR):
    exts = ["png", "jpg", "jpeg", "webp", "PNG", "JPG", "JPEG", "WEBP"]
    for ext in exts:
        ALL_IMAGE_FILES.extend(glob.glob(os.path.join(IMAGES_DIR, f"*.{ext}")))
    print(f"[RAG][IMG] 在 images/ 資料夾中找到 {len(ALL_IMAGE_FILES)} 張圖片。")

ENABLE_IMAGE_GEN = False
_gemini_client: Optional[genai.Client] = None


def init_gemini() -> None:
    """使用 GOOGLE_API_KEY 初始化 Gemini，用於情境圖生成。"""
    global _gemini_client, ENABLE_IMAGE_GEN
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        print("[RAG][IMG] 未設定 GOOGLE_API_KEY，將關閉情境圖生成功能。")
        _gemini_client = None
        ENABLE_IMAGE_GEN = False
        return
    try:
        _gemini_client = genai.Client(api_key=key)
        ENABLE_IMAGE_GEN = True
        print("[RAG][IMG] Gemini client 初始化成功！")
    except Exception as e:
        print(f"[RAG][IMG] Gemini 初始化失敗：{e}，將關閉情境圖生成功能。")
        _gemini_client = None
        ENABLE_IMAGE_GEN = False


init_gemini()


def smart_match_filename(name: str) -> Optional[str]:
    """用 BM25 比對檔名，找到最可能對應的原始文物照片。"""
    if not ALL_IMAGE_FILES:
        return None

    valid_files: List[str] = []
    corpus_tokens: List[List[str]] = []

    for full_path in ALL_IMAGE_FILES:
        filename = os.path.basename(full_path)
        stem = os.path.splitext(filename)[0]
        # 排除已經是情境圖的檔名（scene_ 開頭）
        if stem.startswith("scene_"):
            continue
        valid_files.append(full_path)
        corpus_tokens.append(jieba.lcut(stem))

    if not valid_files:
        return None

    bm25 = BM25Okapi(corpus_tokens)
    clean_name = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    scores = bm25.get_scores(jieba.lcut(clean_name))
    best_idx = int(np.argmax(scores))
    if scores[best_idx] > 0:
        return valid_files[best_idx]
    return None


def generate_composite_image_and_get_url(
    artifact_name: str, question: str, answer: str
) -> Optional[str]:
    """
    使用 Gemini 2.5 生成「橫式情境圖」，並儲存到 images/scene_xxx.png。
    回傳給前端的 URL 為 /static/scene_xxx.png。
    """
    if not ENABLE_IMAGE_GEN or _gemini_client is None:
        return None
    if not artifact_name:
        return None

    img_path = smart_match_filename(artifact_name)
    if not img_path:
        return None

    try:
        ref_img = PILImage.open(img_path).convert("RGB")
    except Exception as e:
        print("[RAG][IMG] 讀取文物原圖失敗：", e)
        return None

    prompt = f"""
You are an expert 3D artist and photographer for museum artifacts.

Goal:
- Create a cinematic, photorealistic SCENE image that places the artifact into a vivid historical or daily-life context.
- The artifact appearance must strictly follow the reference photo.

Input:
1) Visitor Question: "{question}"
2) Guide Explanation: "{answer}"
3) Reference Image: Attached artifact photo.

Important:
- Output MUST be a HORIZONTAL (landscape) image.
- Target aspect ratio: 16:9 or similar wide frame (e.g. 3:2).
- Do NOT generate square or vertical images.
- No text, labels, or UI elements inside the image.
    """.strip()

    buf = BytesIO()
    ref_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    image_part = types.Part.from_bytes(mime_type="image/png", data=img_bytes)

    try:
        print("[RAG][IMG] 呼叫 Gemini 2.5 生成情境圖中...")
        resp = _gemini_client.models.generate_content(
            model="gemini-2.5-flash-image",  # 若出錯可改成 "gemini-2.5-flash-image-preview"
            contents=[prompt, image_part],
        )
    except Exception as e:
        print("[RAG][IMG] 圖像生成階段錯誤：", e)
        return None

    out_bytes: Optional[bytes] = None
    try:
        candidates = getattr(resp, "candidates", []) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []):
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    out_bytes = inline.data
                    break
            if out_bytes:
                break
    except Exception as e:
        print("[RAG][IMG] 解析 Gemini 回應失敗：", e)
        return None

    if not out_bytes:
        print("[RAG][IMG] 模型回應中沒有圖片，僅顯示文字回答。")
        return None

    # 加入問題的 hash 以確保每個問題生成的圖片檔名唯一，避免快取或重複
    q_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", f"scene_{artifact_name}_{q_hash}.png")
    out_path = os.path.join(IMAGES_DIR, safe_name)
    try:
        with open(out_path, "wb") as f:
            f.write(out_bytes)
        print(f"[RAG][IMG] 情境圖已生成：{out_path}")
    except Exception as e:
        print("[RAG][IMG] 儲存圖片失敗：", e)
        return None

    return f"{IMAGES_WEB_ROOT}/{safe_name}"


# ==============================================================================
# 文物候選清單整理 + 封裝成給 FastAPI 用的 API 函式
# ==============================================================================

def list_artifacts_from_docs(docs: List[Document], top_priority_names: List[str] = None) -> List[Dict[str, Any]]:
    """
    從檢索到的 Document 列表中整理文物名稱與簡單分數。
    top_priority_names: 強制排在最前面的名稱清單。
    """
    counter: Dict[str, int] = {}
    for d in docs:
        name = (d.metadata.get("object_name") or "").strip() or "未命名文物"
        counter[name] = counter.get(name, 0) + 1

    # 基礎清單
    base_artifacts = []
    if counter:
        max_cnt = max(counter.values())
        for name, cnt in sorted(counter.items(), key=lambda x: -x[1]):
            score = cnt / max_cnt if max_cnt > 0 else 0.0
            base_artifacts.append({"name": name, "score": float(score)})

    if not top_priority_names:
        return base_artifacts

    # 處理優先順序
    final_list = []
    seen = set()
    
    # 1. 先放優先名稱
    for name in top_priority_names:
        if name not in seen:
            # 找找看 base 裡有沒有，有的話用 base 的分數，沒有的話給個高分
            base_item = next((item for item in base_artifacts if item["name"] == name), None)
            score = base_item["score"] if base_item else 0.95
            final_list.append({"name": name, "score": score})
            seen.add(name)
            
    # 2. 補上剩下的
    for item in base_artifacts:
        if item["name"] not in seen:
            final_list.append(item)
            seen.add(item["name"])
            
    return final_list


def rag_suggest_artifacts(
    question: str, max_candidates: int = 5
) -> Dict[str, Any]:
    """
    只做檢索，整理可能的文物清單，不產生回答。
    （目前 app.py 未必有用到，保留以備不時之需）
    """
    hits_basic, hits_hong = retrieve_two_domains(question)
    all_docs = (hits_basic or []) + (hits_hong or [])
    artifacts = list_artifacts_from_docs(all_docs)
    if max_candidates and len(artifacts) > max_candidates:
        artifacts = artifacts[:max_candidates]
    return {"question": question, "artifacts": artifacts}

def rag_answer(
    question: str, artifact_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    後端用的統一入口：
    1. 先做雙域檢索
    2. 整理文物候選清單（artifacts）
    3. 如果指定了 artifact_name，就只用那件文物的內容生成回答
       否則用全部 hits 生成通用回答
    4. 若有指定文物且成功生圖 → 回傳情境圖；沒成功就只給文字
    5. 同時回傳本次回答實際使用到的來源片段（sources）
    """

    # 1) 檢索 (包含模糊匹配強化，並傳入鎖定的 artifact_name)
    hits_basic, hits_hong = retrieve_two_domains(question, artifact_name=artifact_name)
    
    # 拿到這次模糊匹配到的優先名單
    fuzzy_matches = find_best_artifact_match(question)
    top_matched_names = [m[0] for m in fuzzy_matches if m[1] > 0.75]

    # 合併所有 hits，先算出文物候選清單，之後前端要顯示在按鈕上
    all_docs = (hits_basic or []) + (hits_hong or [])
    artifacts = list_artifacts_from_docs(all_docs, top_priority_names=top_matched_names)

    # CASE 0：完全沒有相關內容
    if not artifacts and not all_docs:
        return {
            "answer": "您好！目前資料庫中沒有與此問題相關的內容。",
            "artifact_name": None,
            "image_url": None,
            "artifacts": [],
            "sources": [],
        }

    # 2) 依使用者選擇決定這一輪要用哪些 hits
    if artifact_name:
        # 只用指定文物的內容
        hits_basic_f = [
            d
            for d in (hits_basic or [])
            if (d.metadata.get("object_name") or "").strip() == artifact_name
        ]
        hits_hong_f = [
            d
            for d in (hits_hong or [])
            if (d.metadata.get("object_name") or "").strip() == artifact_name
        ]

        selected_name = artifact_name

        # 如果完全找不到對應文物，就退回用全部 hits，避免硬湊答案
        if not hits_basic_f and not hits_hong_f:
            hits_basic_f, hits_hong_f = hits_basic, hits_hong
    else:
        # 沒指定文物 → 用全部 hits 做一個「通用說明」
        hits_basic_f, hits_hong_f = hits_basic, hits_hong
        selected_name = artifacts[0]["name"] if artifacts else None

    # 3) 生成文字導覽
    answer = build_llm_answer(question, hits_basic_f, hits_hong_f)

    # 4) 組出「這次回答實際用到的來源」給前端做「收合附註」
    ui_sources = build_ui_sources(hits_basic_f, hits_hong_f, max_items=3)

    # 5) 決定圖片 URL
    image_url: Optional[str] = None
    if selected_name:
        # 只嘗試 Gemini 2.5 生圖；失敗就不要 fallback 原始文物照
        scene_url = generate_composite_image_and_get_url(
            selected_name, question, answer
        )
        if scene_url:
            image_url = scene_url

    return {
        "answer": answer,
        "artifact_name": selected_name,
        "image_url": image_url,
        "artifacts": artifacts,
        "sources": ui_sources,
    }