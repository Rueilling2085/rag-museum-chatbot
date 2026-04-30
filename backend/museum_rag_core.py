# museum_rag_core.py
# ==============================================================================
# 博物館 RAG 核心模組
#   - 雙域檢索（basic / hongloumeng）
#   - TF-IDF rerank
#   - OpenAI 文字導覽
#   - OpenAI gpt-image-2：生成高保真情境圖（若未啟用則使用原始文物照片）
# ==============================================================================

print("[RAG] museum_rag_core 模組正在載入...")

import os
import requests
import base64
import re
import glob
import json
import hashlib
import mimetypes
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import yaml
import jieba
import numpy as np
import difflib

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from PIL import Image as PILImage
from PIL import ImageFilter
import PIL._util
from io import BytesIO

from openai import OpenAI
from google import genai
from google.genai import types

# Pillow 相容修補（某些版本會缺 is_directory）
if not hasattr(PIL._util, "is_directory"):
    PIL._util.is_directory = lambda path: os.path.isdir(path)

# --- 手動載入 .env 檔案 (不依賴 python-dotenv) ---
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(ENV_PATH):
    print(f"[RAG] 正在從 {ENV_PATH} 載入環境變數...")
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()
else:
    print("[RAG] 警告：未找到 .env 檔案，將使用系統環境變數。")

# ==============================================================================
# 基本參數
# ==============================================================================

OCR_ENABLED = False
EMBED_BATCH = 16
K_VEC = 30
K_BM25 = 50
TOP_N_FINAL = 5

BASIC_CHUNK = 800
BASIC_OVERLAP = 150
HONG_CHUNK = 1000
HONG_OVERLAP = 100

# Markdown 資料庫位置
BASE_DIR = os.path.join(os.path.dirname(__file__), "data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
print(f"[RAG] 資料目錄: {BASE_DIR}")

# ==============================================================================
# OpenAI 初始化（文字導覽）
# ==============================================================================

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
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

print("[RAG] 正在初始化 OpenAI Embedding 模型...")

def _get_embedding_client():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    return OpenAI(api_key=key)

def _embed_texts(texts: List[str], is_query: bool) -> np.ndarray:
    client = _get_embedding_client()
    if not client:
        return np.zeros((len(texts), 1536), dtype="float32")
    
    try:
        # 使用 text-embedding-3-small, 這是目前最便宜且強大的模型
        resp = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return np.array([item.embedding for item in resp.data], dtype="float32")
    except Exception as e:
        print(f"[RAG][EMB] Embedding 失敗: {e}")
        return np.zeros((len(texts), 1536), dtype="float32")

def _embed_docs(docs: List[Document]) -> np.ndarray:
    if not docs:
        return np.zeros((0, 1536), dtype="float32")
    texts = [d.page_content for d in docs]
    # OpenAI 有限制一次輸入的數量，我們分批處理 (Batching)
    batch_size = 100
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vecs = _embed_texts(batch, is_query=False)
        all_vecs.append(vecs)
    
    res = np.vstack(all_vecs)
    return res / (np.linalg.norm(res, axis=1, keepdims=True) + 1e-8)

def _embed_query(q: str) -> np.ndarray:
    v = _embed_texts([q], is_query=True)[0]
    return v / (np.linalg.norm(v) + 1e-8)

print("[RAG][EMB] 建立索引 (使用 OpenAI Cloud)...")
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
    超級強化版的模糊匹配 + 同義詞權重：
    """
    if not query or not GLOBAL_ARTIFACTS:
        return []
    
    # 移除標點，保留空格與中文字
    query_clean = re.sub(r'[^\w]', '', query).strip()
    
    # --- 同義詞群組 ---
    TYPE_GROUPS = [
        {"盒", "匣", "奩", "篋", "盒子", "匣盒"},
        {"錶", "鐘", "懷錶", "手錶"},
        {"碗", "盌", "盅", "碟", "盤", "碗器"},
        {"壺", "甖", "提梁壺", "提梁"},
        {"瓶", "尊", "罍", "罐", "瓶器"},
        {"鐲", "簪", "套", "飾", "玩器"},
        {"扇", "箑"}
    ]
    
    # 識別 query 中出現了哪些群組
    active_groups = []
    for group in TYPE_GROUPS:
        if any(keyword in query_clean for keyword in group):
            active_groups.append(group)

    matches = []
    def normalize_for_match(text):
        return text.replace("樑", "梁").replace("提橡", "提梁").replace("壺", "壶")

    q_norm = normalize_for_match(query_clean)
    q_pinyin = to_pinyin_str(q_norm)

    print(f"[RAG][DEBUG] Cleaning Query: '{query_clean}'")

    for art in GLOBAL_ARTIFACTS:
        art_norm = normalize_for_match(art)
        art_pinyin = to_pinyin_str(art_norm)
        
        score = 0.0
        
        # A. 直接匹配 (極高分)
        if art_norm in q_norm or q_norm in art_norm:
            score = 0.95
        
        # B. 類別匹配 (同義詞權重)
        # 如果 query 提到的「類別」與 artifact 一致，給予基礎分
        category_match = False
        for group in active_groups:
            if any(keyword in art_norm for keyword in group):
                category_match = True
                break
        
        if category_match:
            score = max(score, 0.82) # 類別符合基礎分

        # C. 檢查 artifact 的重要部分是否出現在 query
        core_parts = []
        if len(art_norm) > 4:
            core_parts.append(art_norm[-2:]) # 最後兩字 (通常是器型)
            core_parts.append(art_norm[-3:]) # 最後三字
            core_parts.append(art_norm[:3])  # 最前三字
        
        found_core = False
        for part in core_parts:
            if part in q_norm:
                found_core = True
                break
        
        if found_core:
            score = max(score, 0.85)

        # D. 讀音比對 (pinyin) 比例微調
        if len(q_pinyin) > 4 and len(art_pinyin) > 4:
            match = difflib.SequenceMatcher(None, q_pinyin, art_pinyin).find_longest_match(0, len(q_pinyin), 0, len(art_pinyin))
            if match.size >= 6:
                p_bonus = min(match.size / 20, 0.15)
                score = max(score, 0.80 + p_bonus)

        if score >= threshold:
            matches.append((art, score))

    # 按分數排序
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # 去重並限制為最多 5 個候選項，避免介面過於擁擠
    seen = set()
    final_matches = []
    for a, s in matches:
        if a not in seen:
            final_matches.append((a, s))
            seen.add(a)
            if len(final_matches) >= 5:
                break

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
        # 確保維度一致
        if self.mat.shape[1] != q.shape[0]:
            return []
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


def retrieve_two_domains(q: str, artifact_name: Optional[str] = None) -> Tuple[List[Document], List[Document], Optional[str]]:
    cand_basic = merge_dedup(
        [_vec_basic_ret.search(q), _bm25_basic.search(q)], limit=K_VEC
    )
    cand_hong = merge_dedup(
        [_vec_hong_ret.search(q), _bm25_hong.search(q)], limit=K_VEC
    )

    # --- 文物標的鎖定邏輯 ---
    # 優先權：1. 使用者已決定/鎖定的文物 (artifact_name)
    # 2. 如果沒指定，才用問題 (q) 去模糊比對最佳匹配
    top_art = artifact_name
    if not top_art:
        fuzzy_matches = find_best_artifact_match(q)
        top_art = fuzzy_matches[0][0] if (fuzzy_matches and fuzzy_matches[0][1] > 0.65) else None
    else:
        # 如果雖然鎖定了 A，但問題強烈提到 B (信心 > 0.95 且極具體)，此處亦可考慮切換
        # 但為了防止「瑪瑙」關鍵字誤觸，我們這裡先採「完全尊重傳入參數」策略
        pass

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

    return hits_basic, hits_hong, top_art


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

def clean_markdown_for_ui(text: str) -> str:
    """清理 Markdown 標記，提升展示時的可讀性。"""
    if not text: return ""
    # 移除 YAML Front-matter (如果有殘留)
    text = strip_front_matter(text)

    # --- 新增：移除行政/中繼資料 (例如 統一編號、尺寸等) ---
    # 這些整行移除
    admin_keys = ["英文品名", "統一編號", "分類", "時代", "尺寸", "材質", "品等", "數量"]
    # 這些只移除標籤標題，保留後方說明的內容
    header_keys = ["文物說明", "導覽說明"]
    
    # 移除標題如 ## 基本資訊
    text = re.sub(r'#+\s*基本資訊.*$', '', text, flags=re.MULTILINE)

    for key in admin_keys:
        # 移除包含該關鍵字的整行 (支援 Markdown 格式如 - **統一編號**: xxx)
        text = re.sub(rf'^- \*\*?{key}\*\*?[:：\s].*$', '', text, flags=re.MULTILINE)
        # 移除一般格式
        text = re.sub(rf'{key}[:：\s].*$', '', text, flags=re.MULTILINE)

    for key in header_keys:
        # 僅移除標籤字眼本身，保留後方內容
        text = re.sub(rf'^- \*\*?{key}\*\*?[:：\s]', '', text, flags=re.MULTILINE)
        text = re.sub(rf'{key}[:：\s]', '', text)

    # 移除院藏編號 (如 (故雜3610-3611))
    text = re.sub(r'\(故[\u4e00-\u9fa5]+\s*\d+[-、\d]*\)', '', text)

    # --- 新增：處理轉義符號與特殊編碼 ---
    # 處理雙重逸出的換行符號 (\\n -> \n)
    text = text.replace('\\n', '\n')
    # 移除反斜線轉義 (如 \- \. \! )
    text = re.sub(r'\\([\\!\.\-\(\)\[\]])', r'\1', text)

    # 移除 Markdown 強調標記 (如 ** 或 __)
    text = re.sub(r'\*\*|__', '', text)

    # 移除 Markdown 標題 (#)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # 移除常見元數據標籤 (如有)
    text = re.sub(r'(object_name|domain|type|date|artist):.*', '', text, flags=re.IGNORECASE)
    
    # 移除多餘的空白行 (將多個換行合併為一個)
    text = re.sub(r'\n\s*\n+', '\n', text).strip()
    return text


def llm_summarize_chunk(text: str, question: str) -> str:
    """針對特定資料片段，使用 LLM 生成簡短摘要。"""
    if not USE_LLM or _openai_client is None:
        return ""

    prompt = f"""
請將下方的【博物館資料】與使用者的【問題】結合，總結出「10 個字以內」的精髓點。
規則：
1. 嚴格遵守「10 個字以內」。
2. 直接點出這筆資料對該問題的價值。
3. 口吻專業且極簡。

【問題】：{question}
【博物館資料】：
{text[:800]}
""".strip()

    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[RAG] 摘要生成失敗: {e}")
        return ""


def build_ui_sources(
    hits_basic: List[Document],
    hits_hong: List[Document],
    question: str = "",
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    """
    更新後的附註處理函式：包含 AI 摘要與內容清理。
    """
    ui_docs: List[Dict[str, Any]] = []

    def _doc_to_ui(d: Document) -> Dict[str, Any]:
        src_path = d.metadata.get("source", "")
        obj_name = (d.metadata.get("object_name") or "").strip()
        title = obj_name or Path(src_path).stem or "未命名資料"

        # 1. 清理原文
        cleaned_body = clean_markdown_for_ui(d.page_content)

        # 2. 生成 AI 摘要 (Key Insight)
        summary = llm_summarize_chunk(cleaned_body, question)

        # 3. 截斷原文（確保結束在完整句子，並以適當標點結尾）
        body_snippet = cleaned_body.replace("\n", " ").strip()
        limit = 600
        if len(body_snippet) > limit:
            cut = body_snippet[:limit]
            # 找 limit 前最後一個句號類符號
            last_punc = -1
            # 優先搜尋強結尾（。！？）
            for punc in ["。", "！", "？"]:
                pos = cut.rfind(punc)
                if pos > last_punc:
                    last_punc = pos
            
            # 如果沒有強結尾，再找次要結尾（；，」』）
            if last_punc < limit * 0.5: # 如果強結尾太前面，則多找一點
                for punc in ["；", "，", "」", "』", "”", "）", ")"]:
                    pos = cut.rfind(punc)
                    if pos > last_punc:
                        last_punc = pos
            
            if last_punc != -1:
                body_snippet = cut[:last_punc + 1]
            else:
                body_snippet = cut + "..."
        elif len(body_snippet) > 0:
            # 即使沒超過 limit，也要確保最後一個字是完整的标点，否則加上 ...
            # 檢查最後一個字符是否為標點
            if body_snippet[-1] not in ["。", "！", "？", "；", "，", "」", "』", "”", "）", ")"]:
                body_snippet += "..."

        # 4. 將摘要與原文組合，確保標籤都在行首
        if summary:
            final_snippet = f"【重點摘要】 {summary}\n【參考原文】 {body_snippet}"
        else:
            final_snippet = body_snippet

        return {
            "title": title,
            "source": Path(src_path).name,
            "summary": summary,
            "snippet": final_snippet,
        }

    # 先處理 basic，再補 hong
    for d in hits_basic or []:
        if len(ui_docs) >= max_items: break
        ui_docs.append(_doc_to_ui(d))

    for d in hits_hong or []:
        if len(ui_docs) >= max_items: break
        ui_docs.append(_doc_to_ui(d))

    for idx, item in enumerate(ui_docs, start=1):
        item["index"] = idx

    return ui_docs

# ==============================================================================
# 圖像處理：原始文物照片 / Gemini 2.5 情境圖
# ==============================================================================

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
ORIGINAL_IMAGES_DIR = os.path.join(IMAGES_DIR, "original")
GENERATED_IMAGES_DIR = os.path.join(IMAGES_DIR, "generated")

# 確保資料夾存在
Path(ORIGINAL_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(GENERATED_IMAGES_DIR).mkdir(parents=True, exist_ok=True)

# FastAPI 會把 IMAGES_DIR 掛在這個 URL prefix 上
IMAGES_WEB_ROOT = "/static"

def get_all_original_images() -> List[str]:
    """僅掃描 original 資料夾下的文物照片，並避免在 Windows 上重複計算。"""
    all_files = []
    if os.path.exists(ORIGINAL_IMAGES_DIR):
        # 掃描所有檔案
        for entry in os.scandir(ORIGINAL_IMAGES_DIR):
            if entry.is_file():
                ext = entry.name.split('.')[-1].lower()
                if ext in ["png", "jpg", "jpeg", "webp"]:
                    all_files.append(os.path.abspath(entry.path))
    
    # 透過 set 去重 (處理 Windows glob 不分大小寫的問題)
    return sorted(list(set(all_files)))

ALL_IMAGE_FILES = get_all_original_images()
print(f"[RAG][IMG] 在 original/ 資料夾中找到 {len(ALL_IMAGE_FILES)} 張原始文物圖片。")

ENABLE_IMAGE_GEN = False
_gemini_client: Optional[genai.Client] = None

def init_gemini() -> None:
    """使用 GOOGLE_API_KEY 初始化 Gemini，用於圖像生成。"""
    global _gemini_client, ENABLE_IMAGE_GEN
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        print("[RAG][IMG] 未設定 GOOGLE_API_KEY，將停用圖像生成功能。")
        _gemini_client = None
        ENABLE_IMAGE_GEN = False
        return
    try:
        _gemini_client = genai.Client(api_key=key)
        ENABLE_IMAGE_GEN = True
        print("[RAG][IMG] Gemini client 初始化成功！")
    except Exception as e:
        print(f"[RAG][IMG] Gemini 初始化失敗: {e}，將停用圖像生成功能。")
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
    使用 Gemini 2.5 進行情境合成。
    以原圖為參考，直接輸出橫向情境圖。
    """
    if not ENABLE_IMAGE_GEN or _gemini_client is None:
        return None
    if not artifact_name:
        return None

    img_path = smart_match_filename(artifact_name)
    if not img_path:
        print(f"[RAG][IMG] 找不到文物 {artifact_name} 的原始圖片。")
        return None

    try:
        ref_img = PILImage.open(img_path).convert("RGB")
    except Exception as e:
        print("[RAG][IMG] 讀取原始圖片失敗:", e)
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
- Output MUST be a VERTICAL (portrait) image.
- Target aspect ratio: 9:16 or similar vertical frame (e.g. 3:4).
- Do NOT generate square or horizontal images.
- No text, labels, or UI elements inside the image.
    """.strip()

    buf = BytesIO()
    ref_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    image_part = types.Part.from_bytes(mime_type="image/png", data=img_bytes)

    try:
        print("[RAG][IMG] 呼叫 Gemini 2.5 圖像生成中...")
        resp = _gemini_client.models.generate_content(
            model="gemini-2.5-flash-image",  
            contents=[prompt, image_part],
        )
    except Exception as e:
        print("[RAG][IMG] 圖像生成請求失敗:", e)
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
        print("[RAG][IMG] 解析 Gemini 返回資料失敗:", e)
        return None

    if not out_bytes:
        print("[RAG][IMG] 圖像生成未返回有效資料，請確認提示詞是否違規。")
        return None

    # 以問題 hash 作為檔名後綴確保唯一性，避免覆蓋舊圖
    q_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", f"scene_{artifact_name}_{q_hash}.png")
    out_path = os.path.join(GENERATED_IMAGES_DIR, safe_name)
    try:
        with open(out_path, "wb") as f:
            f.write(out_bytes)
        print(f"[RAG][IMG] 圖像已生成: {out_path}")
    except Exception as e:
        print("[RAG][IMG] 儲存圖片失敗:", e)
        return None

    return f"{IMAGES_WEB_ROOT}/generated/{safe_name}"

def _generate_blueprint_fallback(artifact_name, question, answer):
    """
    合成引擎失敗時的回退機制 (使用原本的藍圖模式)
    """
    # ... (此處保留之前的 generate_composite_image_and_get_url 邏輯作為備援)
    # 為了簡潔，這裡省略重複代碼，實際實作時可將其更名為此函數
    return None # 暫時返回 None


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
        return base_artifacts[:5]

    # 處理優先順序
    final_list = []
    seen = set()
    
    # 1. 先放優先名稱
    for name in top_priority_names:
        if name not in seen:
            # 找找看 base 裡有沒有，有的話用 base 的分數，沒有的話給個高分
            base_item = next((item for item in base_artifacts if item["name"] == name), None)
            score = base_item["score"] if base_item else 0.95
            final_list.append({"name": name, "score": float(score)})
            seen.add(name)
            if len(final_list) >= 5:
                break
            
    # 2. 補上剩下的
    for item in base_artifacts:
        if len(final_list) >= 5:
            break
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
    hits_basic, hits_hong, _ = retrieve_two_domains(question)
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
    hits_basic, hits_hong, effective_name = retrieve_two_domains(question, artifact_name=artifact_name)
    
    # 只有在「沒有指定文物」的情況下，才執行獨立的模糊匹配來決定 UI 推薦按鈕
    top_matched_names = []
    if not artifact_name:
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

    # 2) 決定這一輪要用哪些 hits 與使用的文物名稱
    # 優先使用 detect 出來的文物
    selected_name = effective_name
    
    if selected_name:
        # 【強制鎖定模式】只允許使用該文物的內容
        hits_basic_f = [
            d
            for d in (hits_basic or [])
            if (d.metadata.get("object_name") or "").strip() == selected_name
        ]
        hits_hong_f = [
            d
            for d in (hits_hong or [])
            if (d.metadata.get("object_name") or "").strip() == selected_name
        ]
        # 注意：如果鎖定模式下 hits 為空，不再 fallback 到全部，以防資料混淆
    else:
        # 沒特定文物 → 用全部 hits 做一個「通用說明」
        hits_basic_f, hits_hong_f = hits_basic, hits_hong
        # 雖然沒有強烈匹配，但我們可以從候選中推斷一個最可能的作為名稱回傳
        selected_name = artifacts[0]["name"] if artifacts else None

    # 3) 生成文字導覽
    answer = build_llm_answer(question, hits_basic_f, hits_hong_f)

    # 4) 組出「這次回答實際用到的來源」給前端
    ui_sources = build_ui_sources(hits_basic_f, hits_hong_f, question=question, max_items=3)

    # 5) 決定圖片 URL
    image_url: Optional[str] = None
    if selected_name:
        # 嘗試使用 Gemini 2.5 參考原圖生成情境圖；失敗則 fallback 原始文物照
        scene_url = generate_composite_image_and_get_url(
            selected_name, question, answer
        )
        if scene_url:
            image_url = scene_url
        else:
            # FALLBACK: 如果生圖失敗，回退使用原始文物照
            print(f"[RAG][IMG] 情境圖生成失敗，回退至原始文物照片: {selected_name}")
            matched_file = smart_match_filename(selected_name)
            if matched_file:
                # 僅取檔名，因為 IMAGES_WEB_ROOT (/static) 下有 original 資料夾
                filename = os.path.basename(matched_file)
                image_url = f"{IMAGES_WEB_ROOT}/original/{filename}"

    return {
        "answer": answer,
        "artifact_name": selected_name,
        "image_url": image_url,
        "artifacts": artifacts,
        "sources": ui_sources,
    }