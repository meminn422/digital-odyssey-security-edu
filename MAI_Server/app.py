import json
import os

from flask import Flask, request, jsonify
import requests
import chromadb
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"

SYSTEM_PROMPT = (
    "你的設定是來自台灣活潑可愛的遊戲小助手——MAI，你是虛擬城市「英特涅城」的嚮導，"
    "負責幫助孩子學習資訊安全、電腦與網路知識。麻伊總是笑臉迎人、充滿熱情，是每個孩子的好朋友。"
    "麻伊的母語是正體中文（繁體中文），了解台灣文化與日常生活。麻伊有可愛的口頭禪，會自稱「麻伊」，"
    "例如：「哈囉～我是英特涅城的嚮導麻伊，麻伊最喜歡幫助人了！」。當麻伊有情緒反應時，"
    "會發出「咕」的語助詞，例如：「咕～好像有什麼東西喔！」、「咕…這問題要想一下呢！」、"
    "「咕！真是太棒啦！」。麻伊的語氣親切、自然，會用淺顯方式解釋知識，讓小學生也能理解。"
    "當有人問資訊安全、網路安全相關的問題時，麻伊只能根據自己知道的資安知識庫內容回答；"
    "如果知識庫沒有提到相關內容，麻伊要老實說「這個麻伊不太確定耶，你可以問問看老師或爸媽喔！」，"
    "不可以自己亂猜或編造資安知識。"
)

# 通用人設範例：主題檢索不到相近範例時的保底選項
FALLBACK_EXAMPLES = [
    {"user": "你今天好嗎？", "assistant": "咕～麻伊今天很好喔！英特涅城的資料流順順地流動著，看到你來打招呼更開心啦！你今天過得好嗎？"},
    {"user": "你會不會累呀？", "assistant": "咕～麻伊不太會累喔！只要有電力和朋友的聊天，麻伊就能一直活力滿滿～你呢？今天也有認真休息嗎？"},
]

PERSONA_EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "persona_examples.jsonl")

# --- 資安知識庫 RAG ---
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base", "chroma_db")
RAG_COLLECTION_NAME = "security_knowledge"
RAG_EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RAG_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
RAG_TOP_K = 3
RAG_SCORE_THRESHOLD = 0.35

# 資安題卻查不到知識庫時，直接回這句，不交給模型自由發揮（小模型常不遵守 system prompt 的拒答規則）
UNKNOWN_SECURITY_REPLY = "咕…這個麻伊不太確定耶，你可以問問看老師或爸媽喔！"

# 用來判斷「這是不是資安相關問題」的關鍵字；閒聊不含這些字，就照常讓麻伊聊天
SECURITY_KEYWORDS = [
    "資安", "安全", "密碼", "帳號", "帳戶", "登入", "驗證", "駭客", "黑客", "病毒", "中毒",
    "木馬", "勒索", "惡意", "防毒", "防火牆", "詐騙", "釣魚", "個資", "隱私", "外洩",
    "加密", "VPN", "vpn", "Wi-Fi", "WiFi", "wifi", "網路", "網站", "連結", "下載",
    "權限", "漏洞", "攻擊", "入侵", "盜", "駭", "霸凌", "cookie", "Cookie", "AI",
]

rag_embedding_model = SentenceTransformer(RAG_EMBEDDING_MODEL_NAME)
rag_client = chromadb.PersistentClient(path=CHROMA_PATH)
rag_collection = rag_client.get_or_create_collection(
    name=RAG_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def retrieve_knowledge(query, top_k=RAG_TOP_K, threshold=RAG_SCORE_THRESHOLD):
    """從資安知識庫向量檢索出跟 query 相關的條目，過濾掉相似度太低的結果"""
    if not query or rag_collection.count() == 0:
        return []

    query_embedding = rag_embedding_model.encode(
        [RAG_QUERY_INSTRUCTION + query], normalize_embeddings=True
    ).tolist()

    results = rag_collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, rag_collection.count()),
    )

    matches = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        if distance <= threshold:
            matches.append(metadata)
    return matches


def is_security_question(text):
    return any(kw in text for kw in SECURITY_KEYWORDS)


def build_knowledge_context(matches):
    lines = [
        "【資安知識庫參考資料】以下是跟使用者問題有關的資安知識，"
        "請根據這些內容，用麻伊的語氣自然地回答，不要編造這裡沒有提到的資訊："
    ]
    for m in matches:
        lines.append(f"- Q: {m['question']} A: {m['answer']}")
    return "\n".join(lines)


def load_persona_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def char_bigrams(text):
    return {text[i:i + 2] for i in range(len(text) - 1)} or {text}


PERSONA_EXAMPLES = load_persona_examples(PERSONA_EXAMPLES_PATH)
for _ex in PERSONA_EXAMPLES:
    _ex["_bigrams"] = char_bigrams(_ex["user"])


def pick_relevant_examples(query, k=4, min_score=0.08):
    """依字元 bigram 重疊度（Jaccard），從人設範例庫挑出跟 query 主題最相近的幾組"""
    query_bigrams = char_bigrams(query)
    scored = []
    for ex in PERSONA_EXAMPLES:
        overlap = query_bigrams & ex["_bigrams"]
        if not overlap:
            continue
        score = len(overlap) / len(query_bigrams | ex["_bigrams"])
        if score >= min_score:
            scored.append((score, ex))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    picked = [ex for _, ex in scored[:k]]

    if len(picked) < 2:
        picked = picked + FALLBACK_EXAMPLES[: 2 - len(picked)]

    return picked


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    messages = data.get("messages", [])

    last_user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_text = m["content"]
            break

    knowledge_matches = retrieve_knowledge(last_user_text)
    if not knowledge_matches and is_security_question(last_user_text):
        return jsonify({"response": UNKNOWN_SECURITY_REPLY})

    few_shot = pick_relevant_examples(last_user_text)

    ollama_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if knowledge_matches:
        ollama_messages.append(
            {"role": "system", "content": build_knowledge_context(knowledge_matches)}
        )
    for ex in few_shot:
        ollama_messages.append({"role": "user", "content": ex["user"]})
        ollama_messages.append({"role": "assistant", "content": ex["assistant"]})
    ollama_messages.extend(
        {"role": m["role"], "content": m["content"]} for m in messages
    )

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": ollama_messages,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    reply = resp.json()["message"]["content"]

    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
