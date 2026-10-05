# Digital Odyssey — 資安教育遊戲 + 麻伊 AI 助手

| 資料夾 | 內容 |
|-------|------|
| `Assets/`、`Packages/`、`ProjectSettings/` | Unity 2D 遊戲專案（Unity 2022.3.37f1） |
| `MAI_Server/` | 遊戲中 AI 小助手「麻伊」的 RAG 後端（Python） |

以下說明 `MAI_Server/` 的內容。

---

## MAI_Server — 麻伊對話後端

遊戲裡的 AI 小助手「麻伊（MAI）」用的本機後端。Unity 端的 `Assets/Scripts/場景管理/AskMAI_API.cs` 把對話歷史送到這裡，伺服器做三件事後再交給本機 LLM 產生回覆：

1. **資安拒答**：資安問題在知識庫裡查不到，就直接回固定的「不確定」句子，不讓模型自己編。
2. **RAG 知識檢索**：從資安知識庫找出相關的 Q&A，當作參考資料給模型。
3. **Few-shot 人設範例**：挑幾組範例對話放在前面，讓小模型維持麻伊的語氣。

```
Unity (AskMAI_API.cs)
   │  POST /ask  {"messages": [...]}
   ▼
Flask app.py ──► ChromaDB 向量檢索（bge-small-zh）
   │          ──► persona_examples.jsonl 範例挑選
   ▼
Ollama (qwen2.5:3b) ──► {"response": "..."} 回傳 Unity
```

---

## 環境需求

- Python 3（套件見 `MAI_Server/requirements.txt`）
- [Ollama](https://ollama.com/)，並先下載模型：

  ```bash
  ollama pull qwen2.5:3b
  ```

- 第一次執行時，`sentence-transformers` 會從 Hugging Face 下載 `BAAI/bge-small-zh-v1.5`，需要網路。

## 安裝與啟動

```bash
cd MAI_Server
pip install -r requirements.txt

# 1. 建立向量庫（chroma_db 不進 git，clone 下來後要先跑一次）
python build_vector_db.py

# 2. 確認 Ollama 已在背景執行（預設 http://127.0.0.1:11434）

# 3. 啟動伺服器
python app.py          # 監聽 http://127.0.0.1:8000
```

Unity 端 `AskMAI_API` 元件的 `localURL` 預設就是 `http://127.0.0.1:8000/ask`，不用另外設定。

### 快速測試

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"messages\": [{\"role\": \"user\", \"content\": \"密碼可以跟同學說嗎\"}]}"
```

---

## 檔案說明

以下路徑都在 `MAI_Server/` 底下。

| 檔案 | 用途 |
|------|------|
| `app.py` | Flask 伺服器，`/ask` 端點與完整處理流程 |
| `build_vector_db.py` | 把 `security_knowledge.json` 轉成向量寫入 ChromaDB（upsert，可重複執行） |
| `knowledge_base/security_knowledge.json` | 資安知識庫，37 筆 Q&A |
| `knowledge_base/chroma_db/` | 向量庫（由腳本產生，已列入 `.gitignore`） |
| `persona_examples.jsonl` | 麻伊人設範例對話，48 組 |
| `eval_threshold.py` | 評估 RAG 距離門檻的腳本 |
| `eval_threshold_result.csv` | 上述評估的逐題結果 |

### 知識庫分類

| 分類 | id 前綴 | 筆數 |
|------|--------|-----|
| 密碼安全 | `pwd_` | 6 |
| 防詐騙 | `scam_` | 6 |
| 病毒與惡意軟體 | `virus_` | 6 |
| 隱私保護 | `privacy_` | 6 |
| 網路禮儀 | `netiquette_` | 6 |
| AI 素養 | `ai_` | 7 |

---

## `/ask` 處理流程

**請求格式**（就是 Unity 端的完整對話歷史）：

```json
{"messages": [{"role": "assistant", "content": "你好！我是 MAI…"}, {"role": "user", "content": "…"}]}
```

**回應格式**：`{"response": "…"}`

伺服器只拿**最後一句 user 訊息**做判斷：

1. **向量檢索**：用 bge-small-zh 把問題轉成向量，在 ChromaDB 取 top-3，只保留 cosine 距離 ≤ `RAG_SCORE_THRESHOLD`（0.35）的結果。
2. **拒答判斷**：如果沒有檢索到結果，而且問題含有 `SECURITY_KEYWORDS` 裡的字（密碼、詐騙、病毒…），直接回 `UNKNOWN_SECURITY_REPLY`，**不呼叫模型**。這是因為 3B 小模型常常不遵守 system prompt 裡「不知道就說不知道」的規則。
3. **挑選範例**：用字元 bigram 的 Jaccard 相似度，從 `persona_examples.jsonl` 挑出最多 4 組相近的範例；不足 2 組時用 `FALLBACK_EXAMPLES`（兩組固定的日常對話）補滿。
4. **組裝 prompt**，依序是：
   - `system`：麻伊人設（`SYSTEM_PROMPT`）
   - `system`：知識庫參考資料（有檢索到才放）
   - `user` / `assistant`：範例對話
   - 原本的對話歷史
5. 送到 Ollama，回傳模型輸出。

範例和知識庫資料只在伺服器端臨時插入，不會回傳給 Unity，也不會累積在對話歷史裡。

### 為什麼範例要用對話輪次放入

範例用 `user` / `assistant` 的訊息放進去，模型會把它當成「自己說過的話」繼續模仿（in-context learning）。這比在 system prompt 裡描述「要自稱麻伊、要說咕」更容易讓小模型照做。範例主要示範的是語氣和格式，所以保底範例刻意選不含知識內容的日常閒聊，避免模型把範例內容套到不相關的問題上。

> 「至少 2 組」和「最多 4 組」是工程上的經驗值，不是實驗得出的最佳值。如果要驗證，可以參考 `eval_threshold.py` 的做法，比較 0 / 1 / 2 / 4 組範例下的人設遵守率。

---

## 可調參數（`app.py`）

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `MODEL_NAME` | `qwen2.5:3b` | Ollama 模型名稱 |
| `RAG_TOP_K` | 3 | 每次檢索取幾筆 |
| `RAG_SCORE_THRESHOLD` | 0.35 | cosine 距離門檻，越小越嚴格 |
| `SECURITY_KEYWORDS` | — | 判斷是不是資安問題的關鍵字 |
| `pick_relevant_examples(k, min_score)` | 4, 0.08 | 範例最多幾組、最低相似度 |
| `FALLBACK_EXAMPLES` | 2 組 | 檢索不到範例時的保底範例 |

## RAG 門檻評估

`eval_threshold.py` 用 46 題有標註的測試題評估檢索門檻：

- `hit`（28 題）：知識庫有涵蓋，只是換句話說
- `uncovered`（10 題）：資安相關但知識庫沒涵蓋
- `chitchat`（8 題）：一般閒聊

```bash
python eval_threshold.py   # 需先跑過 build_vector_db.py
```

依目前的 `eval_threshold_result.csv`：

| 距離門檻 | 正確命中 | 誤判（不該命中卻命中） |
|---------|---------|---------------------|
| 0.300 | 18 / 28 | 0 / 18 |
| **0.350（目前）** | **21 / 28** | **0 / 18** |
| 0.400 | 24 / 28 | 1 / 18 |

選 0.35 是為了在不誤判的前提下，盡量提高命中率。知識庫或 embedding 模型有變動時，記得重跑評估。

---

## 維護

- **新增或修改知識**：編輯 `knowledge_base/security_knowledge.json`（`id` 不可重複），然後重跑 `python build_vector_db.py`，再重啟 `app.py`。
- **新增人設範例**：在 `persona_examples.jsonl` 加一行 `{"user": "…", "assistant": "…"}`，然後重啟 `app.py`（範例在啟動時載入）。
- **新增資安關鍵字**：如果某類資安問題沒被拒答、讓模型自己亂答，就把相關字詞加進 `SECURITY_KEYWORDS`。

## 已知限制

- 拒答判斷靠關鍵字比對。資安問題如果沒有包含任何關鍵字，查不到資料時會交給模型自由回答。
- 檢索只看最後一句，像「那要怎麼辦？」這類需要上下文的追問可能檢索不到。
- `FALLBACK_EXAMPLES` 跟 `persona_examples.jsonl` 前兩行內容相同。只檢索到其中一組時，補上的保底範例可能和它重複。
