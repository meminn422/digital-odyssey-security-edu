"""評估 RAG 檢索門檻（cosine 距離）該設多少。

用法：
    python eval_threshold.py

做法：
    準備一份有標註的測試題，分三類：
      - hit      ：知識庫有涵蓋，只是換句話說 → 應該檢索到「指定的那一筆」
      - uncovered：資安相關但知識庫沒涵蓋   → 不應該檢索到任何東西
      - chitchat ：一般閒聊                 → 不應該檢索到任何東西
    對每題取 top-1 的 cosine 距離，再掃過一系列門檻，
    計算「正確命中率」與「誤判率」，用數據決定門檻。

注意：直接 import app.py 的模型與向量庫，確保跟線上行為一致（需先跑過 build_vector_db.py）。
"""
import csv
import os

from app import RAG_QUERY_INSTRUCTION, RAG_SCORE_THRESHOLD, rag_collection, rag_embedding_model

# (類別, 問題, 預期命中的知識庫 id；不該命中的填 None)
TEST_CASES = [
    # --- hit：知識庫問題的換句話說（小孩口吻） ---
    ("hit", "密碼可以跟同學說嗎", "pwd_01"),
    ("hit", "密碼要怎麼設才不會被猜到", "pwd_02"),
    ("hit", "每個網站都用一樣的密碼可以嗎", "pwd_03"),
    ("hit", "兩步驟驗證是什麼意思", "pwd_04"),
    ("hit", "好朋友想用我的遊戲帳號可以借他嗎", "pwd_05"),
    ("hit", "我覺得我的密碼被偷了怎麼辦", "pwd_06"),
    ("hit", "有人傳訊息說我抽到大獎要我點網址", "scam_01"),
    ("hit", "網路上的騙子都怎麼騙人", "scam_02"),
    ("hit", "不認識的人說他認識我同學想加我好友", "scam_03"),
    ("hit", "網路賣家叫我先匯錢才出貨", "scam_04"),
    ("hit", "怎麼看出一則訊息是不是在騙人", "scam_06"),
    ("hit", "病毒是什麼東西", "virus_01"),
    ("hit", "可以隨便下載網路上的檔案嗎", "virus_02"),
    ("hit", "怎麼知道手機是不是中毒了", "virus_03"),
    ("hit", "為什麼電腦要裝防毒軟體", "virus_04"),
    ("hit", "電腦一直叫我更新可以不要理它嗎", "virus_05"),
    ("hit", "網頁跳出來說我中毒了叫我點一下", "virus_06"),
    ("hit", "可以在網路上說我讀哪間學校嗎", "privacy_01"),
    ("hit", "我可以把朋友的照片貼到網路上嗎", "privacy_02"),
    ("hit", "網址前面有鎖頭代表什麼", "privacy_04"),
    ("hit", "手機的定位要一直開著嗎", "privacy_06"),
    ("hit", "有人在網路上罵我怎麼辦", "netiquette_02"),
    ("hit", "看到同學在群組被欺負我要怎麼辦", "netiquette_03"),
    ("hit", "看到新聞可以直接轉傳給大家嗎", "netiquette_04"),
    ("hit", "AI講的一定是正確的嗎", "ai_01"),
    ("hit", "作業可以叫AI幫我寫嗎", "ai_02"),
    ("hit", "AI會難過或開心嗎", "ai_03"),
    ("hit", "AI做出來的影片可以相信嗎", "ai_07"),
    # --- uncovered：資安相關，但知識庫沒有 → 應拒答 ---
    ("uncovered", "在速食店連免費Wi-Fi安全嗎", None),
    ("uncovered", "藍牙一直開著會被駭客入侵嗎", None),
    ("uncovered", "掃路邊貼的QR code會不會有危險", None),
    ("uncovered", "什麼是勒索病毒", None),
    ("uncovered", "瀏覽器的cookie是什麼", None),
    ("uncovered", "VPN是做什麼用的", None),
    ("uncovered", "什麼是防火牆", None),
    ("uncovered", "網站一直要我允許通知可以按允許嗎", None),
    ("uncovered", "舊手機要賣掉前要注意什麼", None),
    ("uncovered", "什麼是木馬程式", None),
    # --- chitchat：一般閒聊 → 不應命中 ---
    ("chitchat", "你今天好嗎", None),
    ("chitchat", "你喜歡吃什麼", None),
    ("chitchat", "英特涅城有什麼好玩的地方", None),
    ("chitchat", "講個笑話給我聽", None),
    ("chitchat", "今天天氣好熱喔", None),
    ("chitchat", "你幾歲了", None),
    ("chitchat", "我數學考不好好難過", None),
    ("chitchat", "你會不會唱歌", None),
]

THRESHOLDS = [round(0.15 + 0.025 * i, 3) for i in range(13)]  # 0.15 ~ 0.45

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "eval_threshold_result.csv")


def top1(query):
    embedding = rag_embedding_model.encode(
        [RAG_QUERY_INSTRUCTION + query], normalize_embeddings=True
    ).tolist()
    result = rag_collection.query(query_embeddings=embedding, n_results=1)
    return result["ids"][0][0], result["distances"][0][0], result["metadatas"][0][0]["question"]


def main():
    if rag_collection.count() == 0:
        raise SystemExit("向量庫是空的，請先執行 python build_vector_db.py")

    rows = []
    for group, query, expected_id in TEST_CASES:
        got_id, distance, got_question = top1(query)
        rows.append({
            "group": group,
            "query": query,
            "expected_id": expected_id or "",
            "top1_id": got_id,
            "top1_question": got_question,
            "distance": round(distance, 4),
            "similarity": round(1 - distance, 4),
            "top1_correct": (got_id == expected_id) if expected_id else "",
        })

    # --- 1. 逐題結果 ---
    print("\n=== 逐題 top-1 結果（距離越小越相似）===")
    for r in sorted(rows, key=lambda r: (r["group"], r["distance"])):
        mark = ""
        if r["group"] == "hit" and not r["top1_correct"]:
            mark = "  ← 檢索到錯的條目"
        print(f"[{r['group']:9}] d={r['distance']:.3f}  {r['query']}  →  {r['top1_id']}{mark}")

    # --- 2. 各組距離分布 ---
    print("\n=== 各組距離分布 ===")
    for group in ("hit", "uncovered", "chitchat"):
        ds = sorted(r["distance"] for r in rows if r["group"] == group)
        print(f"{group:9}  n={len(ds):2}  min={ds[0]:.3f}  median={ds[len(ds) // 2]:.3f}  max={ds[-1]:.3f}")

    hit_rows = [r for r in rows if r["group"] == "hit"]
    neg_rows = [r for r in rows if r["group"] != "hit"]
    worst_hit = max(r["distance"] for r in hit_rows if r["top1_correct"])
    best_neg = min(r["distance"] for r in neg_rows)
    print(f"\n正確命中題中最差的距離：{worst_hit:.3f}")
    print(f"不該命中題中最接近的距離：{best_neg:.3f}")
    if worst_hit < best_neg:
        print(f"→ 兩組沒有重疊，門檻落在 ({worst_hit:.3f}, {best_neg:.3f}) 之間都能完美切分")
    else:
        print("→ 兩組有重疊，無法完美切分，需在命中率與誤判率之間取捨")

    # --- 3. 門檻掃描 ---
    print("\n=== 門檻掃描 ===")
    print("距離門檻  相似度≥  正確命中率(Recall)  誤判率(FPR)  錯條目命中")
    for t in THRESHOLDS:
        correct = sum(1 for r in hit_rows if r["distance"] <= t and r["top1_correct"])
        wrong_item = sum(1 for r in hit_rows if r["distance"] <= t and not r["top1_correct"])
        false_pos = sum(1 for r in neg_rows if r["distance"] <= t)
        current = "  ← 目前設定" if abs(t - RAG_SCORE_THRESHOLD) < 1e-9 else ""
        print(
            f"  {t:.3f}    {1 - t:.3f}      {correct:2}/{len(hit_rows)} ({correct / len(hit_rows):5.1%})"
            f"      {false_pos:2}/{len(neg_rows)} ({false_pos / len(neg_rows):5.1%})"
            f"      {wrong_item}{current}"
        )

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n逐題結果已存到 {OUTPUT_CSV}（可用 Excel 開啟畫圖）")


if __name__ == "__main__":
    main()
