import json
import os
from openai import OpenAI
from test_v1 import solve_one, read_jsonl, append_jsonl, write_jsonl, print_lock

# 前述の solve_one, read_jsonl, append_jsonl, write_jsonl が
# 同じファイル内にある、もしくは import できる前提の main 処理です。

def load_preds(path):
    data = {}
    if not os.path.exists(path): return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data[obj["id"]] = obj["prediction"]
    return data

def main():
    # --- 1. 設定 ---
    path1 = "out/preds_test_v1_sorted.jsonl"     
    path2 = "out/preds_test_v2_sorted.jsonl"
    source_data_path = "train/math_level12_easy_train100_student_with_answer_solution.jsonl"
    
    output_consistent_path = "out/preds_compared.jsonl"
    # output_retry_path = "out/preds_retry_v3.jsonl"
    # final_merged_path = "out/preds_compaired.jsonl"
    
    client = OpenAI()

    # --- 2. 一致・不一致の判定 ---
    preds1 = load_preds(path1)
    preds2 = load_preds(path2)
    
    consistent_results = []
    retry_ids = []

    all_ids = sorted(set(preds1.keys()) | set(preds2.keys()))
    for qid in all_ids:
        p1 = preds1.get(qid)
        p2 = preds2.get(qid)
        # 文字列が完全に一致しているかチェック
        if p1 == p2 and p1 is not None:
            consistent_results.append({"id": qid, "prediction": p1})
        else:
            retry_ids.append(qid)

    # 一致分を保存
    write_jsonl(output_consistent_path, consistent_results)
    print(f"一致: {len(consistent_results)} 件 / 不一致: {len(retry_ids)} 件")

    # --- 3. 不一致分の解き直し (gpt-4o-miniを再使用) ---
    retry_results = []
    if retry_ids:
        print(f"ID: {retry_ids} の解き直しを開始します...")
        
        source_rows = {int(r["id"]): r for r in read_jsonl(source_data_path)}
        retry_rows = [source_rows[qid] for qid in retry_ids if qid in source_rows]

        # 【改良ポイント】同じminiでも、思考を促すプロンプトに変更
        retry_system_msg = {
            "role": "system",
            "content": (
                "あなたは数学のスペシャリストです。計算ミスを防ぐため、"
                "まず頭の中でステップバイステップで論理的に解き、"
                "最後に必ず 'FINAL: <答え>' の形式で回答してください。"
                "出力は指定されたJSON形式を厳守してください。"
            ),
        }

        for row in retry_rows:
                    with print_lock:
                        print(f"Retrying ID {row['id']}...")
                    try:
                        # API呼び出し
                        res = solve_one(
                            client=client,
                            model="gpt-4o-mini", 
                            system_msg=retry_system_msg,
                            fewshot_msgs=None,
                            row=row
                        )
                        retry_results.append(res)
                        # 念のためリトライ専用ファイルにも追記保存
                        append_jsonl("out/preds_retry_v3.jsonl", res) 
                    except Exception as e:
                        print(f"ID {row['id']} でエラー: {e}")

    # --- 4. 最終合体 (Merge) して保存 ---
    # 「最初から一致していた結果」と「今解き直した結果」を合わせる
    final_all = consistent_results + retry_results
    
    # ID順に並べ替え（バラバラに解いても綺麗に並びます）
    final_all.sort(key=lambda x: int(x["id"]))
    
    # 【重要】ここで最終的なファイルに書き込みます
    final_output_path = "out/preds_final_fixed.jsonl"
    write_jsonl(final_output_path, final_all)
    
    print("\n=== 工程完了 ===")
    print(f"一致分: {len(consistent_results)}件")
    print(f"解き直し分: {len(retry_results)}件")
    print(f"最終ファイル保存先: {final_output_path}")

if __name__ == "__main__":
    main()