import os
import json
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# --- 環境変数の設定 ---
MODEL = "gpt-4o-mini"
SEED = 42
MAX_RETRIES = 10
MAX_WORKERS = 15

TRAIN_PATH = "train/math_level12_easy_train100_student_with_answer_solution.jsonl"

# CV settings
N_SPLITS = 5
PROMPT_TRAIN_K = 20
TEMPERATURE = 0.0

print_lock = Lock() # マルチスレッド用のロック


# --- IO ---
# JSONL を読む関数
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# --- Evaluation utils ---
# 文字列正規化を行う関数
def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if s.startswith("FINAL:"):
        s = s[len("FINAL:"):].strip()
    return s

# 正答率を求める関数
def accuracy(rows: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> float:
    gold_by_id = {r["id"]: r["answer"] for r in rows}
    correct = 0
    total = 0
    for p in preds:
        qid = p["id"]
        pred = normalize_answer(p.get("prediction", ""))
        gold = normalize_answer(gold_by_id.get(qid, ""))
        total += 1
        if pred == gold:
            correct += 1
    return correct / total if total else 0.0


# --- K-fold split (independent) ---
# K-fold 分割 を行う関数
def kfold_split(rows: List[Dict[str, Any]], n_splits: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """
    rows の index を n_splits に分割し、
    (train_indices, val_indices) を fold ごとに返す。
    """
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)

    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for i, j in enumerate(idx):
        folds[i % n_splits].append(j)

    out: List[Tuple[List[int], List[int]]] = []
    all_idx_set = set(idx)
    for f in range(n_splits):
        val_idx = folds[f]
        train_idx = list(all_idx_set - set(val_idx))
        out.append((train_idx, val_idx))
    return out

# 先頭 k 個を取る汎用関数
def take_k(rows: List[Dict[str, Any]], k: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    tmp = rows[:]
    rng.shuffle(tmp)
    return tmp[: min(k, len(tmp))]


# --- OpenAI call (structured output) ---
# OpenAI API 呼び出しを行う関数
def call_model_json(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_retries: int = 10,
) -> Dict[str, Any]:
    schema = {
        "name": "math_prediction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "prediction": {"type": "string"},
            },
            "required": ["id", "prediction"],
            "additionalProperties": False,
        },
    }

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": schema},
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            sleep_s = 2 ** attempt
            with print_lock:
                print(f"[warn] {e} -> retry in {sleep_s}s")
            time.sleep(sleep_s)

    raise RuntimeError("unreachable")


# --- PromptModel (independent) ---
# few-shot の作り方を定義している
@dataclass
class PromptModel:
    """
    「訓練データの一部」から、推論に使う messages（system + fewshot）を構築する
    ここを差し替えることで "プロンプトモデル" を比較できる。
    """
    fewshot_k: int
    seed: int

    def build_prompt(self, prompt_train_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        """
        return: (system_msg, fewshot_msgs)
        """
        system_msg = {
            "role": "system",
            "content": (
                "あなたは数学問題を解くアシスタントです。"
                "与えられた問題に対して、必ずJSONで {id:int, prediction:str} を返してください。"
                "prediction は必ず 'FINAL: <答え>' の1行だけにしてください。"
                "途中式や解説は prediction に含めないでください。"
            ),
        }

        picked = take_k(prompt_train_rows, self.fewshot_k, self.seed)

        fewshot_msgs: List[Dict[str, str]] = []
        for ex in picked:
            fewshot_msgs.append(
                {
                    "role": "user",
                    "content": f"問題:\n{ex['problem']}\n\n出力は JSON で、prediction は必ず 'FINAL: <答え>' だけにして。",
                }
            )
            fewshot_msgs.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"id": ex["id"], "prediction": f"FINAL: {ex['answer']}"},
                        ensure_ascii=False,
                    ),
                }
            )
        return system_msg, fewshot_msgs


# --- Inference on a dataset given a prompt ---
# 問題を解くための関数
def solve_one(
    client: OpenAI,
    model: str,
    system_msg: Dict[str, str],
    fewshot_msgs: List[Dict[str, str]],
    row: Dict[str, Any],
) -> Dict[str, Any]:
    qid = row["id"]
    problem = row["problem"]

    user_msg = {
        "role": "user",
        "content": (
            f"次の問題を解いて。\n\n"
            f"problem:\n{problem}\n\n"
            f"id は {qid}。\n"
            f"JSONで出力して。prediction は 'FINAL: <答え>' だけ。"
        ),
    }

    messages = [system_msg] + fewshot_msgs + [user_msg]
    pred_obj = call_model_json(
        client=client,
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_retries=MAX_RETRIES,
    )

    # id固定＆フォーマット矯正
    pred_obj["id"] = qid
    p = str(pred_obj.get("prediction", "")).strip()
    if not p.startswith("FINAL:"):
        p = "FINAL: " + p
    pred_obj["prediction"] = p
    return pred_obj

# データセットを解くための関数 (並列化、スレッド数は MAX_WORKERS で指定)
def run_inference(
    client: OpenAI,
    model: str,
    system_msg: Dict[str, str],
    fewshot_msgs: List[Dict[str, str]],
    rows: List[Dict[str, Any]],
    desc: str = "",
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    done = 0
    total = len(rows)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(solve_one, client, model, system_msg, fewshot_msgs, row)
            for row in rows
        ]
        for fut in as_completed(futures):
            pred = fut.result()
            outputs.append(pred)
            done += 1
            if done % 10 == 0:
                with print_lock:
                    print(f"{desc} done {done}/{total}")

    outputs.sort(key=lambda x: x["id"])
    return outputs


# --- Cross-validation runner ---
# 交差検証の本体部分
def cross_validate_prompt_model(
    client: OpenAI,
    model: str,
    rows: List[Dict[str, Any]],
    prompt_model: PromptModel,
    n_splits: int,
    seed: int,
) -> Dict[str, Any]:
    splits = kfold_split(rows, n_splits=n_splits, seed=seed)

    fold_scores: List[float] = []
    for fold_i, (train_idx, val_idx) in enumerate(splits, start=1):
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        # プロンプトの構築
        prompt_train_rows = take_k(train_rows, prompt_model.fewshot_k, seed + fold_i)
        system_msg, fewshot_msgs = prompt_model.build_prompt(prompt_train_rows)

        # 精度の測定
        preds = run_inference(
            client, model, system_msg, fewshot_msgs, val_rows, desc=f"[fold {fold_i}/{n_splits}]"
        )
        fold_acc = accuracy(val_rows, preds)
        fold_scores.append(fold_acc)

        with print_lock:
            print(f"[fold {fold_i}] acc = {fold_acc:.4f} (val={len(val_rows)})")

    mean_acc = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0
    return {
        "model": model,
        "n_splits": n_splits,
        "fewshot_k": prompt_model.fewshot_k,
        "fold_scores": fold_scores,
        "mean_score": mean_acc,
    }


# --- main ---
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が環境変数に設定されていません。")

    client = OpenAI()
    rows = read_jsonl(TRAIN_PATH)

    prompt_model = PromptModel(
        fewshot_k=PROMPT_TRAIN_K,
        seed=SEED,
    )

    result = cross_validate_prompt_model(
        client=client,
        model=MODEL,
        rows=rows,
        prompt_model=prompt_model,
        n_splits=N_SPLITS,
        seed=SEED,
    )

    print("\n=== CV RESULT ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
