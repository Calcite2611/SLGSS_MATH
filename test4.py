import os
import json
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import re
import math

# --- 環境変数の設定 ---
MODEL = "gpt-4o-mini"
SEED = 42
MAX_RETRIES = 10
MAX_WORKERS = 15

TRAIN_PATH = "train/math_level12_easy_train100_student_with_answer_solution.jsonl"

# CV settings
N_SPLITS = 5
PROMPT_TRAIN_K = 40
TEMPERATURE = 0.5

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
import re
import math
from typing import Optional, Tuple

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

def normalize_answer(s: str) -> str:
    """
    表記ゆれをかなり潰す（数学用）。
    目的: 文字列完全一致の事故を減らす。
    """
    if s is None:
        return ""
    s = str(s).strip()

    # "FINAL:" プレフィックス除去
    if s.upper().startswith("FINAL:"):
        s = s.split(":", 1)[1].strip()

    # よくある余計な装飾の除去
    s = s.strip()
    s = s.replace("−", "-")  # U+2212 を ASCII '-' に
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace(",", "")   # 2,000 -> 2000

    # LaTeXっぽい \frac{a}{b} を a/b に寄せる（簡易）
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)

    # 両端の括弧などを剥がす（(2) -> 2, [2] -> 2）
    s = s.strip()
    s = re.sub(r"^[\(\[\{]\s*", "", s)
    s = re.sub(r"\s*[\)\]\}]$", "", s)

    # 空白を詰める（" 1 / 2 " -> "1/2"）
    s = re.sub(r"\s+", "", s)

    return s


def try_parse_number(s: str) -> Optional[float]:
    """
    文字列を「数値」として解釈できるなら float を返す。
    - 整数/小数/指数表記
    - a/b 形式（分数）
    """
    if not s:
        return None

    # 分数
    if "/" in s and s.count("/") == 1:
        a, b = s.split("/")
        if a and b:
            na = try_parse_number(a)
            nb = try_parse_number(b)
            if na is not None and nb is not None and nb != 0:
                return na / nb
        return None

    # 純粋な数値（±, 小数, 指数）
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s):
        try:
            return float(s)
        except:
            return None

    return None


def accuracy(rows, preds, *, rel_tol=1e-9, abs_tol=1e-9) -> float:
    """
    1) normalize した文字列が一致 → 正解
    2) それでも違うなら、数値として解釈できる場合は数値比較（許容誤差つき）
    """
    gold_by_id = {int(r["id"]): r["answer"] for r in rows}

    correct = 0
    total = 0

    for p in preds:
        qid = int(p["id"])
        pred_raw = p.get("prediction", "")
        gold_raw = gold_by_id.get(qid, "")

        pred = normalize_answer(pred_raw)
        gold = normalize_answer(gold_raw)

        total += 1

        if pred == gold:
            correct += 1
            continue

        # 数値比較（両方数値にできるなら）
        pn = try_parse_number(pred)
        gn = try_parse_number(gold)
        if pn is not None and gn is not None:
            if math.isclose(pn, gn, rel_tol=rel_tol, abs_tol=abs_tol):
                correct += 1
                continue

    return correct / total if total else 0.0




# --- K-fold split (independent) ---
# K-fold 分割 を行う関数
# --- K-fold split (stratified by 'type') ---
def kfold_split(rows: List[Dict[str, Any]], n_splits: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """
    rows を 'type' で層化して n_splits に分割し、
    (train_indices, val_indices) を fold ごとに返す。

    仕様:
    - 各 type ごとに index をシャッフル
    - 各 type の件数を n_splits にできるだけ均等に割り振る
      (例: 23件なら 5,5,5,4,4 のように)
    - 余り(＋1)を付ける fold は type ごとに seed でランダム化して偏りを防ぐ
    """
    rng = random.Random(seed)

    # type -> indices
    by_type: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        t = str(r.get("type", "UNKNOWN"))
        by_type.setdefault(t, []).append(i)

    # foldごとの val index を作る
    val_folds: List[List[int]] = [[] for _ in range(n_splits)]

    # typeごとに割り当て
    for t, idxs in by_type.items():
        rng.shuffle(idxs)
        m = len(idxs)

        base = m // n_splits
        rem = m % n_splits  # rem 個の fold に +1 する

        # +1 を付ける fold を type ごとにランダム化
        fold_order = list(range(n_splits))
        rng.shuffle(fold_order)

        quotas = [base] * n_splits
        for k in range(rem):
            quotas[fold_order[k]] += 1

        # quotas に従って切り出して各 fold に追加
        pos = 0
        for f in range(n_splits):
            q = quotas[f]
            if q > 0:
                val_folds[f].extend(idxs[pos:pos + q])
                pos += q

        assert pos == m, f"internal error: type={t} pos={pos} m={m}"

    # fold 内の並びもシャッフルしておく（任意だが偏りを減らす）
    for f in range(n_splits):
        rng.shuffle(val_folds[f])

    # (train_idx, val_idx) を作る
    all_idx = set(range(len(rows)))
    out: List[Tuple[List[int], List[int]]] = []
    for f in range(n_splits):
        val_idx = val_folds[f]
        train_idx = list(all_idx - set(val_idx))
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
                "ユーザーから与えられた id に必ず従い、必ず次のJSONだけを返してください:"
                '{"id": <int>, "prediction": "FINAL: <答え>"}'
                "prediction には答え以外（途中式・説明・改行・余計な文字）を入れないでください。"
            ),
        }
        
        fewshot_msgs = []

        examples_text = []
        for ex in prompt_train_rows:
            problem = str(ex.get("problem", "")).strip()
            solution = str(ex.get("solution", "")).strip()
            answer = str(ex.get("answer", "")).strip()

            examples_text.append(
                "【例題】\n"
                f"問題:\n{problem}\n\n"
                f"解法:\n{solution}\n\n"
                f"解答:\n{answer}\n"
            )

        fewshot_msgs.append({
            "role": "assistant",
            "content": (
                "以下は数学問題の例題・解法・解答例である。"
                "これらを参考にして、同様の形式で問題を解け。\n\n"
                + "\n\n".join(examples_text) +
                "ただし、出力は、以下のようにしてほしい"
                "ユーザーから与えられた id に必ず従い、必ず次のJSONだけを返してください:"
                '{"id": <int>, "prediction": "FINAL: <答え>"}'
                "prediction には答え以外（途中式・説明・改行・余計な文字）を入れないでください。"
            )
        })
        
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

    messages = [system_msg] + [user_msg]
    if fewshot_msgs:
        messages = [system_msg] + fewshot_msgs + [user_msg]
    # print(messages) # デバッグ用
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
