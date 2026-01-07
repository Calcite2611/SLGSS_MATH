import os
import json
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import re
import math

# Config
MODEL = "gpt-4o-mini"
SEED = 42
MAX_RETRIES = 10
MAX_WORKERS = 15

TRAIN_PATH = "train/math_level12_easy_train100_student_with_answer_solution.jsonl"

# CV settings
N_SPLITS = 5
TEMPERATURE = 0.0

# few-shot を「同タイプ全件」にした結果、長すぎる場合のセーフティ（文字数ベース）
# 0 にすると無制限
MAX_FEWSHOT_CHARS = 0

print_lock = Lock()


# IO
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# Evaluation utils
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()

    if s.upper().startswith("FINAL:"):
        s = s.split(":", 1)[1].strip()

    s = s.strip()
    s = s.replace("−", "-")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace(",", "")

    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)

    s = s.strip()
    s = re.sub(r"^[\(\[\{]\s*", "", s)
    s = re.sub(r"\s*[\)\]\}]$", "", s)

    s = re.sub(r"\s+", "", s)
    return s

def try_parse_number(s: str) -> Optional[float]:
    if not s:
        return None

    if "/" in s and s.count("/") == 1:
        a, b = s.split("/")
        if a and b:
            na = try_parse_number(a)
            nb = try_parse_number(b)
            if na is not None and nb is not None and nb != 0:
                return na / nb
        return None

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s):
        try:
            return float(s)
        except:
            return None

    return None

def accuracy(rows, preds, *, rel_tol=1e-9, abs_tol=1e-9) -> float:
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

        pn = try_parse_number(pred)
        gn = try_parse_number(gold)
        if pn is not None and gn is not None:
            if math.isclose(pn, gn, rel_tol=rel_tol, abs_tol=abs_tol):
                correct += 1
                continue

    return correct / total if total else 0.0


# K-fold split
def kfold_split(rows: List[Dict[str, Any]], n_splits: int, seed: int) -> List[Tuple[List[int], List[int]]]:
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


# Same-type only sampling
def build_same_type_examples_only(
    train_by_type: Dict[str, List[Dict[str, Any]]],
    query_type: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    同タイプのみを使う。= train_by_type[query_type] を（seed で）シャッフルして全件返す。
    """
    rng = random.Random(seed)
    pool = train_by_type.get(query_type, [])
    tmp = pool[:]
    rng.shuffle(tmp)
    return tmp


# OpenAI call (structured output)
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


# PromptModel (your requested format: examples packed into one assistant message)
@dataclass
class PromptModel:
    seed: int

    def build_prompt(self, prompt_train_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        system_msg = {
            "role": "system",
            "content": "あなたは数学問題を解くアシスタントです。",
        }

        examples_text: List[str] = []
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

        assistant_content = (
            "以下は数学問題の例題・解法・解答例である。"
            "これらを参考にして、同様の形式で問題を解け。\n\n"
            + "\n\n".join(examples_text)
            + "\n\n"
            "ただし、出力は、以下のようにしてほしい。"
            "ユーザーから与えられた id に必ず従い、必ず次のJSONだけを返してください:"
            '{"id": <int>, "prediction": "FINAL: <答え>"}'
            "prediction には答え以外（途中式・説明・改行・余計な文字）を入れないでください。"
        )

        # セーフティ：長くなりすぎると API 側で落ちるので、文字数で上限を掛ける（必要なら 0 に）
        if MAX_FEWSHOT_CHARS and len(assistant_content) > MAX_FEWSHOT_CHARS:
            with print_lock:
                print(
                    f"[warn] few-shot prompt too long: {len(assistant_content)} chars "
                    f"> {MAX_FEWSHOT_CHARS}. Truncating examples."
                )
            # 先頭の指示は残しつつ、例題部分を後ろから削る
            header = (
                "以下は数学問題の例題・解法・解答例である。"
                "これらを参考にして、同様の形式で問題を解け。\n\n"
            )
            footer = (
                "\n\n"
                "ただし、出力は、以下のようにしてほしい。"
                "ユーザーから与えられた id に必ず従い、必ず次のJSONだけを返してください:"
                '{"id": <int>, "prediction": "FINAL: <答え>"}'
                "prediction には答え以外（途中式・説明・改行・余計な文字）を入れないでください。"
            )

            # 例題を先頭から順に詰め直し（できるだけ多く入れる）
            body = ""
            for t in examples_text:
                cand = (body + ("\n\n" if body else "") + t)
                if len(header) + len(cand) + len(footer) > MAX_FEWSHOT_CHARS:
                    break
                body = cand
            assistant_content = header + body + footer

        fewshot_msgs = [{"role": "assistant", "content": assistant_content}]
        return system_msg, fewshot_msgs


# Solve one (same-type only, k = size of same-type train pool)
def solve_one_same_type_only(
    client: OpenAI,
    model: str,
    prompt_model: PromptModel,
    train_by_type: Dict[str, List[Dict[str, Any]]],
    row: Dict[str, Any],
    fold_i: int,
) -> Dict[str, Any]:
    qid = int(row["id"])
    problem = row["problem"]
    qtype = str(row.get("type", "UNKNOWN"))

    # 同タイプ train 全件を few-shot にする（fold 内 train の同タイプ件数）
    same_type_pool = train_by_type.get(qtype, [])
    fewshot_k = len(same_type_pool)

    # ただし「同タイプだけ」を満たしつつ、順序は seed でシャッフルして固定化
    examples = build_same_type_examples_only(
        train_by_type=train_by_type,
        query_type=qtype,
        seed=prompt_model.seed + fold_i * 100000 + qid,
    )

    # 念のため（取得できた例題数がそのまま fewshot_k）
    if len(examples) != fewshot_k:
        fewshot_k = len(examples)

    system_msg, fewshot_msgs = prompt_model.build_prompt(examples)

    user_msg = {
        "role": "user",
        "content": (
            f"次の問題を解いて。\n\n"
            f"problem:\n{problem}\n\n"
            f"id は {qid}。\n"
            "JSONで出力して。prediction は 'FINAL: <答え>' だけ。"
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


def run_inference_same_type_only(
    client: OpenAI,
    model: str,
    prompt_model: PromptModel,
    train_by_type: Dict[str, List[Dict[str, Any]]],
    rows: List[Dict[str, Any]],
    fold_i: int,
    desc: str = "",
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    done = 0
    total = len(rows)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(solve_one_same_type_only, client, model, prompt_model, train_by_type, row, fold_i)
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


# Cross-validation runner
def cross_validate_prompt_model_same_type_only(
    client: OpenAI,
    model: str,
    rows: List[Dict[str, Any]],
    prompt_model: PromptModel,
    n_splits: int,
    seed: int,
) -> Dict[str, Any]:
    splits = kfold_split(rows, n_splits=n_splits, seed=seed)

    fold_scores: List[float] = []
    fold_fewshot_stats: List[Dict[str, Any]] = []

    for fold_i, (train_idx, val_idx) in enumerate(splits, start=1):
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        # train を type ごとにまとめる
        train_by_type = defaultdict(list)
        for r in train_rows:
            t = str(r.get("type", "UNKNOWN"))
            train_by_type[t].append(r)

        # 参考：val の type ごとの few-shot 件数（= train の同タイプ件数）統計を出す
        type_counts = defaultdict(int)
        for vr in val_rows:
            qt = str(vr.get("type", "UNKNOWN"))
            type_counts[qt] += len(train_by_type.get(qt, []))
        fold_fewshot_stats.append({
            "fold": fold_i,
            "val_size": len(val_rows),
            "types_in_val": len(set(str(v.get("type","UNKNOWN")) for v in val_rows)),
            "avg_fewshot_k_over_val": (sum(type_counts.values()) / len(val_rows)) if val_rows else 0.0,
        })

        preds = run_inference_same_type_only(
            client=client,
            model=model,
            prompt_model=prompt_model,
            train_by_type=train_by_type,
            rows=val_rows,
            fold_i=fold_i,
            desc=f"[fold {fold_i}/{n_splits}]",
        )

        fold_acc = accuracy(val_rows, preds)
        fold_scores.append(fold_acc)

        with print_lock:
            print(f"[fold {fold_i}] acc = {fold_acc:.4f} (val={len(val_rows)})")

    mean_acc = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0
    return {
        "model": model,
        "n_splits": n_splits,
        "fold_scores": fold_scores,
        "mean_score": mean_acc,
        "fewshot_prompt_max_chars": MAX_FEWSHOT_CHARS,
        "fold_fewshot_stats": fold_fewshot_stats,
    }


# main
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が環境変数に設定されていません。")

    client = OpenAI()
    rows = read_jsonl(TRAIN_PATH)

    # type 欠損があると UNKNOWN 扱い
    missing = sum(1 for r in rows if "type" not in r)
    if missing:
        with print_lock:
            print(f"[warn] rows missing 'type': {missing}/{len(rows)} (treated as UNKNOWN)")

    prompt_model = PromptModel(seed=SEED)

    result = cross_validate_prompt_model_same_type_only(
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
