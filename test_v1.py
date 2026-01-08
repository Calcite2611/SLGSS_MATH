import os
import json
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Settings
MODEL = "gpt-4o-mini"
SEED = 42
MAX_RETRIES = 10
MAX_WORKERS = 15

# few-shot / prompt settings
PROMPT_TRAIN_K = 20
TEMPERATURE = 0.0

# paths
TRAIN_PATH = "train/math_level12_easy_train100_student_with_answer_solution.jsonl"
TEST_PATH  = "test/math_level12_easy_test100_student.jsonl"  
OUT_PATH   = "out/preds_test_v1.jsonl"                   

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

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(int(obj["id"]))
            except:
                # 壊れた行が混じっていても継続
                continue
    return done

# Few-shot selection helpers
def take_k(rows: List[Dict[str, Any]], k: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    tmp = rows[:]
    rng.shuffle(tmp)
    return tmp[: min(k, len(tmp))]

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

    last_err = None
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
            last_err = e
            if attempt == max_retries - 1:
                raise
            sleep_s = 2 ** attempt
            with print_lock:
                print(f"[warn] {e} -> retry in {sleep_s}s")
            time.sleep(sleep_s)

    raise RuntimeError(f"unreachable: {last_err}")

# PromptModel 
@dataclass
class PromptModel:
    fewshot_k: int
    seed: int

    def build_prompt(self, prompt_train_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Optional[List[Dict[str, str]]]]:
        system_msg = {
            "role": "system",
            "content": (
                "あなたは数学問題を解くアシスタントです。"
                "与えられた問題に対して、必ずJSONで {id:int, prediction:str} を返してください。"
                "prediction は必ず 'FINAL: <答え>' の1行だけにしてください。"
                "途中式や解説は prediction に含めないでください。"
            ),
        }

        fewshot_msgs = None

        return system_msg, fewshot_msgs

# Inference core
def solve_one(
    client: OpenAI,
    model: str,
    system_msg: Dict[str, str],
    fewshot_msgs: Optional[List[Dict[str, str]]],
    row: Dict[str, Any],
) -> Dict[str, Any]:
    qid = int(row["id"])
    problem = str(row["problem"])

    user_msg = {
        "role": "user",
        "content": (
            f"次の問題を解いて。\n\n"
            f"problem:\n{problem}\n\n"
            f"id は {qid}。\n"
            f"JSONで出力して。prediction は 'FINAL: <答え>' だけ。"
        ),
    }

    if fewshot_msgs:
        messages = [system_msg] + fewshot_msgs + [user_msg]
    else:
        messages = [system_msg] + [user_msg]

    pred_obj = call_model_json(
        client=client,
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_retries=MAX_RETRIES,
    )

    # id固定 & prediction矯正
    pred_obj["id"] = qid
    p = str(pred_obj.get("prediction", "")).strip()
    if not p.startswith("FINAL:"):
        p = "FINAL: " + p
    pred_obj["prediction"] = p
    return pred_obj

def run_inference_streaming(
    client: OpenAI,
    model: str,
    system_msg: Dict[str, str],
    fewshot_msgs: Optional[List[Dict[str, str]]],
    rows: List[Dict[str, Any]],
    out_path: str,
    desc: str = "",
    resume: bool = True,
) -> List[Dict[str, Any]]:
    """
    - 並列推論
    - 結果を out_path に逐次 append（落ちても途中まで残る）
    - resume=True なら out_path に既にある id はスキップ
    """
    done_ids = load_done_ids(out_path) if resume else set()

    # 対象 rows を絞る
    todo = [r for r in rows if int(r["id"]) not in done_ids]
    total = len(rows)
    with print_lock:
        print(f"{desc} total={total}, already_done={len(done_ids)}, todo={len(todo)}")

    outputs: List[Dict[str, Any]] = []
    done_count = 0
    lock_out = Lock()

    def _worker(r):
        return solve_one(client, model, system_msg, fewshot_msgs, r)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_worker, r) for r in todo]
        for fut in as_completed(futures):
            pred = fut.result()
            outputs.append(pred)

            # 逐次保存
            with lock_out:
                append_jsonl(out_path, pred)

            done_count += 1
            if done_count % 10 == 0:
                with print_lock:
                    print(f"{desc} done {done_count}/{len(todo)} (appended)")

    # 既存 + 新規 をまとめて返す（返り値は必須ではないけど便利なので）
    # out_path を読み直して sort して返す
    merged = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    merged.append(json.loads(line))
                except:
                    continue
    merged.sort(key=lambda x: int(x.get("id", 10**18)))
    return merged

# main
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が環境変数に設定されていません。")

    client = OpenAI()

    # train/test を読む
    train_rows = read_jsonl(TRAIN_PATH)
    # test_rows = read_jsonl(TEST_PATH)
    test_rows = read_jsonl(TRAIN_PATH)

    # prompt model
    prompt_model = PromptModel(
        fewshot_k=PROMPT_TRAIN_K,
        seed=SEED,
    )

    # few-shot 用の train サンプルを作る（今は build_prompt が None 返すので実質未使用だが形は合わせる）
    prompt_train_rows = take_k(train_rows, prompt_model.fewshot_k, seed=prompt_model.seed)
    system_msg, fewshot_msgs = prompt_model.build_prompt(prompt_train_rows)

    # 出力ファイルを新規作成するなら消す（resumeしたいなら消さない）
    # ここは好みで。resumeしたいならコメントアウト推奨。
    # if os.path.exists(OUT_PATH):
    #     os.remove(OUT_PATH)

    merged = run_inference_streaming(
        client=client,
        model=MODEL,
        system_msg=system_msg,
        fewshot_msgs=fewshot_msgs,
        rows=test_rows,
        out_path=OUT_PATH,
        desc="[test]",
        resume=True,
    )

    sorted_path = OUT_PATH.replace(".jsonl", "_sorted.jsonl")
    merged.sort(key=lambda x: int(x["id"]))
    write_jsonl(sorted_path, merged)

    with print_lock:
        print(f"sorted wrote: {sorted_path}")
        
    with print_lock:
        print("\n=== DONE ===")
        print(f"wrote: {OUT_PATH}")
        print(f"lines: {len(merged)} (may include only valid json lines)")

if __name__ == "__main__":
    main()
