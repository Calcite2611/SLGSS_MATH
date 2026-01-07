import json
import os
import asyncio
from openai import AsyncOpenAI  # 非同期版クライアントを使用
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
# AsyncOpenAIを使用します
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 同時に実行する最大数（GPT-4o-miniなら20〜50程度でも動きますが、まずは10で設定）
MAX_CONCURRENT_TASKS = 10

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_problem(file_path):
    if file_path.endswith('.jsonl'):
        return load_jsonl(file_path)
    return load_json(file_path)

def get_field(data: dict, keys, default: str = ""):
    for k in keys:
        if k in data and data[k]:
            return data[k]
    return default

async def solve_math_problem_async(idx, target_problem, reference_data, semaphore):
    """
    1つの問題を解く非同期タスク
    """
    async with semaphore:  # 同時実行数を制限
        try:
            # Few-shot用の例を作成
            examples = ""
            for i, item in enumerate(reference_data[:3]):
                q = get_field(item, ["question", "input", "prompt", "problem"])
                a = get_field(item, ["answer", "output", "solution"])
                if q and a:
                    examples += f"例題{i+1}: {q}\n解答{i+1}: {a}\n\n"
            
            system_prompt = "You are a helpful assistant specialized in solving high school level mathematics problems. Provide answers with detailed step-by-step explanations."

            target_question = get_field(target_problem, ["question", "input", "prompt", "problem"])
            if not target_question:
                return {"id": idx, "prediction": "Error: Question field not found."}

            user_content = f"Below are references:\n{examples}\nProblem to solve:\n{target_question}"

            # 非同期でのAPI呼び出し
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0
            )
            
            print(f"  [完了] 問題 {idx+1}")
            return {"id": idx, "prediction": response.choices[0].message.content}

        except Exception as e:
            print(f"  [エラー] 問題 {idx+1}: {e}")
            return {"id": idx, "prediction": f"Error during inference: {str(e)}"}

async def main():
    # データのロード
    train_data = load_jsonl('train_data.jsonl')
    problems = load_problem('input_problem.jsonl')
    if isinstance(problems, dict):  # 単一の辞書だった場合のケア
        problems = [problems]

    print(f"推論を開始します... (総問題数: {len(problems)}, 同時実行数: {MAX_CONCURRENT_TASKS})")
    
    # セマフォの作成
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    # 全問題のタスクを作成
    tasks = [
        solve_math_problem_async(i, prob, train_data, semaphore) 
        for i, prob in enumerate(problems)
    ]
    
    # 全タスクを並列実行して結果を待つ
    results = await asyncio.gather(*tasks)

    # 結果をID順にソート（並列実行だと順番が前後するため）
    results.sort(key=lambda x: x["id"])

    # 結果の保存
    output_path = 'output.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\nすべての処理が完了しました。結果を {output_path} に保存しました。")

if __name__ == "__main__":
    # 非同期処理の開始
    asyncio.run(main())