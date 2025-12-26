import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def solve_math_problem(target_problem, reference_data):
    # 1. Few-shot用の例を作成（最初の3問を例として提示する簡易版）
    # ※ 本来はembeddingを使用して類似問題を抽出するとより精度が上がります
    examples = ""
    for i, item in enumerate(reference_data[:3]):
        examples += f"例題{i+1}: {item['question']}\n解答{i+1}: {item['answer']}\n\n"

    # 2. システムプロンプトの設定
    system_prompt = """
    あなたは優秀な数学の家庭教師です。
    与えられた問題に対して、以下の手順で回答してください。
    1. 問題の要点を整理する
    2. 解法をステップバイステップで説明する
    3. 最終的な数値を明確に提示する
    """

    # 3. GPT-4oへのリクエスト
    user_content = f"""
    以下の例を参考に、新しい問題を解いてください。

    {examples}

    解くべき問題:
    {target_problem['question']}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0  # 数学問題は決定論的な回答が望ましいため0に設定
    )

    return response.choices[0].message.content

def main():
    # データのロード
    train_data = load_json('train_data.json')
    new_problem = load_json('input_problem.json')

    print("推論を開始します...")
    result = solve_math_problem(new_problem, train_data)
    
    print("\n--- 解答結果 ---")
    print(result)

if __name__ == "__main__":
    main()