# score_math_test.py
# pip install -U sympy (sympyがあると同値判定が少し強くなる)

import json
import re

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def extract_final_answer(text: str):
    if text is None:
        return None
    s = str(text)

    # 1) FINAL: を優先
    m = re.search(r"FINAL:\s*(.+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) \boxed{...}
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    if m:
        return m.group(1).strip()

    # 3) 最後の非空行（最後の手段）
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else None

def normalize_latex(expr: str):
    if expr is None:
        return None
    x = expr.strip()

    # remove common latex wrappers/noise
    x = x.replace("$", "")
    x = x.replace("\\left", "").replace("\\right", "")
    x = re.sub(r"\\[,\s;!]+", "", x)

    # unify pi
    x = x.replace("\\pi", "pi")

    # \frac{a}{b} -> (a)/(b)
    # 簡易的に繰り返し置換（ネストが深いと壊れることがあるが、easyフィルタなら大体OK）
    while True:
        m = re.search(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", x)
        if not m:
            break
        a, b = m.group(1), m.group(2)
        x = x[:m.start()] + f"(({a})/({b}))" + x[m.end():]

    # \sqrt{a} -> sqrt(a)
    while True:
        m = re.search(r"\\sqrt\{([^{}]+)\}", x)
        if not m:
            break
        a = m.group(1)
        x = x[:m.start()] + f"sqrt({a})" + x[m.end():]

    # ^ -> **
    x = x.replace("^", "**")

    # spaces
    x = re.sub(r"\s+", "", x)
    return x

def equivalent(gold: str, pred: str) -> bool:
    g = normalize_latex(gold)
    p = normalize_latex(pred)
    if g is None or p is None:
        return False
    if g == p:
        return True

    # sympyで同値判定（可能なときだけ）
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application
        )
        trans = standard_transformations + (implicit_multiplication_application,)
        g_expr = parse_expr(g, transformations=trans, evaluate=True)
        p_expr = parse_expr(p, transformations=trans, evaluate=True)
        return sp.simplify(g_expr - p_expr) == 0
    except Exception:
        return False

def main():
    gold_path = "math_level12_easy_test100_teacher.jsonl"
    pred_path = "preds.jsonl"  # ここを自分の予測ファイル名に

    gold_rows = load_jsonl(gold_path)
    pred_rows = load_jsonl(pred_path)

    gold_by_id = {r["id"]: r for r in gold_rows}
    pred_by_id = {r["id"]: r for r in pred_rows}

    total = 0
    correct = 0
    wrong = []

    for gid, g in gold_by_id.items():
        total += 1
        gold_ans = g.get("answer")
        pred_raw = pred_by_id.get(gid, {}).get("prediction")
        pred_ans = extract_final_answer(pred_raw)

        ok = equivalent(gold_ans, pred_ans)
        if ok:
            correct += 1
        else:
            wrong.append({
                "id": gid,
                "type": g.get("type"),
                "gold": gold_ans,
                "pred": pred_ans,
            })

    acc = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} = {acc:.4f}")

    # 失敗例を少し表示
    if wrong:
        print("\nExamples of wrong answers (up to 10):")
        for w in wrong[:10]:
            print(w)

if __name__ == "__main__":
    main()