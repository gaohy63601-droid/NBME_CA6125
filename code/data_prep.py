"""
Convert NBME splits into instruction-format JSONL for Mistral Nemo Instruct SFT.

Train = train_split.csv (800 pn_num × ~14 features = 11440 rows)
Test  = test_split.csv (200 pn_num × ~14 features = 2860 rows, used for eval/inference)

Each example:
  user:    "Patient note:\n<pn_history>\n\nExtract the text spans that match the rubric feature: <feature_text>\nIf nothing matches, respond NO_MATCH."
  assistant: <each matched span on its own line, joined by " ||| ">  OR  "NO_MATCH"
"""
import os, ast, json
import pandas as pd

SPLITS = "/raid/yiren/ghy/motion_transfer/medical/nbme_baseline/splits"
OUT = "/raid/yiren/ghy/motion_transfer/medical/mistral_nemo/data"
os.makedirs(OUT, exist_ok=True)


def parse_annotation(ann_str):
    if not isinstance(ann_str, str) or ann_str in ("[]", "", "nan"): return []
    try:
        items = ast.literal_eval(ann_str)
        return [str(x).strip() for x in items if str(x).strip()]
    except Exception:
        return []


def feature_text_clean(t):
    return t.replace("-OR-", " or ").replace("-", " ")


def build_example(row):
    note = str(row["pn_history"])
    feat = feature_text_clean(str(row["feature_text"]))
    anns = parse_annotation(row["annotation"])
    user = (f"Patient note:\n{note}\n\n"
            f"Extract the exact text spans from the patient note that match the rubric feature: \"{feat}\".\n"
            f"Respond with each matching span on its own line, separated by \" ||| \". "
            f"If no part of the note matches, respond exactly: NO_MATCH")
    asst = " ||| ".join(anns) if anns else "NO_MATCH"
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": asst},
    ], "id": str(row["id"]), "pn_num": int(row["pn_num"]),
       "case_num": int(row["case_num"]), "feature_num": int(row["feature_num"])}


def main():
    for name in ["train_split", "test_split"]:
        df = pd.read_csv(os.path.join(SPLITS, f"{name}.csv"))
        out_path = os.path.join(OUT, f"{name}.jsonl")
        n_pos = n_neg = 0
        with open(out_path, "w") as f:
            for _, r in df.iterrows():
                ex = build_example(r)
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                if ex["messages"][1]["content"] == "NO_MATCH":
                    n_neg += 1
                else:
                    n_pos += 1
        print(f"{name}: {len(df)} rows  (pos={n_pos}, neg={n_neg})  -> {out_path}")


if __name__ == "__main__":
    main()
