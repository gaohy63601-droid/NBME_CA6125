"""
Within the 800-pn_num train_split.csv, build 5-fold splits (GroupKFold by pn_num,
stratified by case_num so each fold sees all 10 cases).
Held-out test_split.csv (200 pn_num) is untouched - used for final ensemble eval.
"""
import os
import pandas as pd
from sklearn.model_selection import GroupKFold

SPLITS_DIR = "/raid/yiren/ghy/motion_transfer/medical/nbme_baseline/splits"
N_FOLDS = 5
SEED = 42


def main():
    df = pd.read_csv(os.path.join(SPLITS_DIR, "train_split.csv"))
    df = df.reset_index(drop=True)
    print(f"loaded {len(df)} rows, {df['pn_num'].nunique()} unique pn_num")

    # Per-case GroupKFold so every fold has every case_num
    df["fold"] = -1
    for case in sorted(df["case_num"].unique()):
        sub_idx = df.index[df["case_num"] == case].to_numpy()
        sub = df.loc[sub_idx]
        gkf = GroupKFold(n_splits=N_FOLDS)
        for f, (_, val_idx) in enumerate(gkf.split(sub, groups=sub["pn_num"])):
            df.loc[sub_idx[val_idx], "fold"] = f

    assert (df["fold"] >= 0).all()
    print("fold sizes:")
    for f in range(N_FOLDS):
        n_rows = (df["fold"] == f).sum()
        n_pn = df.loc[df["fold"] == f, "pn_num"].nunique()
        print(f"  fold {f}: {n_rows} rows, {n_pn} pn_num")

    # Sanity: no pn_num appears in 2 folds
    for f in range(N_FOLDS):
        in_f = set(df.loc[df["fold"] == f, "pn_num"])
        out_f = set(df.loc[df["fold"] != f, "pn_num"])
        assert not (in_f & out_f), f"fold {f} pn_num leakage"

    out_path = os.path.join(SPLITS_DIR, "train_split_5fold.csv")
    df.to_csv(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
