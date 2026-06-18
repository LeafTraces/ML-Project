"""高层编排：把各模块串起来，notebook 只需调用这里的几个函数。"""
import os
import json
import shutil

from . import config, data as datamod, classical, segmentation, viz


def assemble_all(classical_results, unet_result, kinds=("logreg", "svm")):
    """统一成 [(名称, 测试概率, 指标, 曲线, 学习时间, 阈值), ...]。"""
    ALL = []
    for k in kinds:
        d = classical_results[k]
        ALL.append((d["name"], d["test_prob"], d["m"], d["c"], d["time"], d["thr"]))
    ALL.append((unet_result["name"], unet_result["test_prob"], unet_result["m"],
                unet_result["c"], unet_result["time"], unet_result["thr"]))
    return ALL


def report(data, classical_results, unet_result, out_dir="outputs",
           kinds=("logreg", "svm"), make_figs=True):
    """生成全部图表 + 指标表 + results.json/csv，返回 pandas DataFrame。"""
    import pandas as pd
    fig_dir = os.path.join(out_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)
    ALL = assemble_all(classical_results, unet_result, kinds)

    if make_figs:
        viz.fig_preprocess(data, fig_dir)
        viz.fig_feature_coef(classical_results, fig_dir)
        viz.fig_training_curve(unet_result, fig_dir)
        viz.fig_roc_pr(ALL, fig_dir)
        viz.fig_confusion(ALL, fig_dir)
        viz.fig_qualitative(data, ALL, fig_dir)

    rows = []
    for name, _, m, c, tt, thr in ALL:
        rows.append({"Method": name, "Learn time(s)": round(tt, 1), "Threshold": round(thr, 2),
                     "Accuracy": round(m["accuracy"], 4), "Precision": round(m["precision"], 4),
                     "Recall": round(m["recall"], 4), "Specificity": round(m["specificity"], 4),
                     "F1": round(m["f1"], 4), "IoU": round(m["iou"], 4), "Dice": round(m["dice"], 4),
                     "ROC-AUC": round(c["roc_auc"], 4), "PR-AUC": round(c["pr_auc"], 4)})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out_dir, "results_table.csv"), index=False)
    results = {"work_size": config.img_size(), "methods": rows,
               "confusion": {name: m["cm"].tolist() for name, _, m, c, tt, thr in ALL},
               "lr_feat_coef": classical.lr_feature_coef(classical_results)}
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), ensure_ascii=False, indent=2)
    print(f"\n已保存 {out_dir}/results.json 与 results_table.csv")
    return df


def make_zip(out_dir="outputs", zip_name="exudate_outputs"):
    shutil.make_archive(zip_name, "zip", out_dir)
    print(f"已打包 {zip_name}.zip")
    return f"{zip_name}.zip"


def run_ablation(data, encoder=None, epochs=None, seed=None,
                 out_dir="outputs", make_fig=True, verbose=True):
    """消融实验：同一数据划分、同一随机种子下，对比 ImageNet 预训练 vs 随机初始化的 U-Net。
    两次训练仅“编码器初始权重”不同（结构、损失、优化器、增广、轮数、数据划分全相同），
    从而定量评估预训练带来的增益。返回 {pretrained, random, df}，并保存 ablation 表/图/json。
    """
    import pandas as pd
    seed = config.SEED if seed is None else seed
    print("=" * 56 + "\n[消融 1/2] ImageNet 预训练编码器\n" + "=" * 56)
    pre = segmentation.run_unet(data, encoder=encoder, epochs=epochs,
                                encoder_weights="imagenet", seed=seed, verbose=verbose)
    pre["name"] = "U-Net (ImageNet pretrained)"
    print("=" * 56 + "\n[消融 2/2] 随机初始化编码器\n" + "=" * 56)
    rnd = segmentation.run_unet(data, encoder=encoder, epochs=epochs,
                                encoder_weights=None, seed=seed, verbose=verbose)
    rnd["name"] = "U-Net (random init)"

    def _row(r):
        m, c = r["m"], r["c"]
        return {"Setting": r["name"], "Learn time(s)": round(r["time"], 1), "Threshold": round(r["thr"], 2),
                "Precision": round(m["precision"], 4), "Recall": round(m["recall"], 4),
                "F1": round(m["f1"], 4), "IoU": round(m["iou"], 4), "Dice": round(m["dice"], 4),
                "ROC-AUC": round(c["roc_auc"], 4), "PR-AUC": round(c["pr_auc"], 4)}

    df = pd.DataFrame([_row(pre), _row(rnd)])
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "ablation_table.csv"), index=False)
    dd = pre["m"]["dice"] - rnd["m"]["dice"]; dp = pre["c"]["pr_auc"] - rnd["c"]["pr_auc"]
    json.dump({"rows": df.to_dict("records"), "delta_dice": round(dd, 4), "delta_pr_auc": round(dp, 4)},
              open(os.path.join(out_dir, "ablation.json"), "w"), ensure_ascii=False, indent=2)
    if make_fig:
        viz.fig_ablation(pre, rnd, os.path.join(out_dir, "figures"))
    print("\n" + "=" * 56)
    print(f"[消融结论] 预训练 − 随机:  ΔDice = {dd:+.3f}   ΔPR-AUC = {dp:+.3f}")
    print(df.to_string(index=False))
    print(f"\n已保存 {out_dir}/ablation_table.csv、ablation.json、figures/fig7_ablation.png")
    return {"pretrained": pre, "random": rnd, "df": df}


def run_all(data_root, out_dir="outputs", kinds=("logreg", "svm")):
    """一行跑完整套流程：加载→传统方法→U-Net→出图出表。返回 DataFrame。"""
    data = datamod.load_data(data_root)
    clf = classical.run_classical(data, kinds=kinds)
    unet = segmentation.run_unet(data)
    df = report(data, clf, unet, out_dir=out_dir, kinds=kinds)
    return df
