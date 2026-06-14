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


def run_all(data_root, out_dir="outputs", kinds=("logreg", "svm")):
    """一行跑完整套流程：加载→传统方法→U-Net→出图出表。返回 DataFrame。"""
    data = datamod.load_data(data_root)
    clf = classical.run_classical(data, kinds=kinds)
    unet = segmentation.run_unet(data)
    df = report(data, clf, unet, out_dir=out_dir, kinds=kinds)
    return df
