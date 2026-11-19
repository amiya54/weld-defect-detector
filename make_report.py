#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 detect.py 输出的 JSON 汇总成一份 GitHub 网页可直接查看的 Markdown 报告。

在 GitHub Actions 云端运行，生成 results/报告.md。
GitHub 会自动渲染 Markdown 里的相对图片链接，点开就能看到画框结果图。
"""

import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

CLASS_ZH = {
    "undercut": "咬边", "slaginclusion": "夹渣", "porosity": "气孔",
    "crack": "裂纹", "overlap": "未熔合",
}


def fmt_time(iso_str):
    return iso_str.replace("T", " ") if iso_str else "未知"


def main():
    if not RESULTS_DIR.is_dir():
        print("[ERROR] 找不到 results 目录，请先运行 detect.py")
        return

    summary_path = RESULTS_DIR / "summary.json"
    summary = {}
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    result_files = sorted(RESULTS_DIR.glob("*_result.json"))
    lines = []
    lines.append("# 焊缝缺陷检测报告")
    lines.append("")
    lines.append(f"> 由 GitHub Actions 云端自动生成 · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append(f"- 检测图片数：**{summary.get('total_images', len(result_files))}**")
    lines.append(f"- 检出缺陷总数：**{summary.get('total_detections', '—')}**")
    lines.append("")
    lines.append("## 逐张结果")
    lines.append("")

    if not result_files:
        lines.append("_没有可展示的结果 JSON。_")
    for rf in result_files:
        with open(rf, encoding="utf-8") as f:
            meta = json.load(f)
        stem = rf.name.replace("_result.json", "")
        vis_rel = f"results/{stem}_vis.jpg"
        res = meta.get("result", {})
        dets = res.get("detections", [])
        n = res.get("n_detections", len(dets))
        lines.append(f"### {stem}")
        lines.append("")
        lines.append(f"![{stem} 检测结果]({vis_rel})")
        lines.append("")
        lines.append(f"- 检出缺陷：**{n}** 处")
        lines.append(f"- 参数：置信度 {meta.get('params', {}).get('conf', '—')} · "
                     f"瓦片重叠 {meta.get('params', {}).get('overrate', '—')}")
        lines.append(f"- 检测时间：{fmt_time(meta.get('timestamp'))}")
        if dets:
            lines.append("")
            lines.append("| 类别 | 置信度 | 坐标 (x1,y1,x2,y2) |")
            lines.append("|---|---|---|")
            for d in dets:
                zh = d.get("category_zh", CLASS_ZH.get(d.get("category", ""), d.get("category", "—")))
                box = ", ".join(f"{v:.0f}" for v in d.get("bbox_xyxy", []))
                lines.append(f"| {zh} | {d.get('score', 0):.2f} | {box} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("⚠️ 本工具只做初筛，缺陷判定与质量记录须经人工复核确认。")
    lines.append("")

    out_path = RESULTS_DIR / "报告.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] 报告已生成: {out_path}")


if __name__ == "__main__":
    main()
