#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
焊缝外观缺陷检测 —— 两段式推理（焊缝区域定位 + 缺陷检测）

源自 RH821/ship-ai-system 的 modules/weld_defect/infer.py，
在原算法逻辑不变的前提下做了工程化清理：
  - 移除了原文中两处未解决的 Git 合并冲突标记
  - 移除了硬编码的 F:\\weldtoolv2.1\\... 输出路径
  - 权重路径改为相对本脚本解析，输入/输出改为命令行参数
  - 中间产物改为可选保存，默认只输出最终画框图与 JSON

缺陷类别（对标 CB/T 3802-2019）：
  undercut 咬边 / slaginclusion 夹渣 / porosity 气孔 / crack 裂纹 / overlap 未熔合
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

CLASS_COLOR = {
    "undercut": (0, 0, 255),        # 红 - 咬边
    "slaginclusion": (0, 255, 0),   # 绿 - 夹渣
    "porosity": (255, 0, 0),        # 蓝 - 气孔
    "crack": (255, 255, 0),         # 青 - 裂纹
    "overlap": (0, 165, 255),       # 橙 - 未熔合
}
DEFAULT_COLOR = (0, 128, 255)
CLASS_ZH = {
    "undercut": "咬边", "slaginclusion": "夹渣", "porosity": "气孔",
    "crack": "裂纹", "overlap": "未熔合",
}


# -------------------- 几何辅助函数 --------------------

def rotate90_cw(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=float)


def bbox_scale(box_xywh, scale):
    x, y, w, h = box_xywh
    return np.array([x * scale, y * scale, w * scale, h * scale], dtype=float)


def bbox_translate(box_xywh, tx, ty):
    x, y, w, h = box_xywh
    return np.array([x + tx, y + ty, w, h], dtype=float)


def bbox_from_rotated_to_unrotated_xywh(box_xywh, w0, h0):
    """rotated 图的 box 还原到未旋转图（保持原逻辑，w0 不参与）。"""
    x, y, w, h = box_xywh
    x_u = y
    y_u = h0 - (x + w)
    w_u = h
    h_u = w
    return np.array([x_u, y_u, w_u, h_u], dtype=float)


def clip_box_xyxy(box, W, H):
    x1, y1, x2, y2 = box
    return np.array([
        clamp(x1, 0, W - 1), clamp(y1, 0, H - 1),
        clamp(x2, 0, W - 1), clamp(y2, 0, H - 1),
    ], dtype=float)


# -------------------- 亮度/对比度自适应预处理 --------------------

def adaptive_normalize_tile(tile_bgr: np.ndarray,
                            clahe_clip: float = 2.0,
                            clahe_grid: int = 8,
                            highlight_compress: bool = True,
                            hi_thresh: int = 220,
                            hi_compress: float = 0.45,
                            shadow_lift: bool = True,
                            shadow_q: float = 0.15,
                            shadow_target: int = 95,
                            max_lift: int = 70) -> np.ndarray:
    """适合焊缝：强反光 + 局部暗。
    - CLAHE：提升局部对比度（主力）
    - 高亮压缩：避免反光区域过曝
    - 阴影提升（按暗部分位数）：只抬暗部，不把亮部推爆
    """
    out = tile_bgr.copy()
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip),
                            tileGridSize=(int(clahe_grid), int(clahe_grid)))
    L2 = clahe.apply(L)

    if highlight_compress:
        L2 = np.where(L2 > hi_thresh,
                      hi_thresh + (L2 - hi_thresh) * float(hi_compress),
                      L2).astype(np.uint8)

    if shadow_lift:
        Lf = L2.astype(np.float32)
        qv = float(np.quantile(Lf, float(shadow_q)))
        lift = float(shadow_target) - qv
        lift = float(np.clip(lift, 0.0, float(max_lift)))
        if lift > 1e-3:
            w = (1.0 - (Lf / 255.0)) ** 2
            Lf2 = Lf + lift * w
            L2 = np.clip(Lf2, 0, 255).astype(np.uint8)

    out = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)
    return out


# -------------------- 辅助工具函数 --------------------

def detect_weld_area(area_model: YOLO, image_bgr, conf=0.25, device=None):
    """第一阶段：在整图里找最大的焊缝区域框。"""
    H, W = image_bgr.shape[:2]
    kwargs = dict(source=image_bgr, conf=conf, verbose=False)
    if device:
        kwargs["device"] = device
    res = area_model.predict(**kwargs)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(np.argmax(areas))
    x1, y1, x2, y2 = xyxy[idx]
    return clip_box_xyxy(np.array([x1, y1, x2, y2]), W, H)


def tile_width_positions(total_w, tile_w=640, overrate=0.2):
    overrate = min(max(float(overrate), 0.0), 0.9)
    stride = max(1, int(round(tile_w * (1.0 - overrate))))
    if total_w <= tile_w:
        return [0]
    xs, x = [0], 0
    while x + tile_w < total_w:
        x_next = x + stride
        if x_next + tile_w >= total_w:
            xs.append(total_w - tile_w)
            break
        xs.append(x_next)
        x = x_next
    return xs


def draw_boxes(img_bgr, boxes_xyxy, labels=None, colors=None, thickness=2):
    out = img_bgr.copy()
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if colors is None else colors[i]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if labels is not None:
            txt = labels[i]
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, txt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# -------------------- 主流程 --------------------

def run_pipeline(image_path, area_model, defect_model, out_dir,
                 overrate=0.0, conf=0.25, visualize=True,
                 normalize_tiles=True, clahe_clip=2.0, clahe_grid=8,
                 save_tiles=False, device=None):
    time_start = time.time()
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print("[WARN] 无法读取图像:", image_path)
        return None
    H0, W0 = img_bgr.shape[:2]
    stem = Path(image_path).stem
    os.makedirs(out_dir, exist_ok=True)

    debug_dir = os.path.join(out_dir, "_tiles") if save_tiles else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # 第一阶段：焊缝区域定位
    area_xyxy = detect_weld_area(area_model, img_bgr, conf=conf, device=device)
    if area_xyxy is None:
        print("[WARN] 未检测到焊缝区域，保存原图与空结果")
        cv2.imwrite(os.path.join(out_dir, f"{stem}_vis.jpg"), img_bgr)
        meta = {
            "task": "weld_defect",
            "source": str(image_path),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params": {"conf": float(conf), "overrate": float(overrate)},
            "result": {
                "image_size": [int(W0), int(H0)],
                "n_detections": 0,
                "detections": [],
                "message": "No weld area detected",
            },
        }
        with open(os.path.join(out_dir, f"{stem}_result.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta

    # 裁焊缝区域（长边方向保留整幅）
    ax1, ay1, ax2, ay2 = map(int, area_xyxy)
    if ax2 - ax1 >= ay2 - ay1:
        ax1, ax2 = 0, W0
    else:
        ay1, ay2 = 0, H0
    crop = img_bgr[ay1:ay2, ax1:ax2].copy()
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_weld_area.jpg"), crop)

    # 竖图转横、统一缩放到高 640、横向切 640 瓦片
    ch, cw = crop.shape[:2]
    rotated = False
    if cw < ch:
        crop = rotate90_cw(crop)
        rotated = True
        ch, cw = crop.shape[:2]

    target_h = 640
    scale = target_h / float(ch)
    new_w = max(1, int(round(cw * scale)))
    resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    xs = tile_width_positions(new_w, tile_w=640, overrate=overrate)
    det_records = []

    for i_tile, x0 in enumerate(xs):
        tile = resized[:, x0:x0 + 640].copy()
        if normalize_tiles:
            tile = adaptive_normalize_tile(tile, clahe_clip=clahe_clip, clahe_grid=clahe_grid)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"{stem}_tile{i_tile:03d}.jpg"), tile)

        # 第二阶段：缺陷检测
        kwargs = dict(source=tile, conf=conf, verbose=False)
        if device:
            kwargs["device"] = device
        res = defect_model.predict(**kwargs)[0]
        det_records_tile = []
        if res.boxes is not None and len(res.boxes) > 0:
            b_xyxy = res.boxes.xyxy.cpu().numpy()
            b_cls = res.boxes.cls.cpu().numpy().astype(int)
            b_conf = res.boxes.conf.cpu().numpy()
            # 原逻辑：过滤贴边框
            mask = (b_xyxy[:, 3] < 600) & (b_xyxy[:, 1] > 40)
            b_xyxy, b_cls, b_conf = b_xyxy[mask], b_cls[mask], b_conf[mask]
            for k in range(len(b_xyxy)):
                cls_name = defect_model.names.get(int(b_cls[k]), str(b_cls[k]))
                det_records_tile.append({
                    "bbox_xyxy": [float(v) for v in b_xyxy[k]],
                    "category": cls_name,
                    "category_zh": CLASS_ZH.get(cls_name, cls_name),
                    "score": float(b_conf[k]),
                })

        if debug_dir:
            tile_vis = draw_boxes(
                tile,
                boxes_xyxy=[r["bbox_xyxy"] for r in det_records_tile],
                labels=[f"{r['category']} {r['score']:.2f}" for r in det_records_tile],
                colors=[CLASS_COLOR.get(r["category"], DEFAULT_COLOR) for r in det_records_tile],
            )
            cv2.imwrite(os.path.join(debug_dir, f"{stem}_tile{i_tile:03d}_detected.jpg"), tile_vis)

        # 瓦片坐标 -> 原图坐标
        for r in det_records_tile:
            box_r_xywh = xyxy_to_xywh(r["bbox_xyxy"])
            box_r_xywh[0] += x0
            box_u_xywh = bbox_scale(box_r_xywh, 1.0 / scale)
            box_pre_xywh = bbox_from_rotated_to_unrotated_xywh(box_u_xywh, w0=ch, h0=cw) if rotated else box_u_xywh
            box_img_xywh = bbox_translate(box_pre_xywh, ax1, ay1)
            box_img_xyxy = clip_box_xyxy(xywh_to_xyxy(box_img_xywh), W0, H0)
            det_records.append({
                "bbox_xyxy": [float(v) for v in box_img_xyxy],
                "category": r["category"],
                "category_zh": r["category_zh"],
                "score": r["score"],
            })

    # 整图画框 + JSON
    if visualize:
        vis = draw_boxes(
            img_bgr,
            boxes_xyxy=[r["bbox_xyxy"] for r in det_records],
            labels=[f"{r['category']} {r['score']:.2f}" for r in det_records],
            colors=[CLASS_COLOR.get(r["category"], DEFAULT_COLOR) for r in det_records],
        )
        cv2.imwrite(os.path.join(out_dir, f"{stem}_vis.jpg"), vis)

    meta = {
        "task": "weld_defect",
        "source": str(image_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": {"conf": float(conf), "overrate": float(overrate),
                   "normalize_tiles": bool(normalize_tiles)},
        "result": {
            "image_size": [int(W0), int(H0)],
            "n_detections": len(det_records),
            "detections": det_records,
        },
    }
    with open(os.path.join(out_dir, f"{stem}_result.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] {Path(image_path).name}: {len(det_records)} 处缺陷, "
          f"用时 {time.time() - time_start:.2f}s")
    return meta


def build_parser():
    p = argparse.ArgumentParser(
        description="焊缝外观缺陷检测（两段式：区域定位 + 缺陷检测）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", default="samples",
                   help="输入：单张图片路径，或包含图片的目录")
    p.add_argument("-o", "--output", default="outputs", help="结果输出目录")
    p.add_argument("-w", "--weights-dir", default=str(BASE_DIR / "weights"),
                   help="权重目录（需含 weld_area.pt 与 liangdutongyi.pt）")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--overrate", type=float, default=0.0,
                   help="瓦片横向重叠率 0~0.9；大图漏检可调至 0.2")
    p.add_argument("--no-preprocess", action="store_true",
                   help="关闭亮度归一化预处理（默认开启）")
    p.add_argument("--save-tiles", action="store_true",
                   help="保存中间瓦片图到 outputs/_tiles（调试用）")
    p.add_argument("--no-visual", action="store_true", help="不生成画框图，只输出 JSON")
    p.add_argument("--device", default=None,
                   help="推理设备，如 cpu / cuda / 0；默认自动")
    return p


def main():
    args = build_parser().parse_args()
    wdir = Path(args.weights_dir)
    area_path = wdir / "weld_area.pt"
    defect_path = wdir / "liangdutongyi.pt"
    for p in (area_path, defect_path):
        if not p.is_file():
            print(f"[ERROR] 缺少权重文件: {p}")
            print("        请确认 weights 目录下有 weld_area.pt 与 liangdutongyi.pt")
            sys.exit(1)

    print("[INFO] 加载模型 ...")
    area_model = YOLO(str(area_path))
    defect_model = YOLO(str(defect_path))
    # 预热
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    kw = {"verbose": False}
    if args.device:
        kw["device"] = args.device
    area_model.predict(dummy, **kw)
    defect_model.predict(dummy, **kw)
    print("[INFO] 模型加载并预热完成")

    src = Path(args.input)
    if src.is_file():
        image_files = [src]
    elif src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = sorted(f for f in src.iterdir() if f.suffix.lower() in exts)
    else:
        print(f"[ERROR] 输入路径不存在: {src}")
        sys.exit(1)
    if not image_files:
        print(f"[WARN] {src} 下没有找到图片")
        return

    print(f"[INFO] 共 {len(image_files)} 张图，输出目录: {os.path.abspath(args.output)}")
    summary = []
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {img_path.name}")
        meta = run_pipeline(
            image_path=img_path, area_model=area_model, defect_model=defect_model,
            out_dir=args.output, overrate=args.overrate, conf=args.conf,
            visualize=not args.no_visual, normalize_tiles=not args.no_preprocess,
            save_tiles=args.save_tiles, device=args.device,
        )
        if meta:
            summary.append({
                "image": str(img_path),
                "n_detections": meta["result"]["n_detections"],
            })

    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"total_images": len(summary),
                   "total_detections": sum(s["n_detections"] for s in summary),
                   "items": summary}, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] 结果已保存至: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
