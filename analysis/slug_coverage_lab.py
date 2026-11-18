#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slug_coverage_lab.py — 小字号尖角覆盖率实验台（针对 Slug @ be3c13e 的漏光问题）

比较以下覆盖率估计方法在固定 glyph 集上的误差与成本：
  slug_f64 / slug_f32 / slug_f16tex : SlugPixelShader.hlsl 的忠实向量化移植
                                      （逐像素解析光线投射 + 盒滤波权重，即 "analytic" 路径；
                                       band 结构只是精确裁剪加速，省略不影响结果）
  slug_2x2                          : Slug 解析路径的 2x2 shader 超采样（候选修复，不改字形语义）
  sdf                               : 逐像素有向距离 → 盒映射覆盖率（几何着色器常用做法）
  cpu_4x4                           : CPU 4x4=16 点子像素超采样（离线图集方案）
  msaa4 / msaa8                     : D3D11 标准 4x/8x MSAA 采样 pattern（几何光栅化方案）
  ref64                             : 64x64 超采样参考（并用精确面积法交叉校验）

用法：
  python3 slug_coverage_lab.py            # 全量运行，写 analysis/results/
  python3 slug_coverage_lab.py --verify   # 全量运行并对照 GOLDEN 阈值，回归时退出码非零

确定性：无随机数；所有 fixture 内联定义；字形轮廓数据全程不变（不改变字形语义）。
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np

np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
IMG = os.path.join(OUT, "boundary_images")

SIZES = [8, 10, 12, 16]
PHASES = [((0.0, 0.0), "p00"), ((0.5, 0.5), "p55")]
PHASE_SWEEP = {"A", "V", "C", "wedge30"}   # 这些 glyph 加跑 (0.5,0.5) 相位
GAMMA = 2.2

# =============================================================================
# 固定 glyph fixtures（em 空间，y 向上；直线按 README 建议编码为 p2=p3 的二次贝塞尔）
# =============================================================================

def P(*pts):
    return [(float(x), float(y)) for x, y in pts]

# (轮廓点列表, 是否孔洞)。外轮廓自动定向为 CCW，孔洞 CW（nonzero 规则）。
LETTERS = {
    "A": [(P((0.15, 0.10), (0.85, 0.10), (0.50, 1.05)), False),
          (P((0.36, 0.34), (0.64, 0.34), (0.50, 0.76)), True)],
    "V": [(P((0.15, 1.05), (0.37, 1.05), (0.50, 0.28), (0.63, 1.05),
             (0.85, 1.05), (0.50, 0.02)), False)],
    "W": [(P((0.08, 1.05), (0.24, 1.05), (0.33, 0.32), (0.46, 0.88), (0.54, 0.88),
             (0.67, 0.32), (0.76, 1.05), (0.92, 1.05), (0.75, 0.10), (0.61, 0.10),
             (0.50, 0.62), (0.39, 0.10), (0.25, 0.10)), False)],
    "K": [(P((0.15, 0.10), (0.15, 1.05), (0.30, 1.05), (0.30, 0.62), (0.62, 1.05),
             (0.80, 1.05), (0.42, 0.55), (0.82, 0.10), (0.62, 0.10), (0.30, 0.42),
             (0.30, 0.10)), False)],
    "M": [(P((0.12, 0.10), (0.12, 1.05), (0.88, 1.05), (0.88, 0.10), (0.72, 0.10),
             (0.72, 0.70), (0.50, 0.25), (0.28, 0.70), (0.28, 0.10)), False)],
    "4": [(P((0.55, 0.10), (0.72, 0.10), (0.72, 1.05), (0.58, 1.05), (0.15, 0.48),
             (0.15, 0.35), (0.55, 0.35)), False)],
}

def _arc_seg(c, r, a0, a1):
    """二次贝塞尔逼近圆弧 a0→a1（度）。"""
    t0, t1 = math.radians(a0), math.radians(a1)
    mid = 0.5 * (t0 + t1)
    rc = r / math.cos(0.5 * (t1 - t0))
    return ((c[0] + r * math.cos(t0), c[1] + r * math.sin(t0)),
            (c[0] + rc * math.cos(mid), c[1] + rc * math.sin(mid)),
            (c[0] + r * math.cos(t1), c[1] + r * math.sin(t1)))

def make_C():
    """C 形：外弧 r=0.45 / 内弧 r=0.28，圆心 (0.55,0.575)，右侧开口。含真实二次曲线。"""
    c, ro, ri = (0.55, 0.575), 0.45, 0.28
    curves = [_arc_seg(c, ro, 60, 180), _arc_seg(c, ro, 180, 300)]
    p_in300 = _arc_seg(c, ri, 300, 180)[0]
    curves.append((curves[-1][2], p_in300, p_in300))               # 端帽直线
    curves += [_arc_seg(c, ri, 300, 180), _arc_seg(c, ri, 180, 60)]
    curves.append((curves[-1][2], curves[0][0], curves[0][0]))     # 端帽直线
    return curves

def make_wedge(deg):
    """尖角三角形：顶角 deg，顶点 (0.5, 1.1) 朝上，高 0.8em。"""
    half = math.tan(math.radians(deg / 2.0)) * 0.8
    apex, bl, br = (0.5, 1.1), (0.5 - half, 0.3), (0.5 + half, 0.3)
    return [(bl, br, br), (br, apex, apex), (apex, bl, bl)]

def poly_to_curves(pts):
    n = len(pts)
    return [(pts[i], pts[(i + 1) % n], pts[(i + 1) % n]) for i in range(n)]

def shoelace(pts):
    n = len(pts)
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                     for i in range(n))

def split_rings(curves):
    """按端点连接关系把曲线列表切成闭合环（fixture 保证每环闭合）。"""
    rings, cur, start = [], [], None
    for c in curves:
        if not cur:
            cur, start = [c], c[0]
        elif cur[-1][2] == c[0]:
            cur.append(c)
        else:
            rings.append(cur)
            cur, start = [c], c[0]
        if cur[-1][2] == start:
            rings.append(cur)
            cur, start = [], None
    if cur:
        rings.append(cur)
    return rings

def build_fixtures(px):
    """{name: [curve,...]}，curve=(p1,p2,p3)，em 空间。外轮廓 CCW、孔 CW。"""
    fx = {}
    for name, contours in LETTERS.items():
        curves = []
        for pts, hole in contours:
            a = shoelace(pts)
            if (a < 0) != hole:
                pts = pts[::-1]
            curves += poly_to_curves(pts)
        fx[name] = curves
    fx["C"] = make_C()
    for d in (15, 30, 45, 60, 90):
        fx["wedge%d" % d] = make_wedge(d)
    for wp in (0.5, 1.0, 1.5):       # 线宽 fixture：宽度 = wp 物理像素，故意偏离像素网格
        w = wp / px
        oy, ox = 0.50 + 0.37 / px, 0.60 + 0.19 / px
        fx["strokeH_%g" % wp] = poly_to_curves(P((0.15, oy), (1.25, oy),
                                                 (1.25, oy + w), (0.15, oy + w)))
        fx["strokeV_%g" % wp] = poly_to_curves(P((ox, 0.15), (ox + w, 0.15),
                                                 (ox + w, 1.25), (ox, 1.25)))
    return fx

def fixture_vertices():
    """尖角顶点（em 空间）：多边形fixture中内角 ≤90° 的顶点 + wedge 顶点。"""
    verts = []
    for name, contours in LETTERS.items():
        for pts, hole in contours:
            n = len(pts)
            for i in range(n):
                a = np.subtract(pts[i - 1], pts[i])
                b = np.subtract(pts[(i + 1) % n], pts[i])
                ca = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                ang = math.degrees(math.acos(max(-1.0, min(1.0, ca))))
                if ang <= 90.0:
                    verts.append((name, pts[i], round(ang, 1)))
    for d in (15, 30, 45, 60, 90):
        verts.append(("wedge%d" % d, (0.5, 1.1), float(d)))
    return verts

# =============================================================================
# 几何工具：展平、绕数、精确面积、SDF
# =============================================================================

def flatten_ring(curves, tol_em):
    """自适应展平一个闭合环为点列（不重复末点）。"""
    pts = []
    for p1, p2, p3 in curves:
        stack = [(np.array(p1), np.array(p2), np.array(p3), 0)]
        first = True
        while stack:
            a, b, c, depth = stack.pop()
            chord = c - a
            L = float(np.hypot(chord[0], chord[1]))
            if L > 1e-18:
                flat = abs(chord[0] * (b[1] - a[1]) - chord[1] * (b[0] - a[0])) / L
            else:
                flat = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            if flat <= tol_em or depth >= 16:
                if first:
                    pts.append(tuple(a))
                    first = False
                pts.append(tuple(c))
            else:
                ab, bc = 0.5 * (a + b), 0.5 * (b + c)
                m = 0.5 * (ab + bc)
                stack.append((m, bc, c, depth + 1))
                stack.append((a, ab, m, depth + 1))
    if len(pts) > 1 and pts[-1] == pts[0]:
        pts.pop()
    return pts

def winding_counts(px, py, segs):
    """vectorized nonzero 绕数。px,py: (N,) float；segs: [(x1,y1,x2,y2),...]。"""
    wn = np.zeros(px.shape, dtype=np.int32)
    for x1, y1, x2, y2 in segs:
        up = (y1 <= py) & (y2 > py)
        dn = (y2 <= py) & (y1 > py)
        cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
        wn += np.where(up & (cross > 0), 1, 0)
        wn -= np.where(dn & (cross < 0), 1, 0)
    return wn

def sample_coverage(segs_flat, W, H, offsets):
    """在 offsets=[(dx,dy),...]（像素内 [0,1) 坐标）点采样，返回 (H,W) 覆盖率。"""
    yy, xx = np.mgrid[0:H, 0:W]
    xx = xx.ravel()[None, :]
    yy = yy.ravel()[None, :]
    off = np.array(offsets)
    px = (xx + off[:, 0:1]).ravel()
    py = (yy + off[:, 1:2]).ravel()
    inside = winding_counts(px, py, segs_flat) != 0
    return inside.reshape(len(offsets), H, W).mean(axis=0)

def exact_cell_coverage(polys_px, i, j):
    """Sutherland–Hodgman 裁剪 + 鞋带公式：单元 (i,j) 的精确有符号面积覆盖率。"""
    def clip(poly, axis, keep_gt, v):
        out = []
        n = len(poly)
        for k in range(n):
            p, q = poly[k - 1], poly[k]
            pin = (p[axis] >= v) if keep_gt else (p[axis] <= v)
            qin = (q[axis] >= v) if keep_gt else (q[axis] <= v)
            if pin != qin:
                t = (v - p[axis]) / (q[axis] - p[axis])
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            if qin:
                out.append(q)
        return out
    total = 0.0
    for poly in polys_px:
        shifted = [(x - i, y - j) for x, y in poly]
        for axis, keep_gt, v in ((0, True, 0.0), (0, False, 1.0), (1, True, 0.0), (1, False, 1.0)):
            shifted = clip(shifted, axis, keep_gt, v)
            if not shifted:
                break
        if shifted:
            total += shoelace(shifted)
    return min(1.0, max(0.0, total))

def sdf_coverage(polys_px, W, H):
    """逐像素有向距离（像素中心）→ 盒映射覆盖率。"""
    yy, xx = np.mgrid[0:H, 0:W]
    px = (xx + 0.5).ravel()[None, :].astype(np.float64)
    py = (yy + 0.5).ravel()[None, :].astype(np.float64)
    d2 = np.full(px.shape, 1e18)
    segs_flat = []
    for poly in polys_px:
        n = len(poly)
        for k in range(n):
            x1, y1 = poly[k - 1]
            x2, y2 = poly[k]
            segs_flat.append((x1, y1, x2, y2))
            ex, ey = x2 - x1, y2 - y1
            L2 = ex * ex + ey * ey
            t = np.clip(((px - x1) * ex + (py - y1) * ey) / (L2 + 1e-30), 0.0, 1.0)
            dx = px - (x1 + t * ex)
            dy = py - (y1 + t * ey)
            d2 = np.minimum(d2, dx * dx + dy * dy)
    inside = winding_counts(px.ravel(), py.ravel(), segs_flat).reshape(px.shape) != 0
    signed = np.where(inside, -np.sqrt(d2), np.sqrt(d2))
    return np.clip(0.5 - signed, 0.0, 1.0).reshape(H, W)

# =============================================================================
# SlugPixelShader.hlsl 向量化移植（单样本解析覆盖率）
# =============================================================================

def slug_map(curves, x0, y0, W, H, ppe, ph, prec="f64", ss=1):
    """(H,W) 覆盖率。ss=2 时做 2x2 shader 超采样（子样本盒宽减半，ppe×2）。"""
    dt = np.float32 if prec in ("f32", "f16tex") else np.float64
    yy, xx = np.mgrid[0:H, 0:W]
    subs = [(0.5, 0.5)] if ss == 1 else [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    acc = np.zeros((H, W), np.float64)
    roots_total = 0
    q = (lambda v: float(np.float16(v))) if prec == "f16tex" else (lambda v: v)
    for dx, dy in subs:
        rcx = (x0 + (xx + dx + ph[0]) / ppe).astype(dt)
        rcy = (y0 + (yy + dy + ph[1]) / ppe).astype(dt)
        ppe_s = dt(ppe * ss)
        xcov = np.zeros((H, W), dt)
        ycov = np.zeros((H, W), dt)
        xwgt = np.zeros((H, W), dt)
        ywgt = np.zeros((H, W), dt)
        eps = dt(1.0 / 65536.0)
        for c1, c2, c3 in curves:
            x1 = dt(q(c1[0])) - rcx; y1 = dt(q(c1[1])) - rcy
            x2 = dt(q(c2[0])) - rcx; y2 = dt(q(c2[1])) - rcy
            x3 = dt(q(c3[0])) - rcx; y3 = dt(q(c3[1])) - rcy
            # ---- 水平光线（CalcRootCode + SolveHorizPoly）----
            s = ((y1 < 0).astype(np.uint8) | ((y2 < 0).astype(np.uint8) << 1)
                 | ((y3 < 0).astype(np.uint8) << 2))
            code = (0x2E74 >> s.astype(np.int32)) & 0x0101
            if (code != 0).any():
                ax = x1 - 2 * x2 + x3; ay = y1 - 2 * y2 + y3
                bx = x1 - x2;        by = y1 - y2
                d = np.sqrt(np.maximum(by * by - ay * y1, dt(0)))
                lin = np.abs(ay) < eps
                ra = np.where(lin, dt(0), dt(1) / np.where(ay == 0, dt(1e-30), ay))
                t1 = np.where(lin, y1 * dt(0.5) / np.where(by == 0, dt(1e-30), by), (by - d) * ra)
                t2 = np.where(lin, t1, (by + d) * ra)
                rx = ((ax * t1 - 2 * bx) * t1 + x1) * ppe_s
                ry = ((ax * t2 - 2 * bx) * t2 + x1) * ppe_s
                m1 = (code & 1) != 0
                m2 = code > 1
                sat = lambda v: np.minimum(1.0, np.maximum(0.0, v))
                xcov += np.where(m1, sat(rx + 0.5), dt(0)) - np.where(m2, sat(ry + 0.5), dt(0))
                xwgt = np.maximum(xwgt, np.where(m1, sat(1.0 - 2 * np.abs(rx)), dt(0)))
                xwgt = np.maximum(xwgt, np.where(m2, sat(1.0 - 2 * np.abs(ry)), dt(0)))
                roots_total += int((code != 0).sum())
            # ---- 垂直光线（SolveVertPoly）----
            s = ((x1 < 0).astype(np.uint8) | ((x2 < 0).astype(np.uint8) << 1)
                 | ((x3 < 0).astype(np.uint8) << 2))
            code = (0x2E74 >> s.astype(np.int32)) & 0x0101
            if (code != 0).any():
                ax = x1 - 2 * x2 + x3; ay = y1 - 2 * y2 + y3
                bx = x1 - x2;        by = y1 - y2
                d = np.sqrt(np.maximum(bx * bx - ax * x1, dt(0)))
                lin = np.abs(ax) < eps
                ra = np.where(lin, dt(0), dt(1) / np.where(ax == 0, dt(1e-30), ax))
                t1 = np.where(lin, x1 * dt(0.5) / np.where(bx == 0, dt(1e-30), bx), (bx - d) * ra)
                t2 = np.where(lin, t1, (bx + d) * ra)
                ry1 = ((ay * t1 - 2 * by) * t1 + y1) * ppe_s
                ry2 = ((ay * t2 - 2 * by) * t2 + y1) * ppe_s
                m1 = (code & 1) != 0
                m2 = code > 1
                sat = lambda v: np.minimum(1.0, np.maximum(0.0, v))
                ycov += np.where(m2, sat(ry2 + 0.5), dt(0)) - np.where(m1, sat(ry1 + 0.5), dt(0))
                ywgt = np.maximum(ywgt, np.where(m1, sat(1.0 - 2 * np.abs(ry1)), dt(0)))
                ywgt = np.maximum(ywgt, np.where(m2, sat(1.0 - 2 * np.abs(ry2)), dt(0)))
                roots_total += int((code != 0).sum())
        cov = np.maximum(np.abs(xcov * xwgt + ycov * ywgt) / np.maximum(xwgt + ywgt, 1.0 / 65536.0),
                         np.minimum(np.abs(xcov), np.abs(ycov)))
        acc += np.minimum(1.0, np.maximum(0.0, cov))
    return acc / len(subs), roots_total

# =============================================================================
# MSAA 标准采样 pattern（D3D11，1/16 像素网格）
# =============================================================================
MSAA4 = [(6 / 16, 2 / 16), (14 / 16, 6 / 16), (2 / 16, 10 / 16), (10 / 16, 14 / 16)]
MSAA8 = [(9 / 16, 5 / 16), (7 / 16, 11 / 16), (13 / 16, 9 / 16), (5 / 16, 3 / 16),
         (3 / 16, 13 / 16), (11 / 16, 15 / 16), (15 / 16, 7 / 16), (1 / 16, 1 / 16)]
CPU4X4 = [((k % 4 + 0.5) / 4, (k // 4 + 0.5) / 4) for k in range(16)]
REF64 = [((k % 64 + 0.5) / 64, (k // 64 + 0.5) / 64) for k in range(4096)]

METHODS = ["slug_f64", "slug_f32", "slug_f16tex", "slug_2x2", "sdf",
           "cpu_4x4", "msaa4", "msaa8"]

def srgb_encode(v):
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)

def run():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(IMG, exist_ok=True)
    corner_verts = fixture_vertices()
    rows, boundary, wedge_rows, stroke_rows = [], [], [], []
    maps_for_png = {}
    t0 = time.perf_counter()
    xchk_max = 0.0

    for px in SIZES:
        for name, curves in build_fixtures(px).items():
            is_stroke = name.startswith("stroke")
            phases = PHASES if (name in PHASE_SWEEP and not is_stroke) else PHASES[:1]
            xs = [p[0] for c in curves for p in c]
            ys = [p[1] for c in curves for p in c]
            x0 = math.floor(min(xs) * px - 0.1 * px) / px
            y0 = math.floor(min(ys) * px - 0.1 * px) / px
            x1 = math.ceil(max(xs) * px + 0.1 * px) / px
            y1 = math.ceil(max(ys) * px + 0.1 * px) / px
            W, H = int(round((x1 - x0) * px)), int(round((y1 - y0) * px))
            rings = split_rings(curves)
            polys_em = [flatten_ring(r, 1e-3 / px) for r in rings]
            for ph, phn in phases:
                polys_px = [[((x - x0) * px - ph[0], (y - y0) * px - ph[1]) for x, y in poly]
                            for poly in polys_em]
                segs_flat = [(p[0], p[1], q[0], q[1])
                             for poly in polys_px for p, q in
                             zip(poly, poly[1:] + poly[:1])]
                ref = sample_coverage(segs_flat, W, H, REF64)
                # 精确面积交叉校验（1/4 单元抽样）
                for j in range(0, H, 2):
                    for i in range(0, W, 2):
                        e = abs(exact_cell_coverage(polys_px, i, j) - ref[j, i])
                        xchk_max = max(xchk_max, e)
                results, costs = {}, {}
                for meth, prec, ss in (("slug_f64", "f64", 1), ("slug_f32", "f32", 1),
                                       ("slug_f16tex", "f16tex", 1), ("slug_2x2", "f32", 2)):
                    t = time.perf_counter()
                    results[meth], rts = slug_map(curves, x0, y0, W, H, px, ph, prec, ss)
                    costs[meth] = (time.perf_counter() - t, rts)
                t = time.perf_counter()
                results["sdf"] = sdf_coverage(polys_px, W, H)
                costs["sdf"] = (time.perf_counter() - t, len(segs_flat) * W * H)
                for meth, offs, ns in (("cpu_4x4", CPU4X4, 16), ("msaa4", MSAA4, 4),
                                       ("msaa8", MSAA8, 8)):
                    t = time.perf_counter()
                    results[meth] = sample_coverage(segs_flat, W, H, offs)
                    costs[meth] = (time.perf_counter() - t, ns * W * H)

                ink = max(float(ref.sum()), 1e-9)
                cmask = np.zeros((H, W), bool)
                for gname, (vx, vy), ang in corner_verts:
                    if gname != name:
                        continue
                    cx = (vx - x0) * px - ph[0]
                    cy = (vy - y0) * px - ph[1]
                    jj, ii = np.mgrid[0:H, 0:W]
                    cmask |= (ii + 0.5 - cx) ** 2 + (jj + 0.5 - cy) ** 2 <= 1.0
                refg = np.power(np.clip(ref, 0, 1), 1 / GAMMA)
                for meth, est in results.items():
                    err = est - ref
                    ae = np.abs(err)
                    gerr = np.abs(np.power(np.clip(est, 0, 1), 1 / GAMMA) - refg)
                    rows.append({
                        "glyph": name, "px": px, "phase": phn, "method": meth,
                        "W": W, "H": H,
                        "max_abs": round(float(ae.max()), 5),
                        "mean_abs": round(float(ae.mean()), 5),
                        "rmse": round(float(np.sqrt((err ** 2).mean())), 5),
                        "leak": round(float(np.maximum(-err, 0).sum() / ink), 5),
                        "bleed": round(float(np.maximum(err, 0).sum() / ink), 5),
                        "corner_max_abs": round(float(ae[cmask].max()), 5) if cmask.any() else "",
                        "corner_signed_mean": round(float(err[cmask].mean()), 5) if cmask.any() else "",
                        "gamma_mean_abs": round(float(gerr.mean()), 5),
                        "n_cells": W * H,
                        "time_ms": round(costs[meth][0] * 1e3, 2),
                        "work": int(costs[meth][1]),
                    })
                    ij = np.argwhere(ae > 0.1)
                    for j, i in ij:
                        boundary.append({
                            "glyph": name, "px": px, "phase": phn,
                            "cell": [int(i), int(j)], "method": meth,
                            "ref": round(float(ref[j, i]), 4),
                            "est": round(float(est[j, i]), 4),
                            "signed_err": round(float(err[j, i]), 4),
                            "corner": bool(cmask[j, i]),
                        })
                if name.startswith("wedge"):
                    ai = int((0.5 - x0) * px - ph[0])
                    aj = int((1.1 - y0) * px - ph[1])
                    wedge_rows.append({
                        "glyph": name, "px": px, "phase": phn, "apex_cell": [ai, aj],
                        "ref": round(float(ref[aj, ai]), 4),
                        **{m: round(float(r[aj, ai]), 4) for m, r in results.items()}})
                if is_stroke:
                    nominal = float(name.split("_")[1])
                    for meth, est in results.items():
                        if name.startswith("strokeH"):
                            prof = float(est[:, W // 4:3 * W // 4].sum(axis=0).mean())
                        else:
                            prof = float(est[H // 4:3 * H // 4, :].sum(axis=1).mean())
                        stroke_rows.append({"glyph": name, "px": px, "method": meth,
                                            "nominal_px": nominal,
                                            "measured_px": round(prof, 4),
                                            "width_err_pct": round(100 * (prof - nominal) / nominal, 2)})
                lk = float(np.maximum(ref - results["slug_f32"], 0).sum() / ink)
                maps_for_png[(name, px, phn)] = (lk, ref, results)

    # =========================================================================
    # 输出
    # =========================================================================
    def dump_csv(fn, rows_):
        with open(os.path.join(OUT, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0].keys()))
            w.writeheader()
            w.writerows(rows_)

    dump_csv("coverage_stats.csv", rows)
    boundary.sort(key=lambda b: -abs(b["signed_err"]))
    dump_csv("boundary_samples.csv", boundary[:300])
    dump_csv("wedge_stress.csv", wedge_rows)
    dump_csv("stroke_width.csv", stroke_rows)

    agg = {}
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        agg[m] = {
            "worst_max_abs": round(max(r["max_abs"] for r in rs), 5),
            "mean_mean_abs": round(sum(r["mean_abs"] for r in rs) / len(rs), 5),
            "mean_rmse": round(sum(r["rmse"] for r in rs) / len(rs), 5),
            "mean_leak": round(sum(r["leak"] for r in rs) / len(rs), 5),
            "worst_corner_max_abs": round(max(r["corner_max_abs"] for r in rs
                                              if r["corner_max_abs"] != ""), 5),
            "mean_gamma_err": round(sum(r["gamma_mean_abs"] for r in rs) / len(rs), 5),
            "total_time_ms": round(sum(r["time_ms"] for r in rs), 1),
        }
    per_size = {m: {px: round(float(np.mean([r["mean_abs"] for r in rows
                                             if r["method"] == m and r["px"] == px])), 5)
                    for px in SIZES} for m in METHODS}
    with open(os.path.join(OUT, "aggregate.json"), "w") as f:
        json.dump({"per_method": agg, "mean_abs_by_size": per_size,
                   "ref_exact_area_crosscheck_max_abs": round(xchk_max, 5)}, f, indent=1)

    c = np.linspace(0, 1, 1001)
    gamma_json = {
        "gamma": GAMMA,
        "gamma_roundtrip_residual_max": float(np.max(np.abs(
            srgb_encode(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)) - c))),
        "slug_weight_sqrt_vs_gamma_encode": {
            "mean_abs": float(np.mean(np.abs(np.sqrt(c) - c ** (1 / GAMMA)))),
            "max_abs": float(np.max(np.abs(np.sqrt(c) - c ** (1 / GAMMA)))),
            "note": "SLUG_WEIGHT 的 sqrt() 与 1/2.2 gamma 编码的偏差；它是有意的光学加粗，不是线性光操作",
        },
    }
    with open(os.path.join(OUT, "gamma_check.json"), "w") as f:
        json.dump(gamma_json, f, indent=1)

    try:
        from PIL import Image
        worst = sorted(maps_for_png.items(), key=lambda kv: -kv[1][0])[:4]
        for (name, px, phn), (lk, ref, results) in worst:
            order = ["slug_f32", "slug_2x2", "sdf", "cpu_4x4", "msaa4", "msaa8"]
            panels = [ref] + [results[m] for m in order]
            sc = 12
            Hh, Ww = ref.shape
            canvas = np.zeros((Hh * sc, (Ww * sc + 2) * len(panels), 3), np.uint8)
            for k, panel in enumerate(panels):
                up = np.kron(panel, np.ones((sc, sc)))
                g = (np.clip(up, 0, 1) * 255).astype(np.uint8)
                x0p = k * (Ww * sc + 2)
                canvas[:, x0p:x0p + Ww * sc, :] = g[:, :, None]
            Image.fromarray(canvas[::-1]).save(
                os.path.join(IMG, "boundary_%s_%d_%s.png" % (name, px, phn)))
    except ImportError:
        pass

    print("done in %.1fs; ref exact-area cross-check max|diff|=%.4f"
          % (time.perf_counter() - t0, xchk_max))
    print("wrote %s" % OUT)
    return rows, agg, per_size, wedge_rows, stroke_rows, gamma_json, xchk_max

# =============================================================================
# --verify：固定阈值回归门（可执行验证）
# =============================================================================
GOLDEN = {
    "ref_crosscheck_max": 0.02,        # 64x64 参考 vs 精确面积
    "slug_corner_leak_exists": 0.15,   # 已知 bug 钉住：slug_f32 尖角单元最大误差 > 此值
    "slug2x2_corner_improve": 0.65,    # slug_2x2 尖角最差误差 ≤ slug_f32 × 此比例
    "cpu4x4_max_abs": 0.30,            # 16 级量化 + 网格对齐最坏情形
    "msaa4_worse_than_cpu4x4": 1.5,    # msaa4 平均误差 ≥ cpu_4x4 × 此比例
    "sdf_corner_rounding": 0.10,       # sdf 尖角圆角误差下界（已知现象钉住）
    "stroke_width_err_pct": 5.0,       # slug_2x2/cpu_4x4 线宽误差上限（%）
    "gamma_roundtrip": 1e-9,
    "sqrt_vs_gamma_max": 0.04,
}

def verify():
    rows, agg, per_size, wedge_rows, stroke_rows, gamma_json, xchk = run()
    fails = []

    def chk(ok, msg):
        print(("PASS " if ok else "FAIL ") + msg)
        if not ok:
            fails.append(msg)

    chk(xchk <= GOLDEN["ref_crosscheck_max"],
        "reference cross-check max|ref64-exact|=%.4f <= %.2f" % (xchk, GOLDEN["ref_crosscheck_max"]))
    s32 = max(r["corner_max_abs"] for r in rows
              if r["method"] == "slug_f32" and r["corner_max_abs"] != "")
    chk(s32 > GOLDEN["slug_corner_leak_exists"],
        "known-issue pinned: slug_f32 corner max err %.3f > %.2f"
        % (s32, GOLDEN["slug_corner_leak_exists"]))
    s2x = max(r["corner_max_abs"] for r in rows
              if r["method"] == "slug_2x2" and r["corner_max_abs"] != "")
    chk(s2x <= s32 * GOLDEN["slug2x2_corner_improve"],
        "slug_2x2 corner max err %.3f <= slug_f32 %.3f x %.2f"
        % (s2x, s32, GOLDEN["slug2x2_corner_improve"]))
    c44 = agg["cpu_4x4"]["worst_max_abs"]
    chk(c44 <= GOLDEN["cpu4x4_max_abs"],
        "cpu_4x4 worst max|err| %.3f <= %.2f" % (c44, GOLDEN["cpu4x4_max_abs"]))
    ratio = agg["msaa4"]["mean_mean_abs"] / max(agg["cpu_4x4"]["mean_mean_abs"], 1e-9)
    chk(ratio >= GOLDEN["msaa4_worse_than_cpu4x4"],
        "msaa4 mean err %.4f >= %.1fx cpu_4x4 %.4f"
        % (agg["msaa4"]["mean_mean_abs"], GOLDEN["msaa4_worse_than_cpu4x4"],
           agg["cpu_4x4"]["mean_mean_abs"]))
    sdf_corner = max(r["corner_max_abs"] for r in rows
                     if r["method"] == "sdf" and r["corner_max_abs"] != "")
    chk(sdf_corner > GOLDEN["sdf_corner_rounding"],
        "known-issue pinned: sdf corner rounding %.3f > %.2f"
        % (sdf_corner, GOLDEN["sdf_corner_rounding"]))
    sw = [abs(r["width_err_pct"]) for r in stroke_rows if r["method"] in ("slug_2x2", "cpu_4x4")]
    chk(max(sw) <= GOLDEN["stroke_width_err_pct"],
        "stroke width err (slug_2x2/cpu_4x4) max %.2f%% <= %.1f%%"
        % (max(sw), GOLDEN["stroke_width_err_pct"]))
    chk(gamma_json["gamma_roundtrip_residual_max"] <= GOLDEN["gamma_roundtrip"],
        "gamma roundtrip residual %.2e" % gamma_json["gamma_roundtrip_residual_max"])
    chk(gamma_json["slug_weight_sqrt_vs_gamma_encode"]["max_abs"] <= GOLDEN["sqrt_vs_gamma_max"],
        "sqrt vs 1/2.2 encode max %.4f <= %.2f"
        % (gamma_json["slug_weight_sqrt_vs_gamma_encode"]["max_abs"], GOLDEN["sqrt_vs_gamma_max"]))

    print("\n%d check(s) failed" % len(fails))
    return 1 if fails else 0

if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(verify())
    run()
