#!/usr/bin/env python3
"""R15d: 官方指标天花板前沿 —— 在 keep 率固定时，排序质量再好能值多少分。

官方（01_赛事规则与评分标准.md §3）:
    F_i = 2 S_IoU,i / (N_pred,i + N_gt,i)
记每视频总帧 N, GT 密度 d = N_gt/N, 提交密度 p = N_pred/N,
    rho(p) = 匹配到的 GT 帧 / GT 帧总量  （= 排序器的 recall）
    Rbar   = 匹配帧上 bbox IoU 的均值     （= 空间轴质量）
则  S_IoU = rho * N_gt * Rbar,   F(p) = 100 * 2 * rho(p) * Rbar * d / (p + d).

实测 6 个官方点 (p, score)，全部 bbox 逐字节复制自同一份 B3:
    0.5451 -> 32.57 ; 0.8534 -> 35.52 ; 0.9416 -> 35.22
    0.9550 -> 35.11 ; 0.9624 -> 35.06 ; 1.0000 -> 34.75
关键: p=1.0000 那条【必然】rho=1（提交全帧，GT 帧全部被覆盖），
      所以 200 * Rbar * d / (1 + d) = 34.75  —— 一条方程，两个未知量。
"""
import numpy as np

OBS = [(0.5451, 32.57), (0.8534, 35.52), (0.9416, 35.22),
       (0.9550, 35.11), (0.9624, 35.06), (1.0000, 34.75)]
F_ANCHOR = 34.75          # p=1 实测，rho≡1
P_NOW = 0.8534            # 现役 A025

# ---- 约束: 200*Rbar*d/(1+d) = F_ANCHOR  =>  Rbar(d) = F_ANCHOR*(1+d)/(200*d)
def Rbar_of_d(d):
    return F_ANCHOR * (1.0 + d) / (200.0 * d)

def rho_of_obs(d, p, score):
    """由实测分数反解该工作点的 rho"""
    return score * (p + d) / (200.0 * Rbar_of_d(d) * d)

print("=" * 100)
print("表 1  (d, Rbar) 自由度：6 个实测点无法区分，Rbar<=1 只给 d 一个下界")
print("=" * 100)
print(f"{'d':>8}{'Rbar(d)':>10}{'rho@0.545':>11}{'rho@0.853':>11}{'rho@1':>8}"
      f"{'天花板@keep0.853':>17}{'100*Rbar(全局上限)':>20}")
print("-" * 100)
for d in [0.21029, 0.25, 0.2847, 0.35, 0.45, 0.60, 0.80, 1.00, 1.50, 2.00]:
    Rb = Rbar_of_d(d)
    if Rb > 1.0000001:
        print(f"{d:8.5f}{Rb:10.4f}   —— Rbar>1 非法（d 的下界就是这里）")
        continue
    r1 = rho_of_obs(d, 0.5451, 32.57)
    r2 = rho_of_obs(d, 0.8534, 35.52)
    ceil_now = 100 * 2 * 1.0 * Rb * d / (P_NOW + d)     # rho=1 时的分数
    print(f"{d:8.5f}{Rb:10.4f}{r1:11.4f}{r2:11.4f}{1.0:8.4f}{ceil_now:17.2f}{100*Rb:20.1f}")

d_lo = F_ANCHOR / (200.0 - F_ANCHOR)
print(f"\n  d 的硬下界 d >= {d_lo:.5f}  （由 Rbar<=1 给出）")
print(f"  天花板公式: F_ceil(p) = F_ANCHOR * (1+d) / (p+d)，只依赖 d，不依赖 Rbar 的绝对值")
print(f"  在 p={P_NOW} 处:  d={d_lo:.4f} -> {F_ANCHOR*(1+d_lo)/(P_NOW+d_lo):.2f} ; "
      f"d=0.30 -> {F_ANCHOR*1.30/(P_NOW+0.30):.2f} ; d=0.50 -> {F_ANCHOR*1.50/(P_NOW+0.50):.2f} ; "
      f"d=1.00 -> {F_ANCHOR*2.0/(P_NOW+1.0):.2f}")

print()
print("=" * 100)
print("表 2  结论 A: 现役 keep=0.8534 处的【排序天花板】几乎与 d 无关 ~39 分")
print("=" * 100)
print(f"{'d':>9}{'当前分':>9}{'天花板(rho→1)':>15}{'可涨':>9}"
      f"{'   ---- 达到 40 / 42 / 45 需要 keep 降到 ----':>48}")
print("-" * 100)
for d in [d_lo, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]:
    ceil_now = F_ANCHOR * (1 + d) / (P_NOW + d)
    row = f"{d:9.4f}{35.52:9.2f}{ceil_now:15.2f}{ceil_now-35.52:+9.2f}     "
    for T in (40.0, 42.0, 45.0):
        # 解 F_ANCHOR*(1+d)/(p+d) = T  =>  p = F_ANCHOR*(1+d)/T - d
        p_need = F_ANCHOR * (1 + d) / T - d
        row += f"  {T:.0f}: p<={p_need:.3f}" if p_need > 0 else f"  {T:.0f}: 无解   "
    print(row)
print("\n  读法: 一行内 d 越大，同样的 keep 率天花板越低；要给 40 分留出空间，")
print("        必须把 p 压到表中该行的阈值以下 —— 而这要求排序器在低 p 处仍然准。")

print()
print("=" * 100)
print("表 3  结论 B: 低 p 处我们离天花板极远 —— 真正的失血点")
print("=" * 100)
print(f"{'d':>9}{'p':>8}{'实测':>8}{'反解 rho':>10}{'lift=rho/p':>12}"
      f"{'天花板':>9}{'达成率':>9}{'可涨':>9}")
print("-" * 100)
for d in [d_lo, 0.2847, 0.40]:
    for (p, s) in OBS:
        Rb = Rbar_of_d(d)
        if Rb > 1.0000001:
            continue
        rho = rho_of_obs(d, p, s)
        ceil = F_ANCHOR * (1 + d) / (p + d)
        print(f"{d:9.4f}{p:8.4f}{s:8.2f}{rho:10.4f}{rho/p:12.4f}{ceil:9.2f}"
              f"{100*s/ceil:8.1f}%{ceil-s:+9.2f}")
    print()

print("=" * 100)
print("表 4  结论 C: 排序器要提升到什么水平，低 p 才值得走")
print("=" * 100)
print("以 d=0.2103（最有利世界）为例：F(p)=rho(p)*F_ANCHOR*(1+d)/(p+d)")
for d in [d_lo, 0.2847]:
    K = F_ANCHOR * (1 + d)
    print(f"\n  d={d:.4f}  则 F(p) = rho(p) * {K:.3f} / (p + {d:.4f})"
          f"   超越现役 35.52 所需的 rho:")
    print(f"    {'p':>8}{'需要 rho':>10}{'= lift':>9}{'现役 rho(p) 反解':>18}{'缺口':>9}")
    for p in [0.75, 0.65, 0.5451, 0.45, 0.35, 0.25, 0.2103]:
        need = 35.52 * (p + d) / K
        cur = rho_of_obs(d, p, dict(OBS).get(p, np.nan)) if p in dict(OBS) else np.nan
        cur_s = f"{cur:18.4f}" if not np.isnan(cur) else f"{'未测':>18}"
        gap_s = f"{need-cur:+9.4f}" if not np.isnan(cur) else f"{'—':>9}"
        print(f"    {p:8.4f}{need:10.4f}{need/p:9.4f}{cur_s}{gap_s}")
    print("    对照: 随机选择器的 rho≡p，即 lift≡1.0。lift 1.05 == 与随机几乎无区别。")
