"""
3_KarlWeierstrass_NEXT3_2a3d — POGODNI deo iz 1_KarlWeierstrass_v2.py
Aparat 2a: Brownovo kretanje  +  Test 3d: Sample / Permutation entropy

Self-contained:
  - KORAK 1: ucitavanje 4624 izvlacenja i izgradnja f(t) = lex-indeks
  - KORAK 2a: Brown inkrementi (priprema)
  - KORAK 2a3d: Sample entropy + Permutation entropy nad
                centriranim Brown inkrementima, Brown-putanja kao kontrola,
                shuffled entropy referenca

Output:
  3_KarlWeierstrass_NEXT3_2a3d.png
  3_KarlWeierstrass_NEXT3_2a3d.txt
"""

import csv
import math
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np


T0 = time.time()

CSV_DRAWS = "/data/loto7_4624_k43.csv"

HERE = os.path.dirname(os.path.abspath(__file__))
PNG_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT3_2a3d.png")
TXT_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT3_2a3d.txt")

N_MAX = 39
K_PICK = 7
TOTAL_COMBOS = math.comb(N_MAX, K_PICK)


# ─── helperi (samo oni potrebni za 2a + 2a3d) ────────────────────────
def read_loto_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < K_PICK:
                continue
            try:
                nums = tuple(sorted(int(x) for x in row[:K_PICK]))
            except ValueError:
                continue
            if len(nums) == K_PICK and len(set(nums)) == K_PICK:
                rows.append(nums)
    return rows


def lex_rank_1based(combo, n=N_MAX, k=K_PICK):
    """1-based lex indeks (poklapa se sa rednim brojem u kombinacije_39C7.csv)."""
    combo = tuple(sorted(combo))
    rank0 = 0
    prev = 0
    for i, value in enumerate(combo):
        remaining = k - i - 1
        for candidate in range(prev + 1, value):
            rank0 += math.comb(n - candidate, remaining)
        prev = value
    return rank0 + 1


def sample_entropy(series, m=2, r=None, max_points=1200):
    """Sample entropy; koristi poduzorak ako je niz dug radi brzine."""
    x = np.asarray(series, dtype=float)
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x = x[idx]
    x = (x - x.mean()) / (x.std() + 1e-12)
    if r is None:
        r = 0.2

    def _count(mm):
        templates = np.array([x[i:i + mm] for i in range(len(x) - mm + 1)])
        count = 0
        for i in range(len(templates) - 1):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    b = _count(m)
    a = _count(m + 1)
    if a == 0 or b == 0:
        return float("inf"), a, b, len(x)
    return float(-np.log(a / b)), a, b, len(x)


def permutation_entropy(series, order=3, delay=1):
    """Normalizovana permutation entropy u opsegu 0..1."""
    x = np.asarray(series, dtype=float)
    n_patterns = len(x) - delay * (order - 1)
    if n_patterns <= 0:
        return float("nan"), float("nan"), 0

    counts = {}
    for i in range(n_patterns):
        window = x[i:i + delay * order:delay]
        pattern = tuple(np.argsort(window, kind="mergesort"))
        counts[pattern] = counts.get(pattern, 0) + 1

    probs = np.asarray(list(counts.values()), dtype=float)
    probs = probs / probs.sum()
    pe = float(-np.sum(probs * np.log2(probs)))
    pe_norm = pe / np.log2(math.factorial(order))
    return pe, float(pe_norm), len(counts)


# ─── KORAK 1: f(t) = lex-indeks ──────────────────────────────────────
draws = read_loto_csv(CSV_DRAWS)
N = len(draws)
lex_idx = np.array([lex_rank_1based(c) for c in draws], dtype=np.float64)

print()
print("3_KarlWeierstrass_NEXT3_2a3d — KORAK 1: formiranje krive f(t)")
print(f"  CSV:                  {CSV_DRAWS}")
print(f"  Ucitano izvlacenja:    {N}")
print(f"  C(39,7):              {TOTAL_COMBOS:,}")
print()

with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write("3_KarlWeierstrass_NEXT3_2a3d — Brownovo kretanje + Sample/Permutation entropy (POGODNO)\n")
    f.write("=" * 60 + "\n\n")
    f.write("KORAK 1: Weierstrass-ova funkcija nad svih izvucenih kombinacija\n\n")
    f.write(f"  CSV izvucenih:        {CSV_DRAWS}\n")
    f.write(f"  Ucitano izvlacenja:    {N}\n")
    f.write(f"  C(39,7):              {TOTAL_COMBOS:,}\n")
    f.write("  f(t) = lex-indeks izvucene kombinacije u skupu svih 39C7\n\n")


# ─── KORAK 2a: priprema Brown inkremenata (samo ono sto 2a3d koristi) ─
incr = np.diff(lex_idx)
brown_incr_centered = incr - incr.mean()
brown_path = np.cumsum(brown_incr_centered)


# ─── KORAK 2a3d: Sample / Permutation entropy nad Brown inkrementima ─
T0_2A3D = time.time()

sampen_m = 2
sampen_r = 0.2
sampen_incr, sampen_a, sampen_b, sampen_n = sample_entropy(
    brown_incr_centered, m=sampen_m, r=sampen_r, max_points=1200
)
sampen_path, _, _, _ = sample_entropy(brown_path, m=sampen_m, r=sampen_r, max_points=1200)

pe_orders = [3, 4, 5]
pe_incr_rows = []
pe_path_rows = []
for order in pe_orders:
    pe, pe_norm, patterns = permutation_entropy(brown_incr_centered, order=order, delay=1)
    pe_incr_rows.append((order, pe, pe_norm, patterns, math.factorial(order)))
    pe_p, pe_p_norm, p_patterns = permutation_entropy(brown_path, order=order, delay=1)
    pe_path_rows.append((order, pe_p, pe_p_norm, p_patterns, math.factorial(order)))

rng_2a3d = np.random.default_rng(45)
entropy_shuffle_runs = 100
shuffle_sampen = []
shuffle_pe4 = []
for _ in range(entropy_shuffle_runs):
    shuffled = rng_2a3d.permutation(brown_incr_centered)
    se, _, _, _ = sample_entropy(shuffled, m=sampen_m, r=sampen_r, max_points=1200)
    _, pe4_norm, _ = permutation_entropy(shuffled, order=4, delay=1)
    if np.isfinite(se):
        shuffle_sampen.append(se)
    shuffle_pe4.append(pe4_norm)
shuffle_sampen = np.asarray(shuffle_sampen, dtype=float)
shuffle_pe4 = np.asarray(shuffle_pe4, dtype=float)

shuffle_sampen_mean = float(shuffle_sampen.mean())
shuffle_sampen_std = float(shuffle_sampen.std(ddof=1))
shuffle_sampen_p_low = float(np.mean(shuffle_sampen <= sampen_incr))
shuffle_sampen_z = (sampen_incr - shuffle_sampen_mean) / (shuffle_sampen_std + 1e-12)

pe4_norm = pe_incr_rows[1][2]
shuffle_pe4_mean = float(shuffle_pe4.mean())
shuffle_pe4_std = float(shuffle_pe4.std(ddof=1))
shuffle_pe4_p_low = float(np.mean(shuffle_pe4 <= pe4_norm))
shuffle_pe4_z = (pe4_norm - shuffle_pe4_mean) / (shuffle_pe4_std + 1e-12)

if shuffle_sampen_p_low <= 0.05 or shuffle_pe4_p_low <= 0.05:
    entropy_note = "entropija je niza od shuffled Brown reference (moguca struktura)"
else:
    entropy_note = "entropija je blizu shuffled Brown reference"

print()
print("KORAK 2a3d: Aparat 2a Brownovo kretanje + Test 3d Sample / Permutation entropy")
print(f"  Sample entropy dX: {sampen_incr:.4f}  (m={sampen_m}, r={sampen_r}, n={sampen_n})")
print(f"  Sample entropy Brown-putanja: {sampen_path:.4f}")
print(f"  Permutation entropy dX order=4: {pe4_norm:.4f} normalizovano")
print(f"  shuffled SampEn: mean={shuffle_sampen_mean:.4f} std={shuffle_sampen_std:.4f} "
      f"z={shuffle_sampen_z:.2f} p_low={shuffle_sampen_p_low:.4f}")
print(f"  shuffled PE4: mean={shuffle_pe4_mean:.4f} std={shuffle_pe4_std:.4f} "
      f"z={shuffle_pe4_z:.2f} p_low={shuffle_pe4_p_low:.4f}")
print(f"  ⇒ {entropy_note}")
print()

fig2a3d, ax2a3d = plt.subplots(1, 3, figsize=(16, 5))
fig2a3d.suptitle("KORAK 2a3d: Brownovo kretanje + Sample / Permutation entropy  (POGODNO)",
                 fontsize=13, fontweight="bold")

orders = np.array([row[0] for row in pe_incr_rows], dtype=int)
pe_incr_norms = np.array([row[2] for row in pe_incr_rows], dtype=float)
pe_path_norms = np.array([row[2] for row in pe_path_rows], dtype=float)
ax2a3d[0].plot(orders, pe_incr_norms, "o-", color="darkorange", label="dX")
ax2a3d[0].plot(orders, pe_path_norms, "o-", color="steelblue", label="Brown-putanja")
ax2a3d[0].set_ylim(0, 1.05)
ax2a3d[0].set_title("Normalizovana permutation entropy")
ax2a3d[0].set_xlabel("order")
ax2a3d[0].set_ylabel("PE / max PE")
ax2a3d[0].legend(fontsize=8)
ax2a3d[0].grid(True, alpha=0.25)

ax2a3d[1].hist(shuffle_sampen, bins=20, color="lightgray", edgecolor="white")
ax2a3d[1].axvline(sampen_incr, color="crimson", linewidth=2,
                  label=f"observed={sampen_incr:.3f}")
ax2a3d[1].axvline(shuffle_sampen_mean, color="black", linestyle="--",
                  label=f"shuffle mean={shuffle_sampen_mean:.3f}")
ax2a3d[1].set_title("Shuffled Sample Entropy referenca")
ax2a3d[1].set_xlabel("Sample entropy")
ax2a3d[1].set_ylabel("broj")
ax2a3d[1].legend(fontsize=8)

ax2a3d[2].hist(shuffle_pe4, bins=20, color="lightgray", edgecolor="white")
ax2a3d[2].axvline(pe4_norm, color="crimson", linewidth=2,
                  label=f"observed={pe4_norm:.3f}")
ax2a3d[2].axvline(shuffle_pe4_mean, color="black", linestyle="--",
                  label=f"shuffle mean={shuffle_pe4_mean:.3f}")
ax2a3d[2].set_title("Shuffled PE order=4 referenca")
ax2a3d[2].set_xlabel("normalizovana PE")
ax2a3d[2].set_ylabel("broj")
ax2a3d[2].legend(fontsize=8)

for a in ax2a3d:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig2a3d.tight_layout()
fig2a3d.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
plt.show()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("KORAK 2a3d: Aparat 2a Brownovo kretanje + Test 3d Sample / Permutation entropy\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  PNG:                  {PNG_PATH}\n\n")
    f.write("Sample entropy:\n")
    f.write(f"  m                     = {sampen_m}\n")
    f.write(f"  r                     = {sampen_r}\n")
    f.write(f"  n used                = {sampen_n}\n")
    f.write(f"  SampEn(dX)            = {sampen_incr:.8f}\n")
    f.write(f"  SampEn(Brown-putanja) = {sampen_path:.8f}\n")
    f.write(f"  A count               = {sampen_a}\n")
    f.write(f"  B count               = {sampen_b}\n\n")
    f.write("Permutation entropy:\n")
    f.write(f"  {'order':<8}{'PE bits':>16}{'PE norm':>16}{'patterns':>14}{'max':>10}\n")
    for order, pe, pe_norm, patterns, max_patterns in pe_incr_rows:
        f.write(f"  {order:<8}{pe:>16,.8f}{pe_norm:>16,.8f}{patterns:>14}{max_patterns:>10}\n")
    f.write("\n")
    f.write("Shuffled entropy referenca:\n")
    f.write(f"  runs                  = {entropy_shuffle_runs}\n")
    f.write(f"  SampEn mean           = {shuffle_sampen_mean:.8f}\n")
    f.write(f"  SampEn std            = {shuffle_sampen_std:.8f}\n")
    f.write(f"  SampEn z              = {shuffle_sampen_z:.8f}\n")
    f.write(f"  SampEn p_low          = {shuffle_sampen_p_low:.8f}\n")
    f.write(f"  PE4 mean              = {shuffle_pe4_mean:.8f}\n")
    f.write(f"  PE4 std               = {shuffle_pe4_std:.8f}\n")
    f.write(f"  PE4 z                 = {shuffle_pe4_z:.8f}\n")
    f.write(f"  PE4 p_low             = {shuffle_pe4_p_low:.8f}\n")
    f.write(f"  interpret.            = {entropy_note}\n\n")

    elapsed_2a3d = time.time() - T0_2A3D
    f.write(f"Vreme KORAKA 2a3d: {timedelta(seconds=int(elapsed_2a3d))} ({elapsed_2a3d:.1f} s)\n")
    f.write(f"Ukupno vreme:       {timedelta(seconds=int(time.time()-T0))} ({time.time()-T0:.1f} s)\n")

print(f"PNG saved → {PNG_PATH}")
print(f"TXT saved → {TXT_PATH}")
print(f"Vreme KORAKA 2a3d: {timedelta(seconds=int(time.time()-T0_2A3D))} "
      f"({time.time()-T0_2A3D:.1f} s)")
print(f"Ukupno vreme:      {timedelta(seconds=int(time.time()-T0))} "
      f"({time.time()-T0:.1f} s)")
print()
print("KRAJ 3_KarlWeierstrass_NEXT3_2a3d.")
print()
"""
3_KarlWeierstrass_NEXT3_2a3d — KORAK 1: formiranje krive f(t)
  CSV:                  /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937


KORAK 2a3d: Aparat 2a Brownovo kretanje + Test 3d Sample / Permutation entropy
  Sample entropy dX: 2.2117  (m=2, r=0.2, n=1200)
  Sample entropy Brown-putanja: 2.1418
  Permutation entropy dX order=4: 0.9846 normalizovano
  shuffled SampEn: mean=2.2279 std=0.0317 z=-0.51 p_low=0.3100
  shuffled PE4: mean=0.9993 std=0.0002 z=-65.27 p_low=0.0000
  ⇒ entropija je niza od shuffled Brown reference (moguca struktura)

PNG saved → /3_KarlWeierstrass_NEXT3_2a3d.png
TXT saved → /3_KarlWeierstrass_NEXT3_2a3d.txt
Vreme KORAKA 2a3d: 0:00:30 (30.9 s)
Ukupno vreme:      0:00:30 (30.9 s)

KRAJ 3_KarlWeierstrass_NEXT3_2a3d.
"""



###############   PREDIKCIJA 3  ###############################

"""
NEXT3 (2a3d, PE) — sledeći ordinal pattern najverovatniji za m=4 → smer + magnituda inkrementa.
"""


def lex_unrank_1based(rank, n=N_MAX, k=K_PICK):
    """Vracanje 1-based lex indeksa u Loto 7/39 kombinaciju."""
    rank0 = int(rank) - 1
    combo = []
    prev = 0
    for i in range(k):
        remaining = k - i - 1
        for candidate in range(prev + 1, n + 1):
            count = math.comb(n - candidate, remaining)
            if rank0 >= count:
                rank0 -= count
            else:
                combo.append(candidate)
                prev = candidate
                break
    return tuple(combo)


def ordinal_pattern(values):
    """Ordinalni pattern bez amplitude: redosled elemenata po velicini."""
    return tuple(np.argsort(np.asarray(values, dtype=float), kind="mergesort"))


T0_PRED3 = time.time()

# PE signal je najjaci za order=4: koristimo poslednja 3 dX kao kontekst,
# a istorijske cetvrte vrednosti iz istog ordinalnog konteksta kao prognozu.
pe_pred_order = 4
context_len = pe_pred_order - 1
x = np.asarray(brown_incr_centered, dtype=float)
current_context = ordinal_pattern(x[-context_len:])

matched_next_centered = []
matched_patterns = {}
for i in range(0, len(x) - pe_pred_order + 1):
    context = ordinal_pattern(x[i:i + context_len])
    if context != current_context:
        continue
    next_centered = float(x[i + context_len])
    matched_next_centered.append(next_centered)
    full_pattern = ordinal_pattern(x[i:i + pe_pred_order])
    matched_patterns[full_pattern] = matched_patterns.get(full_pattern, 0) + 1

if matched_next_centered:
    matched_next_centered = np.asarray(matched_next_centered, dtype=float)
    pred3_note = "koristi se istorija istog ordinalnog konteksta"
else:
    matched_next_centered = x
    pred3_note = "nema poklapanja ordinalnog konteksta; koristi se globalna distribucija"

last_lex = float(lex_idx[-1])
last_incr = float(incr[-1])
mean_incr = float(incr.mean())
pred_centered_incr = float(np.mean(matched_next_centered))
pred_incr = mean_incr + pred_centered_incr
pred_lex_float = last_lex + pred_incr
pred_lex = int(np.clip(round(pred_lex_float), 1, TOTAL_COMBOS))
pred_combo = lex_unrank_1based(pred_lex)

top_patterns = sorted(matched_patterns.items(), key=lambda item: item[1], reverse=True)[:5]
quantile_grid = [0.10, 0.25, 0.50, 0.75, 0.90]
candidate_rows = []
seen_lex = set()
for q in quantile_grid:
    cand_centered = float(np.quantile(matched_next_centered, q))
    cand_incr = mean_incr + cand_centered
    cand_lex = int(np.clip(round(last_lex + cand_incr), 1, TOTAL_COMBOS))
    if cand_lex in seen_lex:
        continue
    seen_lex.add(cand_lex)
    candidate_rows.append((q, cand_incr, cand_lex, lex_unrank_1based(cand_lex)))

print()
print("PREDIKCIJA 3 — NEXT3 / 2a3d / PE / ordinalni kontekst")
print(f"  PE order               = {pe_pred_order}")
print(f"  trenutni kontekst      = {current_context}")
print(f"  istorijskih prelaza    = {len(matched_next_centered)}")
print(f"  zadnji lex             = {int(last_lex):,}")
print(f"  zadnji inkrement       = {last_incr:,.2f}")
print(f"  pred. inkrement        = {pred_incr:,.2f}")
print(f"  pred. lex              = {pred_lex:,}")
print(f"  pred. kombinacija      = {pred_combo}")
print(f"  napomena               = {pred3_note}")
print("  najcesci order=4 patterni iz tog konteksta:")
for pattern, count in top_patterns:
    print(f"    pattern={pattern}  count={count}")
print("  kvantil kandidati:")
for q, cand_incr, cand_lex, combo in candidate_rows:
    print(f"    q={q:>4.2f}  dX={cand_incr:>14,.2f}  lex={cand_lex:>10,}  combo={combo}")
print()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("PREDIKCIJA 3: NEXT3 / 2a3d / PE / ordinalni kontekst\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model:\n")
    f.write("  PE order=4 je nizi od shuffled reference.\n")
    f.write("  Poslednja 3 centrirana dX cine ordinalni kontekst.\n")
    f.write("  Predikcija koristi istorijske sledece dX vrednosti iz istog konteksta.\n\n")
    f.write("Parametri:\n")
    f.write(f"  PE order               = {pe_pred_order}\n")
    f.write(f"  trenutni kontekst      = {current_context}\n")
    f.write(f"  broj prelaza           = {len(matched_next_centered)}\n")
    f.write(f"  PE4 norm               = {pe4_norm:.8f}\n")
    f.write(f"  PE4 shuffled mean      = {shuffle_pe4_mean:.8f}\n")
    f.write(f"  PE4 p_low              = {shuffle_pe4_p_low:.8f}\n")
    f.write(f"  mean(dX)               = {mean_incr:,.8f}\n")
    f.write(f"  zadnji lex             = {int(last_lex):,}\n")
    f.write(f"  zadnji inkrement       = {last_incr:,.8f}\n")
    f.write(f"  pred. centrirani dX    = {pred_centered_incr:,.8f}\n")
    f.write(f"  pred. inkrement        = {pred_incr:,.8f}\n")
    f.write(f"  napomena               = {pred3_note}\n\n")
    f.write("Glavna prognoza:\n")
    f.write(f"  pred. lex float        = {pred_lex_float:,.8f}\n")
    f.write(f"  pred. lex              = {pred_lex:,}\n")
    f.write(f"  pred. kombinacija      = {pred_combo}\n\n")
    f.write("Najcesci order=4 patterni iz istog konteksta:\n")
    for pattern, count in top_patterns:
        f.write(f"  pattern={pattern}  count={count}\n")
    f.write("\n")
    f.write("Kvantil kandidati iz ordinalnog konteksta:\n")
    f.write(f"  {'q':>8}{'dX':>18}{'lex':>14}  kombinacija\n")
    for q, cand_incr, cand_lex, combo in candidate_rows:
        f.write(f"  {q:>8.2f}{cand_incr:>18,.8f}{cand_lex:>14,}  {combo}\n")
    f.write("\n")
    elapsed_pred3 = time.time() - T0_PRED3
    f.write(f"Vreme PREDIKCIJE 3: {timedelta(seconds=int(elapsed_pred3))} ({elapsed_pred3:.1f} s)\n")

print(f"TXT updated → {TXT_PATH}")
print(f"Vreme PREDIKCIJE 3: {timedelta(seconds=int(time.time()-T0_PRED3))} "
      f"({time.time()-T0_PRED3:.1f} s)")
print()

"""
iz entropy bloka, pa dodajem predikciju preko ordinalnog obrasca.

poslednja 3 centrirana inkrementa se tretiraju kao ordinalni kontekst, iz istorije se uzimaju sledeći inkrementi za isti kontekst, pa se lex vraća u kombinacije.

koristi PE order=4
poslednja 3 centrirana inkrementa uzima kao ordinalni kontekst
iz istorije uzima sledeće inkremente za isti ordinalni kontekst
računa glavnu prognozu i kvantil-kandidate
vraća svaki lex u Loto 7/39 kombinaciju
upisuje sve u 3_KarlWeierstrass_NEXT3_2a3d.txt
"""



"""
3_KarlWeierstrass_NEXT3_2a3d — KORAK 1: formiranje krive f(t)
  CSV:                  /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937


KORAK 2a3d: Aparat 2a Brownovo kretanje + Test 3d Sample / Permutation entropy
  Sample entropy dX: 2.2117  (m=2, r=0.2, n=1200)
  Sample entropy Brown-putanja: 2.1418
  Permutation entropy dX order=4: 0.9846 normalizovano
  shuffled SampEn: mean=2.2279 std=0.0317 z=-0.51 p_low=0.3100
  shuffled PE4: mean=0.9993 std=0.0002 z=-65.27 p_low=0.0000
  ⇒ entropija je niza od shuffled Brown reference (moguca struktura)

PNG saved → /3_KarlWeierstrass_NEXT3_2a3d.png
TXT saved → /3_KarlWeierstrass_NEXT3_2a3d.txt
Vreme KORAKA 2a3d: 0:00:10 (10.5 s)
Ukupno vreme:      0:00:10 (10.5 s)

KRAJ 3_KarlWeierstrass_NEXT3_2a3d.


PREDIKCIJA 3 — NEXT3 / 2a3d / PE / ordinalni kontekst
  PE order               = 4
  trenutni kontekst      = (2, 1, 0)
  istorijskih prelaza    = 654
  zadnji lex             = 513,114
  zadnji inkrement       = -2,143,496.00
  pred. inkrement        = 2,260,524.41
  pred. lex              = 2,773,638
  pred. kombinacija      = (2, x, 4, y, 11, z, 25)
  napomena               = koristi se istorija istog ordinalnog konteksta
  najcesci order=4 patterni iz tog konteksta:
    pattern=(2, 1, 0, 3)  count=218
    pattern=(2, 1, 3, 0)  count=194
    pattern=(2, 3, 1, 0)  count=121
    pattern=(3, 2, 1, 0)  count=121
  kvantil kandidati:
    q=0.10  dX= -5,779,679.00  lex=         1  combo=(1, 2, 3, 4, 5, 6, 7)
    q=0.50  dX=  2,321,449.50  lex= 2,834,564  combo=(2, x, 5, y, 20, z, 26)
    q=0.75  dX=  6,963,264.00  lex= 7,476,378  combo=(4, x, 13, y, 22, z, 34)
    q=0.90  dX= 10,186,969.40  lex=10,700,083  combo=(6, x, 17, y, 21, z, 34)

TXT updated → /3_KarlWeierstrass_NEXT3_2a3d.txt
Vreme PREDIKCIJE 3: 0:00:00 (0.0 s)
"""
