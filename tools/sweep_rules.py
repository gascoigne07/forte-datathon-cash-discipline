"""Перебор вариантов правил с замером на обучающей выборке.

Правила здесь воспроизводятся параметрически, чтобы перебирать варианты без
правки service/app/rules.py. Выбранный вариант потом переносится в сервис и
обязательно переизмеряется через tools/eval_caps.py.

Критерий выбора задан ТЗ (п.16 FAQ): максимум полноты при ложных <= 2 из 19.
"""
import glob, json, sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import caption  # noqa: E402
from evaluate import LABELS  # noqa: E402

DUPLICATES = {"Х-47", "Х-48"}


@dataclass
class Cfg:
    track_area: int = 60          # порог площади трека для unattended_desk
    left_mode: str = "off"        # off | track | paper — предусловие left_workplace
    left_support: int = 3
    outsider: bool = False        # people>=3 + (трек или бумага)
    outsider_support: int = 2
    outsider_min: int = 3         # порог числа людей
    outsider_need: str = "any"    # any | track — чем подтверждать предмет


def load_series(name, cfg: Cfg):
    cp = Path(f"data/captions/{name}.caps.json")
    if not cp.exists():
        return None
    caps = json.loads(cp.read_text())["captions"]
    fp = Path(f"data/feats/{name}.feats.json")
    feats = json.loads(fp.read_text()) if fp.exists() else {"rows": [], "tracks": []}
    rows = feats["rows"]; byt = {r["t"]: r for r in rows}
    tracks = [t for t in feats.get("tracks", []) if t["area"] >= cfg.track_area]
    out = []
    for c in caps:
        f = caption.facts_from_caption(c["caption"])
        row = byt.get(c["t"]) or (min(rows, key=lambda r: abs(r["t"] - c["t"])) if rows else None)
        f["people_in_cabin"] = row["ppl"] if row else 0
        f["teller_present"] = bool(row and row["ppl"] > 0 and row.get("teller_dist") is not None)
        f["track_on_desk"] = any(t["t0"] - 3 <= c["t"] <= t["t1"] + 3 for t in tracks)
        out.append((c["t"], f))
    return out


def _runs(series, pred):
    out, start = [], None
    for i, (_, f) in enumerate(series):
        if pred(f):
            start = i if start is None else start
        elif start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(series) - 1))
    return out


def events(series, cfg: Cfg):
    """Возвращает [(ключ, таймкод)] — достаточно для метрик вердикта и Δt."""
    ev = []
    f0 = lambda k: any(f.get(k) for _, f in series)      # noqa: E731

    for lo, hi in _runs(series, lambda f: f.get("drink_or_food")):
        if hi - lo + 1 >= 2:
            ev.append(("food_drink", series[lo if lo > 0 else (lo + hi) // 2][0]))

    for lo, hi in _runs(series, lambda f: f.get("personal_bag")
                        and f.get("bag_ctx") == "on_desk"):
        if hi - lo + 1 >= 2:
            ev.append(("personal_bag", series[lo if lo > 0 else (lo + hi) // 2][0]))

    for lo, hi in _runs(series, lambda f: f.get("sealed_bag")
                        and f.get("sealed_ctx") == "on_desk"):
        if hi - lo + 1 >= 2:
            ev.append(("sealed_bag", series[lo if lo > 0 else (lo + hi) // 2][0]))

    for lo, hi in _runs(series, lambda f: f.get("cash_seen")
                        and not f.get("teller_present", True)):
        if hi - lo + 1 >= 2:
            ev.append(("cash_unattended", series[lo][0]))

    # 5. Работник ушёл. Предусловие «в ролике были ценности» параметризовано:
    #    исходное f0("cash_seen") никогда не истинно, потому что денежный канал
    #    не работает — правило было мёртвым.
    pre = {"off": f0("cash_seen") or f0("sealed_bag"),
           "track": f0("track_on_desk"),
           "paper": f0("paper_seen")}[cfg.left_mode if cfg.left_mode != "off" else "off"]
    if pre:
        for lo, hi in _runs(series, lambda f: not f.get("teller_present", True)):
            if hi - lo + 1 >= cfg.left_support:
                t = series[lo][0]
                if not any(k == "cash_unattended" for k, _ in ev):
                    ev.append(("left_workplace", t))

    # 6. Рабочее место без работника, на столешнице предметы.
    rr = _runs(series, lambda f: f.get("no_person_in_caption")
               and (f.get("paper_seen") or f.get("personal_bag")
                    or f.get("sealed_bag") or f.get("track_on_desk")))
    if rr:
        lo, hi = max(rr, key=lambda r: r[1] - r[0])
        t = series[(lo + hi) // 2][0]
        if not any(abs(t - t0) < 3 for _, t0 in ev):
            ev.append(("unattended_desk", t))

    # 7. Посторонние: трое и более одновременно с предметом на столешнице.
    if cfg.outsider:
        need = ((lambda f: f.get("track_on_desk")) if cfg.outsider_need == "track"
                else (lambda f: f.get("track_on_desk") or f.get("paper_seen")))
        for lo, hi in _runs(series, lambda f: f.get("people_in_cabin", 0) >= cfg.outsider_min
                            and need(f)):
            if hi - lo + 1 >= cfg.outsider_support:
                ev.append(("outsider_with_cash", series[lo][0]))
    return ev


def measure(cfg: Cfg):
    tp = fp = 0; errs = []; fps = []; hits = []
    for p in sorted(glob.glob("data/captions/*.caps.json")):
        name = Path(p).stem.replace(".caps", "")
        if name not in LABELS or name in DUPLICATES:
            continue
        s = load_series(name, cfg)
        if not s:
            continue
        ev = events(s, cfg)
        gold = LABELS[name]
        if ev and gold:
            tp += 1; hits.append(name)
            errs.append(min(abs(t - g) for _, t in ev for g in gold))
        elif ev:
            fp += 1; fps.append(name)
    errs.sort()
    med = errs[len(errs) // 2] if errs else -1
    le3 = 100 * sum(e <= 3 for e in errs) / len(errs) if errs else 0
    acc = 100 * (tp + 19 - fp) / 40
    return tp, fp, acc, med, le3, fps, hits


VARIANTS = [
    ("базовая линия", Cfg()),
    ("посторонние 3+, предмет, 3 кадра", Cfg(outsider=True, outsider_support=3)),
    ("посторонние 3+, предмет, 4 кадра", Cfg(outsider=True, outsider_support=4)),
    ("посторонние 3+, ТРЕК, 2 кадра", Cfg(outsider=True, outsider_need="track")),
    ("посторонние 3+, ТРЕК, 3 кадра", Cfg(outsider=True, outsider_need="track", outsider_support=3)),
    ("посторонние 4+, предмет, 2 кадра", Cfg(outsider=True, outsider_min=4)),
    ("посторонние 4+, ТРЕК, 2 кадра", Cfg(outsider=True, outsider_min=4, outsider_need="track")),
    ("трек >=45", Cfg(track_area=45)),
    ("left: трек в ролике, 3 кадра", Cfg(left_mode="track")),
    ("left: трек в ролике, 2 кадра", Cfg(left_mode="track", left_support=2)),
    ("left: бумага в ролике, 2 кадра", Cfg(left_mode="paper", left_support=2)),
    ("посторонние: 3+ и предмет", Cfg(outsider=True)),
    ("трек >=45 + left трек 2", Cfg(track_area=45, left_mode="track", left_support=2)),
    ("трек >=45 + посторонние", Cfg(track_area=45, outsider=True)),
    ("left трек 2 + посторонние", Cfg(left_mode="track", left_support=2, outsider=True)),
    ("всё вместе", Cfg(track_area=45, left_mode="track", left_support=2, outsider=True)),
]

if __name__ == "__main__":
    print(f"{'вариант':34s} {'найд':>5s} {'ложн':>5s} {'вердикт':>8s} {'Δt мед':>7s} {'≤3с':>5s}  ложные")
    for label, cfg in VARIANTS:
        tp, fp, acc, med, le3, fps, hits = measure(cfg)
        flag = "" if fp <= 2 else "  ← выше лимита ложных"
        print(f"{label:34s} {tp:4d}  {fp:4d}  {acc:7.0f}% {med:7.1f} {le3:4.0f}%  "
              f"{','.join(fps)}{flag}")
