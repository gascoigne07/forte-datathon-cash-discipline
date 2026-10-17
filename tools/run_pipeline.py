"""Прогон боевого пайплайна по списку видео (тот же код, что в сервисе).

Нужен, чтобы сверить быструю оценку по кэшу подписей с тем, что реально отдаёт
сервис: в бою ключевые кадры выбираются адаптивно, а не равномерной сеткой.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import pipeline  # noqa: E402
from evaluate import LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="data/runs")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    for v in a.videos:
        name = Path(v).stem
        dst = Path(a.out) / f"{name}.run.json"
        if dst.exists():
            continue
        t0 = time.time()
        try:
            res = pipeline.analyze(v)
        except Exception as exc:                        # noqa: BLE001
            print(f"{name}: ОШИБКА {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        # кадры-доказательства в кэш не пишем, они тяжёлые
        slim = dict(res)
        slim["violations"] = [{k: val for k, val in v0.items() if k != "evidence_frames"}
                              | {"evidence_count": len(v0["evidence_frames"])}
                              for v0 in res["violations"]]
        dst.write_text(json.dumps(slim, ensure_ascii=False, indent=1))

        base = name.split("__")[0]
        gold = LABELS.get(base)
        gold_s = f"разметка={gold}" if gold is not None else ""
        ev = ", ".join(f"{x['type'][:28]}@{x['timecode_sec']:.0f}" for x in res["violations"])
        print(f"{name:26s} {res['verdict']:13s} {time.time()-t0:5.0f}с "
              f"кадров={res['frames_analyzed']:3d} {gold_s:16s} {ev}", file=sys.stderr)


if __name__ == "__main__":
    main()
