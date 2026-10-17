"""Прогон роликов через ЖИВОЙ сервис с записью результата в формате run_pipeline.py.

run_pipeline.py вызывает pipeline.analyze в своём процессе — на GPU-сервере это
вторая копия обеих моделей описания (~19 ГБ видеопамяти) рядом с боевым сервисом.
Здесь ролики идут через HTTP API боевого сервиса: замеряется ровно то, что получит
жюри, а результаты ложатся в data/runs/<имя>.run.json и читаются metrics_summary.py.

    .venv/bin/python tools/run_service.py <base_url> <api_key> <видео> [...] [--par 2]
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jury_parallel import _multipart, request  # noqa: E402


def run(base, key, v: Path, out_dir: Path):
    dst = out_dir / f"{v.stem}.run.json"
    if dst.exists():
        return
    t0 = time.time()
    body, ctype = _multipart(v)
    code, resp = request(f"{base}/api/v1/analyze", key, body, ctype, "POST")
    if code != 200:
        print(f"{v.stem}: analyze -> {code}", file=sys.stderr)
        return
    jid = json.loads(resp)["job_id"]
    while True:
        time.sleep(2)
        d = json.loads(request(f"{base}/api/v1/jobs/{jid}", key)[1])
        if d.get("status") in ("done", "failed"):
            break
    if d.get("status") != "done":
        print(f"{v.stem}: {d.get('status')} {d.get('error')}", file=sys.stderr)
        return
    d["violations"] = [{k: x for k, x in v0.items() if k != "evidence_frames"}
                       | {"evidence_count": len(v0.get("evidence_frames") or [])}
                       for v0 in d.get("violations") or []]
    d["elapsed_from_submit"] = round(time.time() - t0, 1)
    dst.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    ev = ", ".join(f"{x['type'][:28]}@{x['timecode_sec']:.1f}" for x in d["violations"])
    print(f"{v.stem:28s} {d['verdict']:13s} {d['elapsed_from_submit']:5.0f}с "
          f"кадров={d.get('frames_analyzed')} {ev}", file=sys.stderr, flush=True)


def main():
    args = sys.argv[1:]
    par = 2
    if "--par" in args:
        i = args.index("--par"); par = int(args[i + 1]); del args[i:i + 2]
    base, key, videos = args[0].rstrip("/"), args[1], [Path(v) for v in args[2:]]
    out_dir = Path("data/runs"); out_dir.mkdir(parents=True, exist_ok=True)
    queue = list(videos)
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                v = queue.pop(0)
            run(base, key, v, out_dir)

    th = [threading.Thread(target=worker) for _ in range(par)]
    for t in th:
        t.start()
    for t in th:
        t.join()


if __name__ == "__main__":
    main()
