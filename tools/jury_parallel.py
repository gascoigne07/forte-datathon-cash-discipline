"""Приёмка как у жюри, но ПАРАЛЛЕЛЬНО: N видео отправляются одновременно, лимит
5 минут считается от отправки каждого. Плюс границы входа: 401 / 400 / 413 / 404.

tools/jury_check.sh шлёт ролики по очереди и ждёт каждый — так нагрузку очереди
не увидеть (на шаге 9 полнота падала только при параллельной отправке).

    .venv/bin/python tools/jury_parallel.py <base_url> <api_key> <видео> [<видео> ...]
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

LIMIT = 300
HERE = Path(__file__).resolve().parent


def _multipart(path: Path, data: bytes | None = None):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
    body += (data if data is not None else path.read_bytes()) + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def request(url, key=None, body=None, ctype=None, method="GET"):
    req = urllib.request.Request(url, data=body, method=method)
    if key is not None:
        req.add_header("X-API-Key", key)
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def run_one(base, key, video: Path, out: dict):
    t0 = time.time()
    body, ctype = _multipart(video)
    code, resp = request(f"{base}/api/v1/analyze", key, body, ctype, "POST")
    if code != 200:
        out[video.name] = {"error": f"analyze -> {code} {resp[:200]!r}"}
        return
    jid = json.loads(resp)["job_id"]
    while True:
        time.sleep(2)
        code, resp = request(f"{base}/api/v1/jobs/{jid}", key)
        d = json.loads(resp)
        el = time.time() - t0
        if d.get("status") in ("done", "failed") or el > LIMIT + 30:
            break
    out[video.name] = {"elapsed": round(el, 1), "body": d}


def boundaries(base, key, sample: Path) -> list[tuple[str, int, int]]:
    res = []
    body, ctype = _multipart(sample, data=b"\0" * 1024)
    res.append(("без ключа", 401, request(f"{base}/api/v1/analyze", None, body, ctype, "POST")[0]))
    res.append(("неверный ключ", 401, request(f"{base}/api/v1/analyze", "wrong", body, ctype, "POST")[0]))
    body, ctype = _multipart(Path("notes.txt"), data=b"hello")
    res.append((".txt вместо видео", 400, request(f"{base}/api/v1/analyze", key, body, ctype, "POST")[0]))
    big = int(os.environ.get("MAX_UPLOAD_MB", 200)) * 1024 * 1024 + 1024 * 1024
    body, ctype = _multipart(Path("big.mov"), data=b"\0" * big)
    res.append(("файл больше лимита", 413, request(f"{base}/api/v1/analyze", key, body, ctype, "POST")[0]))
    res.append(("неизвестный job_id", 404, request(f"{base}/api/v1/jobs/nonexistent00", key)[0]))
    return res


def main() -> int:
    base, key, videos = sys.argv[1].rstrip("/"), sys.argv[2], [Path(v) for v in sys.argv[3:]]
    fail = 0
    print("== границы входа ==")
    for label, want, got in boundaries(base, key, videos[0]):
        ok = want == got
        fail |= not ok
        print(f"  {label:22s} ожидалось {want}, получено {got}  {'OK' if ok else 'ОШИБКА'}")

    print(f"== {len(videos)} видео одновременно ==")
    out: dict = {}
    th = [threading.Thread(target=run_one, args=(base, key, v, out)) for v in videos]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for v in videos:
        r = out.get(v.name, {"error": "нет результата"})
        print(f"-- {v.name}")
        if "error" in r:
            print(f"  ОШИБКА: {r['error']}")
            fail = 1
            continue
        with tempfile.TemporaryFile("w+") as fh:
            json.dump(r["body"], fh, ensure_ascii=False)
            fh.seek(0)
            p = subprocess.run([sys.executable, str(HERE / "validate_response.py"),
                                str(int(r["elapsed"]))], stdin=fh, capture_output=True, text=True)
        print(p.stdout.rstrip())
        fail |= p.returncode != 0
        if r["elapsed"] > LIMIT:
            print(f"  ОШИБКА: {r['elapsed']} с от отправки — больше {LIMIT} с")
            fail = 1
    print("\nИТОГ:", "все проверки пройдены" if not fail else "есть нарушения, см. выше")
    return fail


if __name__ == "__main__":
    sys.exit(main())
