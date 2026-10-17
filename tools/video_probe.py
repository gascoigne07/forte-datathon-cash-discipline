"""Video-режим VLM: весь ролик за один инференс вместо кадра за кадром.

mlx-vlm умеет generate(..., video=<path>, fps=...) — модель получает всю
последовательность кадров и может рассуждать о времени. Это принципиально
другой режим: один дорогой prefill вместо N отдельных запросов.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import vlm  # noqa: E402
from evaluate import LABELS  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"

PROMPT = """Это запись рабочего места кассира банка (вид сверху, одна камера, звука нет).
На столе: ноутбук, стационарный телефон, счётно-денежная машина. Рядом серый металлический шкаф.
Через стеклянную стену видно соседнее помещение — происходящее ЗА СТЕКЛОМ не описывай.

Проверяющего интересуют только эти наблюдаемые признаки:
1) банкноты в руках или на столешнице;
2) пересчёт денег руками, когда счётно-денежная машина не используется;
3) деньги или ценности остались на столе, а работник отошёл;
4) в комнату вошёл посторонний, пока на столе наличные;
5) на столе стакан, кружка, банка, бутылка или еда;
6) на рабочем месте личная сумка, рюкзак или пакет;
7) инкассаторская сумка вне сейфа.

Опиши по порядку, что происходит в ролике, указывая время в секундах.
Затем ответь строго в таком виде:
ПРИЗНАК: <номер из списка или «нет»>
СЕКУНДА: <число или «нет»>
ЧТО_ВИДНО: <одно предложение>"""


def main():
    names = sys.argv[1:] or ["X-1", "Х-3"]
    fps = 1.0
    t0 = time.time(); vlm.warmup()
    print(f"модель загружена за {time.time()-t0:.0f}с, fps={fps}\n")
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, config = vlm._load()

    for name in names:
        p = f"{VID}/{name}.mov"
        if not Path(p).exists():
            print(f"{name}: нет файла"); continue
        fmt = apply_chat_template(processor, config, PROMPT, num_images=0, video=True)
        t = time.time()
        try:
            res = generate(model, processor, fmt, video=[p], fps=fps,
                           max_tokens=320, temperature=0.0, verbose=False)
        except Exception as exc:                            # noqa: BLE001
            print(f"{name}: ОШИБКА {type(exc).__name__}: {exc}"); continue
        txt = getattr(res, "text", str(res))
        gold = LABELS.get(name)
        print(f"=== {name}  разметка={gold}  [{time.time()-t:.0f}с]")
        print(txt.strip()[:900])
        print()


if __name__ == "__main__":
    main()
