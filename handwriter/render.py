"""Сборка конспекта: текст -> рукописные символы на фото чистого листа."""
import json
import math
import os
import random
import re

import cv2
import numpy as np
from PIL import Image

from .extract import INSET, SINGLE_STROKE, clean_alpha, imread_any, imwrite_any, trim
from .paper import auto_orient, detect_lines, plan_rows, preview_rows, row_offset

# Буквы без выносных элементов — по ним меряем высоту строчных (x-height) почерка.
XH_CHARS = set("аеиоснмпшacemnorsuvwxz")

# Если символа нет в шрифте — чем его заменить (похожие буквы кириллицы/латиницы и т.п.).
FALLBACK = {
    "ё": "е", "Ё": "Е", "й": "и", "Й": "И", "—": "-", "–": "-", "‑": "-", "−": "-",
    "“": "\"", "”": "\"", "„": "\"", "«": "\"", "»": "\"", "\"": "«", "’": "'", "‘": "'",
    "ʼ": "'", "`": "'", "×": "x", "·": ".", "№": "N",
    "і": "i", "І": "I", "ґ": "г", "Ґ": "Г", "ў": "у", "Ў": "У",
}
_LOOK = "аaеeоoрpсcуyхxАAВBЕEКKМMНHОOРPСCТTХX0О3З"
for _i in range(0, len(_LOOK), 2):
    FALLBACK.setdefault(_LOOK[_i], _LOOK[_i + 1])
    FALLBACK.setdefault(_LOOK[_i + 1], _LOOK[_i])

# Сетка высот (в высотах строчных относительно базовой линии) для расстановки букв по контуру.
YGRID = np.arange(-2.2, 1.6, 0.05)
KERN_GAP = 0.16   # минимальный зазор между чернилами соседних букв, в высотах строчных
SQUASH_MIN = 0.55  # насколько можно сжать хвост/верх буквы по вертикали, прежде чем уменьшать её целиком

# Буквы, которые можно получить из других: є = отражённая э, ї = і с двумя точками.
DERIVED = {"є": ("э", "mirror"), "Є": ("Э", "mirror"), "ї": ("і", "two_dots"), "Ї": ("І", "two_dots")}


def _mirror(g):
    return dict(g, img=g["img"][:, ::-1].copy(), prof=None)


def _two_dots(g):
    a = g["img"]
    n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.3).astype(np.uint8), connectivity=8)
    if n < 3:
        return g
    body = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    bt = st[body, cv2.CC_STAT_TOP]
    dots = [i for i in range(1, n) if i != body
            and st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] <= bt + 3
            and st[i, cv2.CC_STAT_AREA] < 0.4 * st[body, cv2.CC_STAT_AREA]]
    if not dots:
        return g
    d = dots[0]
    dm = (cv2.dilate((lab == d).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    dot = a * dm
    s = int(st[d, cv2.CC_STAT_WIDTH] * 0.9) + 4
    out = np.zeros((a.shape[0], a.shape[1] + 2 * s), np.float32)
    out[:, s:s + a.shape[1]] = a * ~dm
    out[:, 0:a.shape[1]] = np.maximum(out[:, 0:a.shape[1]], dot)
    out[:, 2 * s:] = np.maximum(out[:, 2 * s:], dot)
    t = trim(out, pad=3)
    if t is None:
        return g
    img, y0, _ = t
    return dict(g, img=img, w=img.shape[1], h=img.shape[0], top=g["top"] + y0, prof=None)


class HandFont:
    def __init__(self, font_dir):
        with open(os.path.join(font_dir, "font.json"), encoding="utf-8") as f:
            meta = json.load(f)
        geo = self._geometry(font_dir, meta)
        self.by_char = {}
        for g in meta["glyphs"]:
            path = os.path.join(font_dir, "glyphs", g["file"])
            if not os.path.exists(path):      # букву удалили вручную — пропускаем
                continue
            a = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            g = dict(g, img=a)
            if g.get("clean", 0) < 2 and geo:
                g = self._repair(g, *geo)      # шрифт, собранный старой версией
                if g is None:
                    continue
            self.by_char.setdefault(g["char"], []).append(g)
        if not self.by_char:
            raise ValueError("В шрифте нет ни одного символа — сначала загрузите фото шаблона.")
        hs = [g["h"] for c, gl in self.by_char.items() if c in XH_CHARS for g in gl]
        self.xheight = float(np.median(hs)) if hs else meta.get("box_h", 250) * 0.25
        self.ink_rgb = meta.get("ink_rgb", [30, 40, 120])
        self.missing, self.cache = set(), {}
        self._fix_punct()

    def _fix_punct(self):
        """В пустой рамке запятую и точку часто пишут крупно — в тексте они тогда
        выглядят как скобка или «о». Приводим к обычному размеру относительно строчных."""
        xh = self.xheight
        for ch, max_h, new_h, top in ((",", 0.8, 0.6, -0.2), (".", 0.4, 0.28, None)):
            for g in self.by_char.get(ch, []):
                if g["h"] <= max_h * xh:
                    continue
                f = new_h * xh / g["h"]
                img = cv2.resize(g["img"], (max(1, round(g["w"] * f)), max(1, round(g["h"] * f))),
                                 interpolation=cv2.INTER_AREA)
                centre = g["top"] + g["h"] / 2
                g.update(img=img, w=img.shape[1], h=img.shape[0],
                         top=int(round(top * xh if top is not None else centre - img.shape[0] / 2)))
        # строчные без хвостов, вписанные в шаблон выше или ниже пунктира, ставим на линию
        for ch in XH_CHARS:
            for g in self.by_char.get(ch, []):
                bottom = g["top"] + g["h"]
                if bottom < -0.1 * xh or bottom > 0.2 * xh:
                    g["top"] = int(round(0.03 * xh - g["h"]))

    def profile(self, g):
        """Левый и правый край чернил символа по высоте (сетка YGRID, единицы шрифта).
        По ним буквы ставятся рядом «по контуру», а не по прямоугольникам: у наклонного
        почерка прямоугольники дают слишком большие дыры между буквами."""
        p = g.get("prof")
        if p is not None:
            return p
        a = g["img"] > 0.35
        h, w = a.shape
        any_ = a.any(axis=1)
        left = np.where(any_, a.argmax(axis=1), w).astype(float)
        right = np.where(any_, w - 1 - a[:, ::-1].argmax(axis=1), -1).astype(float)
        d = max(1, int(0.12 * self.xheight))
        L, R = np.full(len(YGRID), np.nan), np.full(len(YGRID), np.nan)
        for i, yg in enumerate(YGRID):
            r = int(round(yg * self.xheight - g["top"]))
            lo, hi = max(0, r - d), min(h, r + d + 1)
            if lo < hi and any_[lo:hi].any():
                L[i], R[i] = left[lo:hi].min(), right[lo:hi].max()
        g["prof"] = (L, R)
        return g["prof"]

    @staticmethod
    def _geometry(font_dir, meta):
        if "crop_w" in meta:
            return meta["crop_w"], meta["crop_h"], meta["crop_top"], meta["crop_bottom"]
        try:
            with open(os.path.join(font_dir, "layout.json"), encoding="utf-8") as f:
                cell = json.load(f)["pages"][0]["cells"][0]
        except (OSError, KeyError, IndexError):
            return None
        _, by, bw, bh = cell["box"]
        off = cell["baseline"] - by
        return bw - 2 * INSET, bh - 2 * INSET, INSET - off, bh - INSET - off

    @staticmethod
    def _repair(g, crop_w, crop_h, crop_top, crop_bottom):
        """Убирает из старых букв куски рамки и пятна, заново обрезает по чернилам."""
        a = g["img"]
        h, w = a.shape
        edges = {"top": 0 if g["top"] <= crop_top + 1 else None,
                 "bottom": h if g["top"] + h >= crop_bottom - 1 else None, "left": 0, "right": w}
        c = clean_alpha(a, edges, crop_w, crop_h, v_frac=0.3 if w >= crop_w - 2 else 0.75,
                        single=g["char"] in SINGLE_STROKE)
        t = trim(c) if c is not None else None
        if t is None or (t[0] > 0.5).sum() < 40:
            return None
        img, y0, _ = t
        return dict(g, img=img, w=img.shape[1], h=img.shape[0], top=g["top"] + y0)

    def variants(self, ch):
        if ch in self.cache:
            return self.cache[ch]
        res = self._lookup(ch, set())
        if res is None:
            self.missing.add(ch)
        self.cache[ch] = res
        return res

    def _lookup(self, ch, seen):
        queue = [ch]
        while queue:
            c = queue.pop(0)
            if c in seen:
                continue
            seen.add(c)
            if c in self.by_char:
                return self.by_char[c]
            if c in DERIVED:
                src, how = DERIVED[c]
                base = self._lookup(src, seen)
                if base:
                    return [_mirror(g) if how == "mirror" else _two_dots(g) for g in base]
            if c in FALLBACK:
                queue.append(FALLBACK[c])
            if c.swapcase() != c:
                queue.append(c.swapcase())
        return None


def _rotate(a, deg):
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0), (nw - w) / 2, (nh - h) / 2


class Composer:
    """Пишет текст строка за строкой. Листов может быть несколько (например, правая и левая
    страница тетради): каждая следующая страница конспекта берёт следующий лист по очереди."""

    def __init__(self, font, papers, opts, rng, start=0):
        self.f, self.o, self.rng = font, opts, rng
        if isinstance(papers, np.ndarray):
            papers = [("лист", papers, 1.0)]
        self.sheets = []
        for name, img, f in papers:
            info = detect_lines(img)
            rows, pitch = plan_rows(
                info, opts["cells_per_line"], opts["descender"], opts["baseline_lift"],
                opts["pitch_px"] * f, opts["top_px"] * f, opts["left_px"] * f, opts["right_px"] * f)
            if not rows:
                raise ValueError(f"Не удалось разметить строки на листе «{name}» — задайте шаг строк вручную.")
            self.sheets.append({"name": name, "img": img, "info": info, "rows": rows, "pitch": pitch})
        self.start = start % len(self.sheets)
        self.ink = np.array(opts["ink_rgb"] or font.ink_rgb, np.float32)[::-1] / 255.0  # BGR
        self.pages, self.page_sheets, self.used_pages = [], [], set()
        self._use_sheet(self.sheets[self.start])
        self._new_page()

    def _use_sheet(self, s):
        self.sheet, self.paper, self.info, self.rows, self.pitch = s, s["img"], s["info"], s["rows"], s["pitch"]
        self.xh = self.o["size"] * self.pitch        # высота строчных букв на листе, px
        self.scale = self.xh / self.f.xheight        # шаблон -> лист

    # ---------- раскладка ----------
    def _new_page(self):
        self._use_sheet(self.sheets[(self.start + len(self.pages)) % len(self.sheets)])
        self.canvas = self.paper.astype(np.float32) / 255.0
        self.pages.append(self.canvas)
        self.page_sheets.append(self.sheet)
        self.row = -1
        self._next_row()

    def _next_row(self):
        self.row += 1
        if self.row >= len(self.rows):
            self._new_page()
            return
        j = self.o["jitter"]
        self.R = self.rows[self.row]
        self.x = self.R["left"] + self.rng.uniform(0, 0.15) * self.xh * j
        self.slope = self.rng.uniform(-1, 1) * 0.004 * j
        self.wave = (self.rng.uniform(0, 6.28), self.rng.uniform(0.5, 1.5))
        self.line_empty = True

    def _word_items(self, word):
        """Для каждой буквы заранее выбираем вариант, размер и отступ — чтобы ширина считалась точно.
        Размер и отступ хранятся относительно строки: слово, перешедшее на другой лист,
        само подстраивается под его линейку."""
        items, prev = [], None
        j, gap = self.o["jitter"], KERN_GAP + self.o["letter_spacing"]
        for ch in word:
            vs = self.f.variants(ch)
            if not vs:
                continue
            g = self.rng.choice([v for v in vs if v is not prev] or vs)
            k = 1 + self.rng.uniform(-1, 1) * 0.04 * j
            items.append((g, k, gap * (1 + self.rng.uniform(-0.4, 0.4) * j)))
            prev = g
        return items

    def _abs(self, items):
        """Относительные размеры букв -> пиксели текущего листа."""
        return [(g, k * self.scale, gap * self.xh) for g, k, gap in items]

    def _advance(self, a, b):
        """Сдвиг от левого края буквы a до левого края буквы b: ставим b так, чтобы
        ближайшие места их чернил были на расстоянии зазора (по контуру, а не по рамкам)."""
        (ga, ka, gap), (gb, kb, _) = a, b
        bbox = ga["w"] * ka + gap
        if not (ga["char"].isalnum() and gb["char"].isalnum()):
            return bbox          # кавычки, скобки, знаки — не подсовываем под соседнюю букву
        _, ra = self.f.profile(ga)
        lb, _ = self.f.profile(gb)
        ok = ~np.isnan(ra) & ~np.isnan(lb)
        if not ok.any():
            return bbox
        adv = float(np.max(ra[ok] * ka - lb[ok] * kb)) + gap
        return min(bbox, max(adv, 0.3 * ga["w"] * ka))

    def _layout(self, items):
        """x каждой буквы слова относительно его начала и полная ширина слова."""
        items = self._abs(items)
        xs = [0.0]
        for a, b in zip(items, items[1:]):
            xs.append(xs[-1] + self._advance(a, b))
        width = max(x + g["w"] * k for x, (g, k, _) in zip(xs, items))
        return xs, width

    def _width(self, items):
        return self._layout(items)[1]

    def paragraph(self, text):
        if not text.strip():
            if not self.line_empty:
                self._next_row()
            self._next_row()
            return
        if not self.line_empty:
            self._next_row()
        self.x += self.o["indent"] * self.xh
        for word in text.split():
            items = self._word_items(word)
            if not items:
                continue
            if not self.line_empty and self.x + self._width(items) > self.R["right"]:
                self._next_row()
            # слово длиннее строки — режем с переносом
            while self.x + self._width(items) > self.R["right"] and len(items) > 1:
                hy = self._word_items("-")
                n = len(items) - 1
                while n > 1 and self.x + self._width(items[:n] + hy) > self.R["right"]:
                    n -= 1
                self._draw_items(items[:n] + hy)
                items = items[n:]
                self._next_row()
            self._draw_items(items)
            self.x += self.o["word_spacing"] * self.xh * (1 + self.rng.uniform(-0.2, 0.2) * self.o["jitter"])
            self.line_empty = False

    # ---------- отрисовка ----------
    def _draw_items(self, items):
        xs, width = self._layout(items)
        x0 = self.x
        for x, (g, k, _) in zip(xs, self._abs(items)):
            self.x = x0 + x
            self._draw_glyph(g, k)
        self.x = x0 + width
        self.line_empty = False

    def _draw_glyph(self, g, k):
        """Рисует символ строго внутри полосы строки. Возвращает фактический масштаб."""
        R, j = self.R, self.o["jitter"]
        dx = self.x - R["left"]
        # строка идёт вдоль линейки на фото (наклон/изгиб листа), полоса строки — вместе с ней
        off = row_offset(R, self.x + g["w"] * k / 2)
        top_b, bot_b = R["top"] + off, R["bottom"] + off
        base = R["base"] + off + self.slope * dx
        base += math.sin(self.wave[0] + dx / (self.xh * 6 * self.wave[1])) * 0.04 * self.xh * j
        base += self.rng.uniform(-1, 1) * 0.04 * self.xh * j
        base = min(max(base, top_b + 1), bot_b - 1)
        # Не влезает по высоте: часть над линией и хвост под линией сжимаем по вертикали
        # отдельно — сама буква остаётся того же размера, что и остальные. Уменьшаем букву
        # целиком, только если одного сжатия мало.
        img, top = g["img"], g["top"]
        up_need = -top * k if top < 0 else 0.0
        dn_need = (top + g["h"]) * k
        s_up = (base - top_b) / up_need if up_need > 0 else 1e9
        s_dn = (bot_b - base) / dn_need if dn_need > 0 else 1e9
        u = max(0.05, min(1.0, s_up / SQUASH_MIN, s_dn / SQUASH_MIN))
        k *= u
        a_up, a_dn = min(1.0, s_up / u), min(1.0, s_dn / u)
        w = max(1, round(g["w"] * k))
        r0 = int(np.clip(-top, 0, img.shape[0]))          # строка базовой линии в картинке буквы
        parts = []
        for part, a_v in ((img[:r0], a_up), (img[r0:], a_dn)):
            ph = round(part.shape[0] * k * a_v)
            if part.shape[0] and ph > 0:
                parts.append(cv2.resize(part, (w, ph), interpolation=cv2.INTER_AREA if k * a_v < 1
                                        else cv2.INTER_CUBIC))
        if not parts:
            return k
        a = np.vstack(parts)
        top_px = top * k * (a_up if top < 0 else a_dn)
        # буквы наклоняем вместе с линейкой
        tilt = -math.degrees(math.atan(row_offset(R, self.x + w) - row_offset(R, self.x))
                             / max(w, 1)) if R.get("curve") is not None else 0.0
        a, ox, oy = _rotate(a, tilt + self.rng.uniform(-2, 2) * j)
        a = cv2.GaussianBlur(a, (0, 0), 0.6) * self.rng.uniform(0.86, 1.0)
        x0 = int(round(self.x - ox))
        y0 = int(round(base + top_px - oy))
        y0 = int(min(max(y0, top_b - oy), bot_b - a.shape[0] + oy))
        self._blend(a, x0, y0)
        return k

    def _blend(self, a, x0, y0):
        """«Умножение», как ручка на бумаге. За красную линию чернила не попадают никогда."""
        H, W = self.canvas.shape[:2]
        h, w = a.shape
        xa, xb = max(0, x0, int(math.ceil(self.R["clip_l"]))), min(W, x0 + w, int(self.R["clip_r"]))
        ya, yb = max(0, y0), min(H, y0 + h)
        if xa >= xb or ya >= yb:
            return
        a = np.clip(a[ya - y0:yb - y0, xa - x0:xb - x0], 0, 1)[..., None]
        reg = self.canvas[ya:yb, xa:xb]
        reg *= 1 - a * (1 - self.ink)
        self.used_pages.add(len(self.pages) - 1)


DEFAULTS = dict(size=0.36, letter_spacing=-0.03, word_spacing=1.1, indent=2.0, jitter=1.0,
                descender=0.32, baseline_lift=0.0, cells_per_line=2, pitch_px=0, top_px=0,
                left_px=0, right_px=0, ink_rgb=None, seed=None)
MIN_PAPER_W = 2000     # маленькие фото листа увеличиваем, иначе буквы выходят «квадратными»


def read_text(path):
    for enc in ("utf-8-sig", "cp1251"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            pass
    raise ValueError("Не удалось прочитать текст (нужна кодировка UTF-8 или Windows-1251).")


IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_sheets(path):
    """Папка -> все фото листов в ней по порядку имён (1_…, 2_…, 10_…); файл -> [файл]."""
    if isinstance(path, (list, tuple)):
        return list(path)
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(IMG_EXT)]
        return sorted(files, key=lambda p: [int(t) if t.isdigit() else t.lower()
                                            for t in re.split(r"(\d+)", os.path.basename(p))])
    return [path]


def _load_sheet(path, log):
    name = os.path.basename(path)
    paper, angle = auto_orient(imread_any(path))
    if angle:
        log(f"«{name}»: лист снят боком — повёрнут на {abs(angle)}°, чтобы строки шли горизонтально.")
    f = 1.0
    if paper.shape[1] < MIN_PAPER_W:
        f = MIN_PAPER_W / paper.shape[1]
        log(f"«{name}»: фото листа маленькое ({paper.shape[1]}×{paper.shape[0]}) — увеличено в {f:.1f} раза. "
            f"Для лучшего вида снимите лист в хорошем качестве.")
        paper = cv2.resize(paper, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return name, paper, f


def render_text(text, font_dir, paper_path, out_dir, log=print, start_sheet=1, **opts):
    """paper_path — фото листа, папка с листами или список фото. Страницы конспекта берут
    листы по очереди (1, 2, …, снова 1); start_sheet — с какого листа начать (с 1)."""
    o = dict(DEFAULTS, **{k: v for k, v in opts.items() if v is not None})
    rng = random.Random(o["seed"])
    font = HandFont(font_dir)
    files = list_sheets(paper_path)
    if not files:
        raise ValueError("В папке листов нет ни одного фото (jpg, png…).")
    comp = Composer(font, [_load_sheet(p, log) for p in files], o, rng, start=int(start_sheet) - 1)
    kinds = {"lined": "линейка", "grid": "клетка", "plain": "без линий"}
    for s in comp.sheets:
        W = s["img"].shape[1]
        sides = ["слева" if (xt + xb) / 2 < W / 2 else "справа" for xt, xb in s["info"]["red"]]
        red = f", красная линия {' и '.join(sides)} — текст за неё не заходит" if sides else ""
        log(f"«{s['name']}»: {kinds[s['info']['kind']]}, строк {len(s['rows'])}, шаг {s['pitch']:.0f} px{red}.")

    text = text.replace("\r", "").replace("\t", " ").replace("…", "...")
    for par in text.split("\n"):
        comp.paragraph(par)
    # пустые страницы в конце (например, от пустых строк в конце текста) не сохраняем
    while len(comp.pages) > 1 and len(comp.pages) - 1 not in comp.used_pages:
        comp.pages.pop()
        comp.page_sheets.pop()

    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        if old.startswith("_lines_preview") and old.endswith(".jpg"):
            os.remove(os.path.join(out_dir, old))
    for i, s in enumerate(comp.sheets, 1):
        imwrite_any(os.path.join(out_dir, f"_lines_preview_{i}.jpg"), preview_rows(s["img"], s["rows"]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    if len(comp.sheets) > 1:
        log("Листы по страницам: " + ", ".join(f"стр. {i} — «{s['name']}»"
                                                for i, s in enumerate(comp.page_sheets, 1)))
        last = next(i for i, s in enumerate(comp.sheets) if s is comp.page_sheets[-1])
        nxt = (last + 1) % len(comp.sheets) + 1
        log(f"Следующий конспект, чтобы продолжить по очереди, начните с листа № {nxt}.")
    paths, pil = [], []
    for i, page in enumerate(comp.pages, 1):
        img = (np.clip(page, 0, 1) * 255).astype(np.uint8)
        p = os.path.join(out_dir, f"page_{i:02d}.jpg")
        imwrite_any(p, img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        paths.append(p)
        pil.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    # старые страницы от прошлых запусков удаляем, чтобы не путались
    k = len(paths) + 1
    while os.path.exists(os.path.join(out_dir, f"page_{k:02d}.jpg")):
        os.remove(os.path.join(out_dir, f"page_{k:02d}.jpg"))
        k += 1
    pil[0].save(os.path.join(out_dir, "konspekt.pdf"), save_all=True, append_images=pil[1:])
    if font.missing:
        log("Нет в шрифте (пропущены): " + " ".join(sorted(font.missing)))
    if len(paths) > 1:
        log(f"Готово: {len(paths)} стр. — page_01.jpg … page_{len(paths):02d}.jpg, "
            f"и все вместе в konspekt.pdf → {out_dir}")
    else:
        log(f"Готово: 1 стр. → {out_dir}")
    return paths
