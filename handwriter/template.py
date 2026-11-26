"""Генерация листов-шаблонов: распечатать, заполнить своим почерком, сфотографировать.

Каждый лист — A4 (300 dpi) с четырьмя ArUco-метками по углам. По меткам программа
потом сама находит лист на фото, выравнивает перспективу и узнаёт номер страницы,
поэтому фото можно делать в любом порядке.
"""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---- геометрия листа (пиксели при 300 dpi) ----
PAGE_W, PAGE_H = 2480, 3508
MARKER = 160          # размер метки
MARKER_OFF = 90       # отступ метки от края листа
GRID_X0, GRID_X1 = 280, PAGE_W - 280
GRID_Y0, GRID_Y1 = 360, PAGE_H - 280
COLS, ROWS = 7, 9
LABEL_H = 55          # полоска с подсказкой над рамкой
BASELINE_FRAC = 0.68  # где в рамке проходит базовая линия (доля высоты рамки)

# 4X4_1000 включает в себя 4X4_250 (первые 250 меток совпадают) — старые распечатки тоже узнаются
ARUCO_DICT = cv2.aruco.DICT_4X4_1000

CHARSETS = {
    "ru_lower": "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    "ru_upper": "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
    "digits": "0123456789",
    "punct": ".,:;!?-()«»\"'/+=%№*<>",
    "en_lower": "abcdefghijklmnopqrstuvwxyz",
    "en_upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "ua": "іїєґІЇЄҐ",
}
DEFAULT_SETS = ["ru_lower", "ru_upper", "digits", "punct", "en_lower", "en_upper", "ua"]


def load_font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        for path in (name, os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", name)):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default(size)


def build_char_list(sets=None, extra=""):
    chars = []
    for s in sets or DEFAULT_SETS:
        chars.extend(CHARSETS[s])
    chars.extend(extra)
    seen, out = set(), []
    for c in chars:
        if c not in seen and not c.isspace():
            seen.add(c)
            out.append(c)
    return out


def cell_geometry(index_on_page):
    """Рамка для письма [x, y, w, h] и абсолютная y базовой линии."""
    col, row = index_on_page % COLS, index_on_page // COLS
    cw = (GRID_X1 - GRID_X0) / COLS
    ch = (GRID_Y1 - GRID_Y0) / ROWS
    cx, cy = GRID_X0 + col * cw, GRID_Y0 + row * ch
    x, y = int(cx + 10), int(cy + LABEL_H)
    w, h = int(cw - 20), int(ch - LABEL_H - 12)
    return [x, y, w, h], int(y + h * BASELINE_FRAC)


def marker_positions():
    """Левые верхние углы меток: TL, TR, BR, BL."""
    a, b = MARKER_OFF, PAGE_W - MARKER_OFF - MARKER
    c = PAGE_H - MARKER_OFF - MARKER
    return [(a, a), (b, a), (b, c), (a, c)]


def make_layout(chars, variants=2):
    per_page = COLS * ROWS
    items = [(c, v) for c in chars for v in range(variants)]
    pages = []
    for p in range(0, len(items), per_page):
        page_no = len(pages)
        cells = []
        for i, (c, v) in enumerate(items[p:p + per_page]):
            box, base = cell_geometry(i)
            cells.append({"char": c, "variant": v, "box": box, "baseline": base})
        pages.append({"page": page_no, "marker_ids": [page_no * 4 + k for k in range(4)], "cells": cells})
    return {"version": 1, "page_size": [PAGE_W, PAGE_H], "marker": MARKER,
            "marker_pos": marker_positions(), "pages": pages}


def _dotted_hline(draw, x0, x1, y, color, step=14, r=1):
    for x in range(int(x0) + 6, int(x1) - 6, step):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def render_page(layout, page):
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(img)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    for mid, (mx, my) in zip(page["marker_ids"], layout["marker_pos"]):
        m = cv2.aruco.generateImageMarker(dictionary, mid, MARKER)
        img.paste(Image.fromarray(m).convert("RGB"), (mx, my))

    title = page.get("title") or f"лист {page['page'] + 1} из {len(layout['pages'])}"
    d.text((GRID_X0, 120), f"Шаблон почерка — {title}", fill=(0, 0, 0), font=load_font(52))
    d.text((GRID_X0, 200),
           "Пишите по одному символу в рамке ручкой, которой пишете конспекты. Низ букв — на пунктир, "
           "хвосты (у, р, д, щ) — ниже пунктира. Не заезжайте на рамку, метки по углам не закрывайте.",
           fill=(90, 90, 90), font=load_font(30))
    d.text((GRID_X0, 250),
           "Пишите в своём обычном размере: маленькие буквы — маленькими, заглавные — крупнее.",
           fill=(90, 90, 90), font=load_font(30))

    lab_font, small = load_font(42), load_font(26)
    for cell in page["cells"]:
        x, y, w, h = cell["box"]
        d.rectangle([x, y, x + w, y + h], outline=(150, 150, 150), width=2)
        _dotted_hline(d, x, x + w, cell["baseline"], (195, 195, 195))
        label = cell["char"] if cell["char"] != " " else "␣"
        d.text((x + 4, y - LABEL_H + 2), label, fill=(110, 110, 110), font=lab_font)
        d.text((x + w - 50, y - LABEL_H + 18), f"{cell['variant'] + 1}", fill=(170, 170, 170), font=small)
    return img


MAX_PAGE_ID = 1000 // 4    # столько листов умещается в словарь меток


def _load_layout(out_dir):
    path = os.path.join(out_dir, "layout.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _add_pages(out_dir, layout, new_pages, title_fmt, pdf_name):
    """Даёт новым листам свободные номера меток, рисует их и сохраняет разметку.

    Уже напечатанные листы из разметки никогда не удаляются — их фото всегда распознаются.
    """
    start = max((p["page"] for p in layout["pages"]), default=-1) + 1
    for i, p in enumerate(new_pages):
        p["page"] = start + i
        p["marker_ids"] = [p["page"] * 4 + k for k in range(4)]
        p["title"] = title_fmt.format(i=i + 1, n=len(new_pages))
    if start + len(new_pages) > MAX_PAGE_ID:
        raise ValueError("В этой папке почерка слишком много листов шаблона — создайте новую папку почерка.")
    layout["pages"].extend(new_pages)
    images = [render_page(layout, p) for p in new_pages]
    for im, p in zip(images, new_pages):
        im.save(os.path.join(out_dir, f"template_page_{p['page'] + 1}.png"))
    pdf = os.path.join(out_dir, pdf_name)
    images[0].save(pdf, save_all=True, append_images=images[1:], resolution=300)
    with open(os.path.join(out_dir, "layout.json"), "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False)
    return pdf, len(images)


def generate_template(out_dir, sets=None, variants=2, extra=""):
    """Создаёт template.pdf для печати. Если шаблон в папке уже был, старые листы остаются
    в разметке (помечаются как прежние), а новые получают новые номера меток."""
    os.makedirs(out_dir, exist_ok=True)
    fresh = make_layout(build_char_list(sets, extra), variants)
    layout = _load_layout(out_dir)
    if layout is None:
        layout = dict(fresh, pages=[])
    else:
        for p in layout["pages"]:
            p["old"] = True
    return _add_pages(out_dir, layout, fresh["pages"], "лист {i} из {n}", "template.pdf")


def append_template(out_dir, sets=None, variants=2, extra=""):
    """Дописывает к шаблону листы только с недостающими символами (старые фото остаются годными).

    Возвращает (путь к pdf, число новых листов) или (None, 0), если добавлять нечего.
    """
    layout = _load_layout(out_dir)
    if layout is None:
        raise ValueError("Шаблона ещё нет — сначала создайте его.")
    have = {c["char"] for p in layout["pages"] if not p.get("old") for c in p["cells"]}
    new = [c for c in build_char_list(sets, extra) if c not in have]
    if not new:
        return None, 0
    pages = make_layout(new, variants)["pages"]
    first = max(p["page"] for p in layout["pages"]) + 2
    return _add_pages(out_dir, layout, pages, "дополнительный лист {i} из {n}", f"template_dop_list_{first}.pdf")
