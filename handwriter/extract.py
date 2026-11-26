"""Вырезание символов почерка из фото/сканов заполненных шаблонов."""
import json
import os

import cv2
import numpy as np

from .template import ARUCO_DICT, BASELINE_FRAC, COLS, render_page

INSET = 9          # отступ внутрь от рамки, px шаблона


def imread_any(path):
    """cv2.imread не понимает русские пути на Windows — читаем через numpy."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось открыть изображение: {path}")
    return img


def imwrite_any(path, img, params=None):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise ValueError(f"Не удалось сохранить {path}")
    buf.tofile(path)


def _find_pages(img, layout):
    """Находит на фото листы шаблона.

    Возвращает ({номер листа: (выровненное изображение, размер метки в px фото)},
                номера листов, метки которых видны, но которых нет в разметке этой папки).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}, set()
    known = {p["page"] for p in layout["pages"]}
    m = layout["marker"]
    by_page, unknown = {}, set()
    for c, mid in zip(corners, ids.flatten()):
        page, pos = int(mid) // 4, int(mid) % 4
        if page not in known:
            unknown.add(page)
            continue
        mx, my = layout["marker_pos"][pos]
        dst = np.float32([[mx, my], [mx + m, my], [mx + m, my + m], [mx, my + m]])
        by_page.setdefault(page, ([], []))
        by_page[page][0].append(c.reshape(4, 2))
        by_page[page][1].append(dst)
    W, H = layout["page_size"]
    out = {}
    for page, (src, dst) in by_page.items():
        if len(src) < 2:      # одной метки мало для надёжного выравнивания
            continue
        side = float(np.mean([np.linalg.norm(s[0] - s[1]) for s in src]))
        Hm, _ = cv2.findHomography(np.concatenate(src), np.concatenate(dst), cv2.RANSAC, 5.0)
        if Hm is not None:
            out[page] = (cv2.warpPerspective(img, Hm, (W, H), flags=cv2.INTER_CUBIC,
                                             borderValue=(255, 255, 255)), side)
    return out, unknown


def _ink_alpha(warped):
    """Карта «чернильности» 0..1 с выравниванием освещения."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    bg = cv2.GaussianBlur(bg, (0, 0), 15)
    dark = np.clip(1.0 - gray / np.maximum(bg, 1), 0, 1)
    strong = dark[dark > 0.25]
    hi = max(0.35, float(np.percentile(strong, 90)) * 0.85) if strong.size > 500 else 0.5
    lo = 0.13
    return np.clip((dark - lo) / (hi - lo), 0, 1)


def _line_maps(dark, L):
    """Только длинные прямые линии (рамки): штрихи букв такой длины не бывают."""
    m = (dark > 0.15).astype(np.uint8)
    lh = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))).astype(np.float32)
    lv = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))).astype(np.float32)
    return lh, lv


def _refine_boxes(alpha, cells, page_img, margin=45, search=70):
    """Находит настоящие рамки ячеек: лист на фото бывает изогнут, и рамки «съезжают».

    1) кусок чистого шаблона вокруг ячейки (рамка + края соседних рамок) ищем на фото — это
       однозначное грубое совпадение; 2) каждую сторону рамки точно «прищёлкиваем» к линии.
    """
    w0 = cells[0]["box"][2]
    L = int(w0 * 0.5)
    lh, lv = _line_maps(alpha, L)
    P = cv2.GaussianBlur(np.maximum(lh, lv), (0, 0), 3)
    tdark = 1.0 - cv2.cvtColor(np.asarray(page_img), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    th, tv = _line_maps(tdark, L)
    T = cv2.GaussianBlur(np.maximum(th, tv), (0, 0), 3)
    H, W = alpha.shape
    found = []
    for cell in cells:
        x, y, w, h = cell["box"]
        tx0, ty0 = max(0, x - margin), max(0, y - margin)
        tpl = T[ty0:y + h + margin, tx0:x + w + margin]
        sx0, sy0 = max(0, tx0 - search), max(0, ty0 - search)
        reg = P[sy0:min(H, ty0 + tpl.shape[0] + search), sx0:min(W, tx0 + tpl.shape[1] + search)]
        if reg.shape[0] <= tpl.shape[0] or reg.shape[1] <= tpl.shape[1] or reg.max() == 0:
            found.append((0, 0, 0.0))
            continue
        res = cv2.matchTemplate(reg, tpl, cv2.TM_CCORR_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        found.append((sx0 + loc[0] - tx0, sy0 + loc[1] - ty0, float(score)))
    good = [f[:2] for f in found if f[2] > 0.35]
    med = tuple(np.median(np.array(good), axis=0).astype(int)) if good else (0, 0)
    boxes = []
    for i, (cell, (dx, dy, s)) in enumerate(zip(cells, found)):
        x, y, w, h = cell["box"]
        if s <= 0.35:   # рамка почти не видна — берём сдвиг соседей
            near = [f[:2] for f in found[max(0, i - 8):i + 9] if f[2] > 0.35]
            dx, dy = tuple(np.median(np.array(near), axis=0).astype(int)) if near else med
        x, y = x + dx, y + dy
        rows = lh[:, max(0, x + w // 8): x + w - w // 8].mean(axis=1)
        cols = lv[max(0, y + h // 8): y + h - h // 8, :].mean(axis=0)

        def snap(prof, pos, r):
            lo = max(0, pos - r)
            seg = prof[lo:pos + r + 1]
            return lo + int(np.argmax(seg)) if seg.size and seg.max() > 0.2 else pos

        # по вертикали до соседних рамок далеко (подпись), по горизонтали — всего ~20 px
        t, b = snap(rows, y, 25), snap(rows, y + h, 25)
        l, rr = snap(cols, x, 9), snap(cols, x + w, 9)
        if not (0.85 * h < b - t < 1.18 * h):
            t, b = y, y + h
        if not (0.85 * w < rr - l < 1.18 * w):
            l, rr = x, x + w
        boxes.append([int(l), int(t), int(rr - l), int(b - t)])
    return _smooth_boxes(cells, boxes)


def _smooth_boxes(cells, boxes, tol=14):
    """Изгиб листа плавный: если ячейка «прыгнула» относительно соседей — берём их сдвиг."""
    off = np.array([[b[0] - c["box"][0], b[1] - c["box"][1], b[2] - c["box"][2], b[3] - c["box"][3]]
                    for c, b in zip(cells, boxes)], dtype=float)
    pos = [(i // COLS, i % COLS) for i in range(len(cells))]
    out = []
    for i, (c, b) in enumerate(zip(cells, boxes)):
        nb = [j for j in range(len(cells)) if j != i
              and abs(pos[j][0] - pos[i][0]) <= 1 and abs(pos[j][1] - pos[i][1]) <= 1]
        if len(nb) >= 2:
            med = np.median(off[nb], axis=0)
            if np.abs(off[i] - med).max() > tol:
                x, y, w, h = c["box"]
                b = [int(x + med[0]), int(y + med[1]), int(w + med[2]), int(h + med[3])]
        out.append(b)
    return out


SINGLE_STROKE = set(".,-'")    # такие знаки — один штрих: берём самый заметный, остальное мусор


def clean_alpha(a, edges, crop_w, crop_h, v_frac=0.3, single=False):
    """Оставляет сам символ: убирает пылинки, бледные пятна/тени и остатки рамки у краёв.

    edges — где в координатах `a` проходят края ячейки (top/bottom/left/right), None если неизвестно.
    """
    binary = (a > 0.3).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    band = 16
    best, best_score = None, 0
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if area < 25:
            continue
        m = lab == i
        vals = a[m]
        # бледное пятно, тень, край надрыва бумаги: у настоящих чернил ядро почти 1.0
        if np.percentile(vals, 95) < 0.75 or vals.mean() < 0.5:
            continue
        thin_h = ch <= 14 and cw >= 0.3 * crop_w
        thin_v = cw <= 14 and ch >= v_frac * crop_h
        near_t = edges.get("top") is not None and y <= edges["top"] + band
        near_b = edges.get("bottom") is not None and y + ch >= edges["bottom"] - band
        near_l = edges.get("left") is not None and x <= edges["left"] + band
        near_r = edges.get("right") is not None and x + cw >= edges["right"] - band
        if (thin_h and (near_t or near_b)) or (thin_v and (near_l or near_r)):
            continue                                  # кусок рамки
        if edges.get("top") is not None and y + ch <= edges["top"] + band + 8 and ch < 0.2 * crop_h:
            continue                                  # кусочек печатной подсказки над рамкой
        if edges.get("bottom") is not None and y >= edges["bottom"] - band - 8 and ch < 0.2 * crop_h:
            continue                                  # подсказка следующего ряда снизу
        if single:
            score = area * vals.mean()
            if score > best_score:
                best, best_score = m, score
            continue
        keep[m] = 1
    if single and best is not None:
        keep[best] = 1
    if not keep.any():
        return None
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))
    return a * keep


def trim(a, pad=3):
    """Обрезает по чернилам. Возвращает (кусок, y0, x0) или None."""
    ys, xs = np.where(a > 0.15)
    if ys.size == 0:
        return None
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + 1 + pad)
    return a[y0:y1, x0:x1], y0, x0


def extract_font(photo_paths, font_dir, log=print):
    """Обрабатывает фото шаблонов и (до)собирает шрифт в font_dir/font.json."""
    with open(os.path.join(font_dir, "layout.json"), encoding="utf-8") as f:
        layout = json.load(f)
    gdir, ddir = os.path.join(font_dir, "glyphs"), os.path.join(font_dir, "debug")
    os.makedirs(gdir, exist_ok=True)
    os.makedirs(ddir, exist_ok=True)

    font_path = os.path.join(font_dir, "font.json")
    font = {"version": 1, "glyphs": []}
    if os.path.exists(font_path):
        with open(font_path, encoding="utf-8") as f:
            font = json.load(f)
    glyphs = {(g["char"], g["variant"]): g for g in font["glyphs"]}
    ink_samples = []
    _, _, bw, bh = layout["pages"][0]["cells"][0]["box"]
    crop_w, crop_h = bw - 2 * INSET, bh - 2 * INSET

    by_id = {p["page"]: p for p in layout["pages"]}
    for path in photo_paths:
        name = os.path.basename(path)
        pages, unknown = _find_pages(imread_any(path), layout)
        if not pages:
            if unknown:
                log(f"✗ {name}: метки видны, но это лист, которого нет в шаблоне этой папки почерка "
                    f"(номер {', '.join(str(u + 1) for u in sorted(unknown))}). Лист печатали из другой "
                    f"папки почерка или до пересоздания шаблона старой версией программы.")
            else:
                log(f"✗ {name}: метки по углам не найдены (весь лист должен быть в кадре, квадраты не закрыты)")
            continue
        for pno, (warped, side) in pages.items():
            if side < 90:
                log(f"  ! {name}: фото маленькое (метка {side:.0f}px) — буквы будут размытыми. "
                    f"Отправляйте фото «файлом», без сжатия, или снимайте ближе.")
            alpha = _ink_alpha(warped)
            dbg = warped.copy()
            found = 0
            page = by_id[pno]
            cells = page["cells"]
            boxes = _refine_boxes(alpha, cells, render_page(layout, page))
            for cell, (x, y, w, h) in zip(cells, boxes):
                # найденную рамку приводим к шаблонному размеру: все буквы в одном масштабе
                nw, nh = cell["box"][2], cell["box"][3]
                x, y = max(0, x), max(0, y)
                full_a = cv2.resize(alpha[y:y + h, x:x + w], (nw, nh), interpolation=cv2.INTER_AREA)
                full_c = cv2.resize(warped[y:y + h, x:x + w], (nw, nh), interpolation=cv2.INTER_AREA)
                base = cell["baseline"] - cell["box"][1]          # в координатах ячейки
                region_a = full_a[INSET:nh - INSET, INSET:nw - INSET]
                ch_, cw_ = region_a.shape
                a = clean_alpha(region_a, {"top": 0, "bottom": ch_, "left": 0, "right": cw_}, crop_w, crop_h,
                                single=cell["char"] in SINGLE_STROKE)
                ok = a is not None and (a > 0.5).sum() > 40
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 180, 0) if ok else (0, 0, 255), 3)
                if not ok:
                    continue
                crop, y0, x0 = trim(a)
                c, v = cell["char"], cell["variant"]
                fname = f"u{ord(c):04x}_v{v}.png"
                imwrite_any(os.path.join(gdir, fname), (crop * 255).astype(np.uint8))
                glyphs[(c, v)] = {
                    "char": c, "variant": v, "file": fname,
                    "w": int(crop.shape[1]), "h": int(crop.shape[0]),
                    # верх символа относительно базовой линии (отрицательный = выше линии)
                    "top": int(round(INSET + y0 - base)), "clean": 2,
                }
                region = full_c[INSET:nh - INSET, INSET:nw - INSET]
                sel = a > 0.75
                if sel.sum() > 30:
                    ink_samples.append(region[sel].mean(axis=0))
                found += 1
            imwrite_any(os.path.join(ddir, f"page_{pno + 1}.jpg"),
                        cv2.resize(dbg, None, fx=0.35, fy=0.35), [cv2.IMWRITE_JPEG_QUALITY, 80])
            log(f"✓ {name}: {page.get('title') or f'лист {pno + 1}'}, символов: {found} из {len(cells)}")

    base_off = layout["pages"][0]["cells"][0]["baseline"] - layout["pages"][0]["cells"][0]["box"][1]
    font.update(box_h=bh, baseline_frac=BASELINE_FRAC, crop_w=crop_w, crop_h=crop_h,
                crop_top=INSET - base_off, crop_bottom=bh - INSET - base_off)
    if ink_samples:
        b, g, r = np.median(np.array(ink_samples), axis=0)
        font["ink_rgb"] = [int(r), int(g), int(b)]
    font["glyphs"] = sorted(glyphs.values(), key=lambda g: (g["char"], g["variant"]))
    with open(font_path, "w", encoding="utf-8") as f:
        json.dump(font, f, ensure_ascii=False, indent=1)

    have = {g["char"] for g in font["glyphs"]}
    wanted = {c["char"] for p in layout["pages"] for c in p["cells"]}
    missing = "".join(sorted(wanted - have))
    log(f"Итого в шрифте: {len(font['glyphs'])} начертаний, {len(have)} разных символов.")
    if missing:
        log(f"Нет символов: {missing}")
    return font
