"""Анализ фото чистого листа: где линейки/клетки, поля, шаг строк."""
import cv2
import numpy as np


def _line_centers(profile, thr):
    rows = np.where(profile > thr)[0]
    if rows.size == 0:
        return []
    groups, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - prev > 2:
            groups.append((start + prev) / 2)
            start = r
        prev = r
    groups.append((start + prev) / 2)
    return groups


def _regular(centers, min_count=5):
    """Оставляет только линии с регулярным шагом, возвращает (линии, шаг)."""
    if len(centers) < min_count:
        return None, None
    c = np.array(centers, dtype=float)
    diffs = np.diff(c)
    pitch = float(np.median(diffs))
    if pitch < 8:
        return None, None
    good = [c[0]]
    for y in c[1:]:
        if y - good[-1] > 0.6 * pitch:
            good.append(y)
    # уточняем шаг и равномерно восстанавливаем пропущенные линии
    good = np.array(good)
    idx = np.round((good - good[0]) / pitch)
    if len(np.unique(idx)) >= 3:
        pitch, offset = np.polyfit(idx, good, 1)
    else:
        offset = good[0]
    n = int(idx.max()) + 1
    return [offset + k * pitch for k in range(n)], float(pitch)


N_STRIPS = 20


def _strip_peaks(piece, n=N_STRIPS, min_sep=6.0, max_thick=None, gray=None, side=0):
    """Делит лист на вертикальные полосы и в каждой ищет y линеек.

    На фото линейки почти всегда чуть наклонены и изогнуты (лист не идеально ровный,
    тетрадь выгибается у корешка), поэтому линию ищем по кусочкам, а не по всей ширине.
    Если дана яркость (gray), линия должна иметь бумагу с обеих сторон — так отсекается
    край листа (с одной стороны стол или тень).
    """
    h, w = piece.shape
    sw = w / n
    d = int(round(side))
    out = []
    for s in range(n):
        a, b = int(s * sw), int((s + 1) * sw)
        prof = piece[:, a:b].mean(axis=1) / 255.0
        sm = np.convolve(prof, np.ones(5) / 5, "same")
        cand = np.where((sm > 0.08) & (sm >= np.roll(sm, 1)) & (sm >= np.roll(sm, -1)))[0]
        keep = []
        for y in sorted(cand, key=lambda y: -sm[y]):
            if all(abs(y - k) >= min_sep for k in keep):
                keep.append(y)
        ys = []
        for y in sorted(keep):
            # центр масс и толщина: широкая полоса — это край листа/тень, а не линейка
            lo, hi = y, y
            half = sm[y] / 2
            while lo > 0 and sm[lo - 1] > half:
                lo -= 1
            while hi < h - 1 and sm[hi + 1] > half:
                hi += 1
            if max_thick and hi - lo + 1 > max_thick:
                continue
            if gray is not None and d:
                if y - d - 2 < 0 or y + d + 3 > h:
                    continue
                up = float(np.median(gray[y - d - 2:y - d + 3, a:b]))
                dn = float(np.median(gray[y + d - 2:y + d + 3, a:b]))
                if min(up, dn) < 0.82 * max(up, dn):
                    continue
            seg = prof[lo:hi + 1]
            ys.append(float((np.arange(lo, hi + 1) * seg).sum() / max(seg.sum(), 1e-6)))
        out.append(((a + b) / 2.0, np.array(ys)))
    return out, sw


def _track(strips, pitch):
    """Сшивает найденные в полосах кусочки в линии. Возвращает [(xs, ys)] наблюдённых точек."""
    n = len(strips)
    start = max(range(n), key=lambda s: (len(strips[s][1]), -abs(s - n / 2)))
    tracks = [{"pts": {start: y}} for y in strips[start][1]]
    tol = 0.3 * pitch

    def predict(t, s, step=0):
        ks = sorted(t["pts"], key=lambda k: abs(k - s))[:2]
        if len(ks) == 2:
            k1, k2 = ks
            return t["pts"][k1] + (t["pts"][k1] - t["pts"][k2]) / (k1 - k2) * (s - k1)
        # у линии одна точка — наклон берём у соседних линий
        slopes = []
        for o in tracks:
            ok = sorted(o["pts"])
            if len(ok) >= 2:
                slopes.append((o["pts"][ok[-1]] - o["pts"][ok[0]]) / (ok[-1] - ok[0]))
        return t["pts"][ks[0]] + (float(np.median(slopes)) if slopes else 0.0) * (s - ks[0])

    for step in (1, -1):
        s = start + step
        while 0 <= s < n:
            ys = strips[s][1]
            used = set()
            for t in tracks:
                last = max(t["pts"]) if step > 0 else min(t["pts"])
                if abs(s - last) > 3:            # линия оборвалась (край листа) — дальше не ведём
                    continue
                if not ys.size:
                    continue
                p = predict(t, s, step)
                i = int(np.argmin(np.abs(ys - p)))
                if abs(ys[i] - p) < tol and i not in used:
                    t["pts"][s] = float(ys[i])
                    used.add(i)
            for i, y in enumerate(ys):
                if i not in used and all(abs(y - predict(t, s, step)) > 0.5 * pitch for t in tracks):
                    tracks.append({"pts": {s: float(y)}})
            s += step
    out = []
    for t in tracks:
        ks = sorted(t["pts"])
        if len(ks) < max(3, n // 4):
            continue
        xs = np.array([strips[k][0] for k in ks])
        ys = np.array([t["pts"][k] for k in ks])
        # выбросы относительно плавной кривой убираем, остальное слегка сглаживаем
        fit = np.polyval(np.polyfit(xs, ys, 2 if len(ks) >= 5 else 1), xs)
        ok = np.abs(ys - fit) < 0.12 * pitch
        xs, ys = xs[ok], ys[ok]
        if len(xs) < 3:
            continue
        sm = ys.copy()
        sm[1:-1] = (ys[:-2] + ys[1:-1] + ys[2:]) / 3
        out.append((xs, sm))
    return out


def curve_y(curve, x):
    """y линии в точке x; за краями наблюдений — продолжаем по последнему наклону."""
    xs, ys = curve
    if len(xs) == 1:
        return float(ys[0])
    if x < xs[0]:
        return float(ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x - xs[0]))
    if x > xs[-1]:
        return float(ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (x - xs[-1]))
    return float(np.interp(x, xs, ys))


def curve_ys(curve, x):
    """То же, что curve_y, для массива x."""
    xs, ys = curve
    x = np.asarray(x, dtype=float)
    if len(xs) == 1:
        return np.full_like(x, ys[0])
    out = np.interp(x, xs, ys)
    lo, hi = x < xs[0], x > xs[-1]
    out[lo] = ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x[lo] - xs[0])
    out[hi] = ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (x[hi] - xs[-1])
    return out


def _extent(piece, curve, pitch):
    """Докуда по x линейка реально напечатана (идём от середины, пока линия есть под кривой)."""
    h, w = piece.shape
    xs = np.arange(w)
    yi = np.round(curve_ys(curve, xs)).astype(int)
    ok = (yi >= 3) & (yi < h - 3)
    hit = np.zeros(w, bool)
    for d in range(-3, 4):
        hit[ok] |= piece[yi[ok] + d, xs[ok]] > 0
    gap = max(4, int(0.4 * pitch))
    xc = int(np.clip(np.median(curve[0]), 0, w - 1))
    ends = []
    for step in (-1, 1):
        x, last, miss = xc, xc, 0
        while 0 <= x + step < w and miss <= gap:
            x += step
            if hit[x]:
                last, miss = x, 0
            else:
                miss += 1
        ends.append(last)
    return ends


def _paper_mask(img, seeds, width=300):
    """Маска «где бумага». Растим область от точек на листе (там, где линейки):
    плавная тень на бумаге пропускается, а резкий край листа (стол, ткань, фон) — нет.
    Возвращает (маска, масштаб к исходному фото)."""
    s = width / img.shape[1]
    sm = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sm = cv2.medianBlur(cv2.medianBlur(sm, 5), 5)        # линейки, поле, скрепки — убираем
    h, w = sm.shape[:2]
    ff = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in seeds:
        xi, yi = int(np.clip(x * s, 0, w - 1)), int(np.clip(y * s, 0, h - 1))
        if ff[yi + 1, xi + 1]:
            continue
        cv2.floodFill(sm, ff, (xi, yi), 0, (5, 5, 5), (5, 5, 5), 4 | (1 << 8) | cv2.FLOODFILL_MASK_ONLY)
    mask = cv2.morphologyEx(ff[1:-1, 1:-1], cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # дырки внутри листа (скрепки, пятна) — тоже лист
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in cnts if cv2.contourArea(c) > 0.02 * h * w]
    out = np.zeros_like(mask)
    cv2.drawContours(out, big, -1, 1, -1)
    if out.sum() < 0.1 * h * w:          # не получилось — не ограничиваем
        out[:] = 1
    return out, s


def _on_paper(paper, xs, ys):
    """Доля точек (xs, ys) исходного фото, попавших на бумагу."""
    if paper is None:
        return 1.0
    mask, s = paper
    h, w = mask.shape
    xi = np.clip(np.round(np.asarray(xs) * s).astype(int), 0, w - 1)
    yi = np.round(np.asarray(ys) * s).astype(int)
    inside = (yi >= 0) & (yi < h)
    return float((inside & (mask[np.clip(yi, 0, h - 1), xi] > 0)).mean())


def _cut_at_fold(vpiece, curves, reds, x0, x1, pitch):
    """Ищет длинную вертикальную тёмную линию (сгиб/корешок, край соседней страницы)
    внутри области линеек и оставляет ту часть, где больше листа."""
    h, w = vpiece.shape
    ya = int(max(0, min(curve_y(c, (x0 + x1) / 2) for c in curves)))
    yb = int(min(h, max(curve_y(c, (x0 + x1) / 2) for c in curves)))
    if yb - ya < 5 * pitch:
        return x0, x1
    band = cv2.dilate(vpiece[ya:yb], np.ones((1, 5), np.uint8))
    cov = (band > 0).mean(axis=0)
    margin = int(1.5 * pitch)
    best = None
    for x in range(x0 + 2, x1 - 2):
        if cov[x] < 0.5 or any(abs(x - (xt + xb) / 2) < 15 for xt, xb in reds):
            continue
        if x - x0 < margin or x1 - x < margin:     # край самой страницы — ничего не режем
            continue
        if best is None or cov[x] > cov[best]:
            best = x
    if best is None:
        return x0, x1
    return (best + 3, x1) if best - x0 < x1 - best else (x0, best - 3)


def _clean_tracks(curves, pitch, w):
    """Сортирует линии, убирает слишком близкие (дубли) и дорисовывает пропущенные."""
    xm = w / 2
    curves = sorted(curves, key=lambda c: curve_y(c, xm))
    good = []
    for c in curves:
        if good and curve_y(c, xm) - curve_y(good[-1], xm) < 0.6 * pitch:
            if len(c[0]) > len(good[-1][0]):
                good[-1] = c
            continue
        good.append(c)
    filled = []
    for c in good:
        if filled:
            prev = filled[-1]
            gap = (curve_y(c, xm) - curve_y(prev, xm)) / pitch
            k = int(round(gap))
            if 2 <= k <= 4 and abs(gap - k) < 0.25:
                xs = np.linspace(max(prev[0][0], c[0][0]), min(prev[0][-1], c[0][-1]), 12)
                for j in range(1, k):
                    ys = [curve_y(prev, x) + (curve_y(c, x) - curve_y(prev, x)) * j / k for x in xs]
                    filled.append((xs, np.array(ys)))
        filled.append(c)
    return filled


def detect_lines(img):
    """Ищет линейки или клетку (с учётом наклона и изгиба), границы листа и красное поле."""
    H, W = img.shape[:2]
    scale = 1400.0 / W if W > 1400 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else img
    h, w = small.shape[:2]
    mn = small.min(axis=2)  # синие/серые/фиолетовые линии темнее всего в «минимальном» канале
    thr = cv2.adaptiveThreshold(mn, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    piece = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 50), 1)))

    # 1) шаг линеек — по полосам (в каждой линии почти горизонтальны)
    strips, sw = _strip_peaks(piece)
    diffs = np.concatenate([np.diff(ys) for _, ys in strips if len(ys) >= 5] or [np.array([])])
    curves, pitch = [], None
    if diffs.size >= 10:
        pitch = float(np.median(diffs))
        if pitch >= 8:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            strips, sw = _strip_peaks(piece, min_sep=0.55 * pitch, max_thick=max(8.0, 0.25 * pitch),
                                      gray=gray, side=0.4 * pitch)
            diffs = np.concatenate([np.diff(ys) for _, ys in strips if len(ys) >= 5] or [np.array([])])
            if diffs.size >= 10:
                d = diffs[(diffs > 0.7 * pitch) & (diffs < 1.3 * pitch)]
                pitch = float(np.median(d)) if d.size else pitch
                curves = _clean_tracks(_track(strips, pitch), pitch, w)
    if len(curves) < 5:
        curves, pitch = [], None

    # 2) где по ширине напечатаны линейки — это и есть лист (а не стол или соседняя страница)
    x0, x1 = 0, w
    if curves:
        ext = np.array([_extent(piece, c, pitch) for c in curves])
        x0, x1 = int(np.median(ext[:, 0])), int(np.median(ext[:, 1]))
        if x1 - x0 < 0.3 * w:
            x0, x1 = 0, w

    # клетка или линейка: в клетке есть и вертикальные линии с тем же шагом
    # (тоже ищем по кусочкам — у корешка клетка сильно изгибается)
    vpiece = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 50))))
    vstrips, _ = _strip_peaks(np.ascontiguousarray(vpiece[:, x0:x1].T), min_sep=0.55 * pitch if pitch else 6.0)
    vd = np.concatenate([np.diff(xs) for _, xs in vstrips if len(xs) >= 8] or [np.array([])])
    vpitch = float(np.median(vd)) if vd.size >= 20 else None
    is_grid = pitch is not None and (
        (vpitch is not None and abs(vpitch - pitch) < 0.2 * pitch)
        or (vpitch is None and pitch < (x1 - x0) / 30))      # клетка ~5 мм, линейка ~8 мм

    # красные линии: (x вверху листа, x внизу листа) — линия может быть наклонной.
    # Учитываем только те, что внутри листа (край стола бывает рыжеватым).
    inset = (pitch or 0) * 0.8
    reds_small = [(xt, xb) for xt, xb in _red_lines(small) if x0 + inset < (xt + xb) / 2 < x1 - inset]

    # линейки разворота идут через корешок без разрыва — лист обрезаем по сгибу
    if curves and not is_grid:
        x0, x1 = _cut_at_fold(vpiece, curves, reds_small, x0, x1, pitch)
        reds_small = [r for r in reds_small if x0 + inset < (r[0] + r[1]) / 2 < x1 - inset]
    red = [(xt / scale, xb / scale) for xt, xb in reds_small]
    # точки, которые точно на листе: между первой и последней линейкой
    if curves:
        xm = (x0 + x1) / 2
        ya, yb = curve_y(curves[0], xm), curve_y(curves[-1], xm)
    else:
        ya, yb = 0.3 * h, 0.7 * h
    seeds = [((x0 + (x1 - x0) * fx) / scale, (ya + (yb - ya) * fy) / scale)
             for fx in (0.2, 0.4, 0.6, 0.8) for fy in (0.2, 0.5, 0.8)]
    base = {"red": red, "W": W, "H": H, "x_range": (x0 / scale, x1 / scale), "lines_v": vpitch is not None,
            "paper": _paper_mask(img, seeds)}
    if not curves:
        return dict(base, kind="plain", lines=None, curves=None, pitch=None)
    full = [(xs / scale, ys / scale) for xs, ys in curves]
    xm = (x0 + x1) / 2 / scale
    return dict(base, kind="grid" if is_grid else "lined", curves=full,
                lines=[curve_y(c, xm) for c in full], pitch=pitch / scale)


def auto_orient(img):
    """Лист снят боком (линейки вертикальные)? Поворачиваем на 90° так, чтобы строки шли
    горизонтально, а красное поле оказалось справа. Возвращает (изображение, угол)."""
    a = detect_lines(img)
    n_a = len(a["lines"] or [])
    if n_a >= 8:
        return img, 0
    cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    b = detect_lines(cw)
    if len(b["lines"] or []) < max(8, n_a + 3):
        return img, 0
    W = cw.shape[1]
    if b["red"] and all((xt + xb) / 2 < W / 2 for xt, xb in b["red"]):
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), -90
    return cw, 90


def _red_lines(small):
    """Ищет тонкие почти вертикальные красноватые линии (поле) с любой стороны листа."""
    h, w = small.shape[:2]
    b, g, r = [small[:, :, i].astype(np.float32) for i in range(3)]
    redness = r - g
    mask = (((redness - cv2.blur(redness, (31, 31))) > 7) & (redness > 8)).astype(np.uint8) * 255
    wide = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1)))
    mask = cv2.subtract(mask, wide)                     # только тонкие линии (не край стола и т.п.)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)))
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=int(h * 0.1),
                           minLineLength=int(h * 0.15), maxLineGap=int(h * 0.04))
    if segs is None:
        return []
    groups = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4).astype(float):
        if abs(y2 - y1) < 1 or abs(x2 - x1) > 0.12 * abs(y2 - y1):
            continue
        a = (x2 - x1) / (y2 - y1)
        xm = x1 + a * (h / 2 - y1)
        for gr in groups:
            if abs(gr["xm"] - xm) < 12:
                gr["pts"] += [(y1, x1), (y2, x2)]
                gr["len"] += abs(y2 - y1)
                break
        else:
            groups.append({"xm": xm, "pts": [(y1, x1), (y2, x2)], "len": abs(y2 - y1)})
    out = []
    for gr in groups:
        ys = np.array([p[0] for p in gr["pts"]])
        if gr["len"] < 0.3 * h or ys.max() - ys.min() < 0.3 * h:
            continue
        k, c = np.polyfit(ys, np.array([p[1] for p in gr["pts"]]), 1)
        out.append((float(c), float(c + k * h)))
    return sorted(out)


def plan_rows(info, cells_per_line=2, descender=0.32, baseline_lift=0.0,
              pitch_px=0, top_px=0, left_px=0, right_px=0):
    """Возвращает список строк [(y_базовой_линии, верх_полосы, низ_полосы)] и поля по x.

    Полоса строки = её законное пространство: символ никогда не выходит за неё,
    поэтому соседние строки не налезают друг на друга.
    """
    W, H = info["W"], info["H"]
    curves = None
    if pitch_px > 0 or info["lines"] is None:
        pitch = pitch_px or W * 0.038           # ~8 мм на листе A4
        first = top_px or H * 0.07
        n = int((H * 0.97 - first) / pitch) + 1
        baselines = [first + k * pitch for k in range(n)]
    else:
        step = cells_per_line if info["kind"] == "grid" else 1
        baselines = info["lines"][::step]
        curves = (info.get("curves") or [None] * len(info["lines"]))[::step]
        pitch = info["pitch"] * step
        if top_px:
            keep = [i for i, y in enumerate(baselines) if y >= top_px]
            baselines, curves = [baselines[i] for i in keep], [curves[i] for i in keep]
    lift = baseline_lift * pitch
    # красная линия — барьер: текст не пересекает её ни при каких настройках
    reds = info.get("red") or []

    def red_x(line, y):
        xt, xb = line
        return xt + (xb - xt) * y / H

    px0, px1 = info.get("x_range") or (0, W)
    pw = px1 - px0
    base_left = left_px or px0 + pw * 0.08
    base_right = W - right_px if right_px else px1 - pw * 0.05
    gap = pitch * 0.35
    rows = []
    for i, y in enumerate(baselines):
        cv = curves[i] if curves else None
        if cv is not None:
            cv = (np.asarray(cv[0], float), np.asarray(cv[1], float) - lift)
        # строка идёт вдоль своей линейки (она может быть наклонной и изогнутой)
        ends = curve_ys(cv, [base_left, base_right]) if cv is not None else [y - lift] * 2
        top, bottom = min(ends) - (1 - descender) * pitch, max(ends) + descender * pitch
        if top < H * 0.01 or bottom > H * 0.99:
            continue
        # для каждой строки берём самое «строгое» положение линии в пределах её высоты
        mid = (top + bottom) / 2
        lx = [max(red_x(ln, top), red_x(ln, bottom)) for ln in reds if red_x(ln, mid) < W / 2]
        rx = [min(red_x(ln, top), red_x(ln, bottom)) for ln in reds if red_x(ln, mid) >= W / 2]
        clip_l, clip_r = (max(lx) + 3 if lx else 0), (min(rx) - 3 if rx else W)
        left = max(base_left, max(lx) + gap) if lx and not left_px else base_left
        right = min(base_right, min(rx) - gap) if rx else base_right
        left, right = max(left, clip_l), min(right, clip_r)
        if right - left > pitch * 3:
            base = curve_y(cv, left) if cv is not None else y - lift
            row = {"base": base, "top": base - (1 - descender - 0.03) * pitch,
                   "bottom": base + (descender - 0.03) * pitch,
                   "left": left, "right": right, "clip_l": clip_l, "clip_r": clip_r, "curve": cv}
            # строка целиком должна лежать на бумаге (не на столе и не за краем листа)
            xs = np.linspace(left, right, 24)
            off = np.array([row_offset(row, x) for x in xs])
            ys = np.concatenate([row["top"] + off, row["bottom"] + off])
            if _on_paper(info.get("paper"), np.concatenate([xs, xs]), ys) >= 0.95:
                rows.append(row)
    return rows, pitch


def row_offset(row, x):
    """На сколько px строка в точке x ниже (+) или выше (−), чем в начале строки."""
    cv = row.get("curve")
    return curve_y(cv, x) - row["base"] if cv is not None else 0.0


def preview_rows(img, rows):
    """Картинка для проверки: зелёный — базовые линии, жёлтый — границы строк, фиолетовый — барьер."""
    out = img.copy()
    W = out.shape[1]
    for r in rows:
        xs = np.linspace(r["left"], r["right"], 40)
        off = np.array([row_offset(r, x) for x in xs])
        for y0, col, th in ((r["top"], (0, 200, 255), 2), (r["bottom"], (0, 200, 255), 2), (r["base"], (0, 160, 0), 2)):
            pts = np.stack([xs, y0 + off], axis=1).round().astype(np.int32)
            cv2.polylines(out, [pts], False, col, th)
        for x, y in ((r["left"], 0), (r["right"], off[-1])):
            cv2.line(out, (int(x), int(r["top"] + y)), (int(x), int(r["bottom"] + y)), (0, 200, 255), 2)
        for x in (r["clip_l"], r["clip_r"]):
            if 0 < x < W:
                o = row_offset(r, x)
                cv2.line(out, (int(x), int(r["top"] + o)), (int(x), int(r["bottom"] + o)), (200, 0, 200), 3)
    return out
