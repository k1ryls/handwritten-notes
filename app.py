"""Окно программы «Рукописный конспект»."""
import os
import queue
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

from handwriter.extract import extract_font
from handwriter.render import list_sheets, read_text, render_text
from handwriter.template import CHARSETS, DEFAULT_SETS, append_template, generate_template

HERE = os.path.dirname(os.path.abspath(__file__))
SHEETS_DIR = os.path.join(HERE, "листы")     # сохранённые фото чистых страниц, берутся по очереди
SET_NAMES = {"ru_lower": "русские строчные", "ru_upper": "русские заглавные", "digits": "цифры",
             "punct": "знаки препинания", "en_lower": "латинские строчные", "en_upper": "латинские заглавные",
             "ua": "украинские буквы (і ї є ґ)"}


def open_path(p):
    try:
        os.startfile(p)
    except Exception:
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Рукописный конспект")
        self.geometry("820x720")
        self.q = queue.Queue()
        self.font_dir = tk.StringVar(value=os.path.join(HERE, "my_font"))

        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Папка почерка:").pack(side="left")
        ttk.Entry(top, textvariable=self.font_dir).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(top, text="…", width=3, command=self.pick_font_dir).pack(side="left")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8)
        nb.add(self.tab_template(nb), text="1. Шаблон")
        nb.add(self.tab_extract(nb), text="2. Мой почерк")
        nb.add(self.tab_render(nb), text="3. Конспект")

        self.log_box = tk.Text(self, height=8, state="disabled", bg="#f4f4f4")
        self.log_box.pack(fill="x", padx=8, pady=8)
        self.after(100, self.pump)

    # ---------- общее ----------
    def pick_font_dir(self):
        d = filedialog.askdirectory(initialdir=self.font_dir.get())
        if d:
            self.font_dir.set(d)

    def log(self, msg):
        self.q.put(msg)

    def pump(self):
        while not self.q.empty():
            item = self.q.get()
            if callable(item):
                item()
                continue
            self.log_box.configure(state="normal")
            self.log_box.insert("end", item + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(100, self.pump)

    def run_bg(self, fn, done=None):
        def work():
            try:
                res = fn()
                if done:
                    self.q.put(lambda: done(res))
            except Exception as e:
                self.log(f"Ошибка: {e}")
        threading.Thread(target=work, daemon=True).start()

    # ---------- 1. шаблон ----------
    def tab_template(self, nb):
        f = ttk.Frame(nb, padding=12)
        ttk.Label(f, text="Распечатайте шаблон, впишите каждый символ своей ручкой и сфотографируйте листы.\n"
                          "Чем больше вариантов каждой буквы, тем живее получится текст.",
                  justify="left").pack(anchor="w")
        self.sets = {}
        box = ttk.LabelFrame(f, text="Какие символы", padding=8)
        box.pack(fill="x", pady=8)
        for k in CHARSETS:
            v = tk.BooleanVar(value=k in DEFAULT_SETS)
            self.sets[k] = v
            ttk.Checkbutton(box, text=SET_NAMES[k], variable=v).pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Вариантов каждого символа:").pack(side="left")
        self.variants = tk.IntVar(value=2)
        ttk.Spinbox(row, from_=1, to=4, textvariable=self.variants, width=4).pack(side="left", padx=5)
        row2 = ttk.Frame(f)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="Свои доп. символы (например  ∑√≈°±):").pack(side="left")
        self.extra = tk.StringVar()
        ttk.Entry(row2, textvariable=self.extra, width=25).pack(side="left", padx=5)
        ttk.Button(f, text="Создать шаблон для печати", command=self.make_template).pack(anchor="w", pady=(10, 4))
        ttk.Button(f, text="Дописать недостающие символы (доп. лист к уже заполненному шаблону)",
                   command=self.add_template).pack(anchor="w")
        ttk.Label(f, text="«Дописать» добавляет только новые символы (например, украинские буквы):\n"
                          "старые листы и их фото остаются годными, заново ничего заполнять не нужно.",
                  foreground="#666", justify="left").pack(anchor="w", pady=6)
        return f

    def _chosen_sets(self):
        sets = [k for k, v in self.sets.items() if v.get()]
        if not sets and not self.extra.get().strip():
            messagebox.showwarning("Шаблон", "Выберите хотя бы один набор символов.")
            return None
        return sets

    def make_template(self):
        sets = self._chosen_sets()
        if sets is None:
            return
        fd = self.font_dir.get()
        if os.path.exists(os.path.join(fd, "font.json")) and not messagebox.askyesno(
                "Шаблон", "В этой папке уже есть почерк. Будет напечатан полный новый комплект листов "
                          "(с новыми номерами). Уже заполненные листы и буквы никуда не денутся.\n\n"
                          "Если нужно только добавить несколько символов, нажмите «Нет» и выберите "
                          "«Дописать недостающие символы».\n\nСоздать полный комплект заново?"):
            return

        def job():
            pdf, n = generate_template(fd, sets, self.variants.get(), self.extra.get())
            self.log(f"Шаблон: {n} лист(ов) → {pdf}")
            return pdf
        self.run_bg(job, open_path)

    def add_template(self):
        sets = self._chosen_sets()
        if sets is None:
            return
        fd = self.font_dir.get()
        if not os.path.exists(os.path.join(fd, "layout.json")):
            messagebox.showinfo("Шаблон", "Шаблона ещё нет — сначала нажмите «Создать шаблон для печати».")
            return

        def job():
            pdf, n = append_template(fd, sets, self.variants.get(), self.extra.get())
            self.log(f"Доп. шаблон: {n} лист(ов) → {pdf}" if pdf else "Все выбранные символы уже есть в шаблоне.")
            return pdf
        self.run_bg(job, lambda p: p and open_path(p))

    # ---------- 2. почерк ----------
    def tab_extract(self, nb):
        f = ttk.Frame(nb, padding=12)
        ttk.Label(f, text="Выберите фото или сканы заполненных листов (можно все сразу, в любом порядке).\n"
                          "Советы: лист целиком в кадре, все 4 чёрных квадрата видны, ровный свет без теней,\n"
                          "лист лежит ровно, снимать сверху. Скан — идеально. Переснятый лист можно загрузить ещё раз.\n"
                          "Важно: в Telegram отправляйте «как файл» — обычное фото там сильно сжимается,\n"
                          "и буквы получаются размытыми.",
                  justify="left").pack(anchor="w")
        ttk.Button(f, text="Загрузить фото листов…", command=self.extract).pack(anchor="w", pady=10)
        ttk.Button(f, text="Открыть проверку (зелёные рамки — символ найден)",
                   command=lambda: open_path(os.path.join(self.font_dir.get(), "debug"))).pack(anchor="w")
        ttk.Button(f, text="Открыть папку с буквами",
                   command=lambda: open_path(os.path.join(self.font_dir.get(), "glyphs"))).pack(anchor="w", pady=4)
        ttk.Label(f, text="Плохую букву можно удалить из папки glyphs — тогда она возьмётся из другого варианта.",
                  foreground="#666").pack(anchor="w", pady=6)
        return f

    def extract(self):
        fd = self.font_dir.get()
        if not os.path.exists(os.path.join(fd, "layout.json")):
            messagebox.showwarning("Почерк", "Сначала создайте шаблон на вкладке 1 (в этой же папке почерка).")
            return
        files = filedialog.askopenfilenames(filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        if files:
            self.log("Обработка фото…")
            self.run_bg(lambda: extract_font(list(files), fd, log=self.log))

    # ---------- 3. конспект ----------
    def tab_render(self, nb):
        f = ttk.Frame(nb, padding=12)
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        ttk.Label(bar, text="Текст конспекта (пустая строка = отступ между абзацами):").pack(side="left")
        ttk.Button(bar, text="Открыть .txt…", command=self.load_txt).pack(side="right")
        self.text = tk.Text(f, height=10, wrap="word")
        self.text.pack(fill="both", expand=True, pady=4)

        # листы: папка с фото чистых страниц (по очереди) или одно фото
        self.paper = tk.StringVar(value=SHEETS_DIR)
        r = ttk.Frame(f)
        r.pack(fill="x", pady=2)
        ttk.Label(r, text="Чистые листы:", width=20).pack(side="left")
        ttk.Entry(r, textvariable=self.paper).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r, text="Папка…", command=self.pick_paper_dir).pack(side="left")
        ttk.Button(r, text="Одно фото…", command=self.pick_paper).pack(side="left", padx=(4, 0))
        r = ttk.Frame(f)
        r.pack(fill="x", pady=2)
        self.sheets_info = ttk.Label(r, foreground="#555")
        self.sheets_info.pack(side="left", padx=(0, 8))
        ttk.Button(r, text="Открыть папку листов", command=self.open_sheets).pack(side="right")
        self.start_sheet = tk.IntVar(value=1)
        ttk.Spinbox(r, from_=1, to=99, textvariable=self.start_sheet, width=4).pack(side="right", padx=(4, 10))
        ttk.Label(r, text="Начать с листа №").pack(side="right")
        self.paper.trace_add("write", lambda *_: self.update_sheets_info())
        self.update_sheets_info()

        self.out_dir = tk.StringVar(value=os.path.join(HERE, "result"))
        r = ttk.Frame(f)
        r.pack(fill="x", pady=2)
        ttk.Label(r, text="Куда сохранить:", width=20).pack(side="left")
        ttk.Entry(r, textvariable=self.out_dir).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(r, text="…", width=3, command=self.pick_out).pack(side="left")

        g = ttk.LabelFrame(f, text="Настройки", padding=6)
        g.pack(fill="x", pady=6)
        self.opt = {}
        spec = [("size", "Размер букв (доля строки)", 0.36, 0.2, 0.6, 0.02),
                ("jitter", "Небрежность", 1.0, 0.0, 2.5, 0.1),
                ("letter_spacing", "Между буквами", -0.03, -0.2, 0.5, 0.01),
                ("word_spacing", "Между словами", 1.1, 0.3, 2.5, 0.05),
                ("indent", "Красная строка", 2.0, 0.0, 6.0, 0.5),
                ("descender", "Хвосты ниже линии (доля)", 0.32, 0.0, 0.45, 0.02),
                ("cells_per_line", "Клеток на строку (лист в клетку)", 2, 1, 3, 1),
                ("pitch_px", "Шаг строк вручную, px (0 = авто)", 0, 0, 400, 1)]
        for i, (key, label, val, lo, hi, step) in enumerate(spec):
            var = tk.DoubleVar(value=val)
            self.opt[key] = var
            ttk.Label(g, text=label).grid(row=i // 2, column=(i % 2) * 2, sticky="w", padx=4, pady=2)
            ttk.Spinbox(g, from_=lo, to=hi, increment=step, textvariable=var, width=7).grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=4)
        self.ink = None
        r = ttk.Frame(f)
        r.pack(fill="x")
        self.ink_btn = ttk.Button(r, text="Цвет чернил: как на скане", command=self.pick_ink)
        self.ink_btn.pack(side="left")
        ttk.Button(r, text="Собрать конспект", command=self.render).pack(side="right")
        ttk.Button(r, text="Открыть результат (все страницы)",
                   command=lambda: open_path(self.out_dir.get())).pack(side="right", padx=6)
        return f

    def load_txt(self):
        p = filedialog.askopenfilename(filetypes=[("Текст", "*.txt"), ("Все файлы", "*.*")])
        if p:
            try:
                self.text.delete("1.0", "end")
                self.text.insert("1.0", read_text(p))
            except Exception as e:
                messagebox.showerror("Текст", str(e))

    def pick_paper(self):
        p = filedialog.askopenfilename(filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        if p:
            self.paper.set(p)

    def pick_paper_dir(self):
        d = filedialog.askdirectory(initialdir=self.paper.get() if os.path.isdir(self.paper.get()) else SHEETS_DIR)
        if d:
            self.paper.set(d)

    def open_sheets(self):
        p = self.paper.get()
        open_path(p if os.path.isdir(p) else os.path.dirname(p) or SHEETS_DIR)

    def update_sheets_info(self):
        p = self.paper.get()
        files = list_sheets(p) if os.path.exists(p) else []
        if not files:
            txt = "Нет фото листов — положите фото чистых страниц в папку или выберите фото."
        elif len(files) == 1:
            txt = f"Все страницы на листе «{os.path.basename(files[0])}»."
        else:
            names = " → ".join(os.path.basename(x) for x in files)
            txt = f"Страницы по очереди: {names} → снова первый"
        self.sheets_info.configure(text=txt)

    def pick_out(self):
        d = filedialog.askdirectory(initialdir=self.out_dir.get())
        if d:
            self.out_dir.set(d)

    def pick_ink(self):
        c = colorchooser.askcolor(title="Цвет чернил")
        if c and c[0]:
            self.ink = [int(x) for x in c[0]]
            self.ink_btn.configure(text=f"Цвет чернил: {c[1]}")

    def render(self):
        text = self.text.get("1.0", "end").strip()
        fd, paper, out = self.font_dir.get(), self.paper.get(), self.out_dir.get()
        if not text:
            messagebox.showwarning("Конспект", "Вставьте текст конспекта.")
            return
        if not os.path.exists(os.path.join(fd, "font.json")):
            messagebox.showwarning("Конспект", "Почерк ещё не загружен (вкладка 2).")
            return
        if not os.path.exists(paper) or not list_sheets(paper):
            messagebox.showwarning("Конспект", "Нет фото чистого листа: положите фото страниц в папку листов "
                                               "или выберите одно фото.")
            return
        try:
            o = {k: float(v.get()) for k, v in self.opt.items()}
            o["start_sheet"] = max(1, int(self.start_sheet.get()))
        except (tk.TclError, ValueError):
            messagebox.showwarning("Конспект", "В настройках должны быть числа (дробная часть через точку).")
            return
        o["cells_per_line"] = int(o["cells_per_line"])
        o["ink_rgb"] = self.ink
        self.log("Собираю конспект…")

        def done(paths):
            # одна страница — сразу её; несколько — папку, где видны все страницы и общий PDF
            if paths:
                open_path(paths[0] if len(paths) == 1 else os.path.dirname(paths[0]))
        self.run_bg(lambda: render_text(text, fd, paper, out, log=self.log, **o), done)


if __name__ == "__main__":
    App().mainloop()
