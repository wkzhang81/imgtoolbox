"""
tkinter GUI: pick a tool on the left, a parameter form is generated dynamically on the
right, with a log and a Run button at the bottom.
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..registry import REGISTRY, list_tools


class StdoutRedirector:
    """Push print output to the main thread's queue, which the GUI polls and refreshes."""

    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, s: str):
        if s:
            self.q.put(s)

    def flush(self):
        pass


class ToolGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("imgtoolbox — Scientific Image Processing Toolkit")
        root.geometry("820x720")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.param_vars: dict[str, tk.Variable] = {}

        self._build_ui()
        self._poll_log()

        # Trigger loading of the tool list
        try:
            tools = list_tools()
        except Exception as exc:
            messagebox.showerror("Failed to load tools", str(exc))
            tools = []

        names = [t.label for t in tools]
        self.tool_keys = [t.key for t in tools]
        self.tool_combo["values"] = names
        if names:
            self.tool_combo.current(0)
            self._on_tool_change()

    # ---------- UI construction ----------

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Tool:").pack(side=tk.LEFT)
        self.tool_combo = ttk.Combobox(top, state="readonly", width=50)
        self.tool_combo.pack(side=tk.LEFT, padx=6)
        self.tool_combo.bind("<<ComboboxSelected>>", lambda e: self._on_tool_change())

        # Description
        desc_frame = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        desc_frame.pack(fill=tk.X)
        self.desc_label = ttk.Label(desc_frame, text="", wraplength=780, foreground="#555")
        self.desc_label.pack(anchor=tk.W)

        # Parameter form (scrollable)
        form_outer = ttk.LabelFrame(self.root, text="Parameters", padding=8)
        form_outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        canvas = tk.Canvas(form_outer, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(form_outer, orient="vertical", command=canvas.yview)
        self.form_inner = ttk.Frame(canvas)

        self.form_inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.form_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel: scroll the form only when the pointer is over the form area
        def _on_wheel(e):
            w = canvas.winfo_containing(e.x_root, e.y_root)
            while w is not None and w is not canvas:
                w = getattr(w, "master", None)
            if w is canvas:
                canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_wheel)

        # Run row
        run_frame = ttk.Frame(self.root, padding=8)
        run_frame.pack(fill=tk.X)
        self.run_btn = ttk.Button(run_frame, text="Run", command=self._on_run)
        self.run_btn.pack(side=tk.LEFT)
        self.status_label = ttk.Label(run_frame, text="Ready", foreground="#0a0")
        self.status_label.pack(side=tk.LEFT, padx=12)

        # Log
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.log_text = tk.Text(log_frame, height=12, wrap="word", state=tk.DISABLED)
        log_sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_sb.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------- Tool switching / parameter form ----------

    def _on_tool_change(self):
        idx = self.tool_combo.current()
        if idx < 0:
            return
        key = self.tool_keys[idx]
        tool = REGISTRY[key]

        self.desc_label.configure(text=tool.description)

        # Clear the old form
        for w in self.form_inner.winfo_children():
            w.destroy()
        self.param_vars.clear()

        for row, p in enumerate(tool.params):
            ttk.Label(self.form_inner, text=f"{p.label}:").grid(
                row=row * 2, column=0, sticky="w", padx=(0, 8), pady=(4, 0)
            )
            self._build_param_widget(row * 2, p)
            if p.help:
                ttk.Label(
                    self.form_inner, text=p.help, foreground="#888"
                ).grid(row=row * 2 + 1, column=1, columnspan=2, sticky="w", padx=2, pady=(0, 4))

        self.form_inner.columnconfigure(1, weight=1)

    def _build_param_widget(self, row: int, p):
        key = p.key
        t = p.type

        if t == "bool":
            var = tk.BooleanVar(value=bool(p.default))
            self.param_vars[key] = var
            ttk.Checkbutton(
                self.form_inner, variable=var
            ).grid(row=row, column=1, sticky="w", pady=4)
            return

        if t == "choice":
            var = tk.StringVar(value=str(p.default))
            self.param_vars[key] = var
            ttk.Combobox(
                self.form_inner,
                textvariable=var,
                values=p.choices,
                state="readonly",
            ).grid(row=row, column=1, sticky="ew", pady=4)
            return

        # Text-like types (file/dir/files/str/int/float) use an Entry
        var = tk.StringVar(value=str(p.default) if p.default is not None else "")
        self.param_vars[key] = var
        entry = ttk.Entry(self.form_inner, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=4)

        if t in ("file", "files", "dir"):
            suffix = " (optional)" if p.optional else ""
            btn = ttk.Button(
                self.form_inner,
                text=f"Browse…{suffix}",
                command=lambda v=var, pt=p: self._browse(v, pt),
            )
            btn.grid(row=row, column=2, padx=(6, 0), pady=4)

    # The help text is placed on the next row by _on_tool_change, so we don't add it again here

    def _browse(self, var: tk.StringVar, p):
        if p.type == "dir":
            d = filedialog.askdirectory()
            if d:
                var.set(d)
        elif p.type == "files":
            files = filedialog.askopenfilenames(
                filetypes=p.filetypes or [("Images", "*.jpg *.png *.tif")]
            )
            if files:
                var.set(";".join(files))
        elif p.type == "file":
            if p.file_mode == "save":
                f = filedialog.asksaveasfilename(
                    filetypes=p.filetypes or [("All", "*.*")],
                    defaultextension=".png",
                )
            else:
                f = filedialog.askopenfilename(
                    filetypes=p.filetypes or [("All", "*.*")]
                )
            if f:
                var.set(f)

    # ---------- Run ----------

    @staticmethod
    def _check_range(p, v):
        if p.min is not None and v < p.min:
            raise ValueError(f"Cannot be less than {p.min}")
        if p.max is not None and v > p.max:
            raise ValueError(f"Cannot be greater than {p.max}")

    def _collect_params(self) -> tuple[str, dict]:
        idx = self.tool_combo.current()
        if idx < 0:
            raise ValueError("Please select a tool first.")
        key = self.tool_keys[idx]
        tool = REGISTRY[key]

        out = {}
        for p in tool.params:
            raw = self.param_vars[p.key].get()
            t = p.type
            try:
                if t == "bool":
                    out[p.key] = bool(raw)
                elif t == "int":
                    if raw in ("", None):
                        out[p.key] = p.default
                    else:
                        v = int(raw)
                        self._check_range(p, v)
                        out[p.key] = v
                elif t == "float":
                    if raw in ("", None):
                        out[p.key] = p.default
                    else:
                        v = float(raw)
                        self._check_range(p, v)
                        out[p.key] = v
                elif t == "choice":
                    out[p.key] = raw
                else:
                    v = (raw or "").strip()
                    if not p.optional and v == "":
                        raise ValueError("Cannot be empty")
                    if v:
                        if t == "file" and p.file_mode == "open" and not Path(v).exists():
                            raise ValueError(f"File does not exist: {v}")
                        if t == "dir" and not Path(v).is_dir():
                            raise ValueError(f"Directory does not exist: {v}")
                        if t == "files":
                            missing = [s for s in v.split(";")
                                       if s.strip() and not Path(s.strip()).exists()]
                            if missing:
                                raise ValueError(f"File does not exist: {missing[0]}")
                    out[p.key] = v
            except ValueError as exc:
                raise ValueError(f"Invalid parameter '{p.label}': {exc}") from exc
        return key, out

    def _on_run(self):
        try:
            key, params = self._collect_params()
        except Exception as exc:
            messagebox.showerror("Parameter error", str(exc))
            return

        tool = REGISTRY[key]
        self.run_btn.configure(state=tk.DISABLED)
        self.status_label.configure(text=f"Running: {tool.label} …", foreground="#c60")
        self._log(f"\n===== Run: {tool.label} =====")
        self._log(f"Parameters: {params}\n")

        # Background thread
        def worker():
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            redirector = StdoutRedirector(self.log_queue)
            sys.stdout = redirector
            sys.stderr = redirector
            try:
                summary = tool.run(params)
                self.log_queue.put(f"\n[Done] {summary}\n")
                self.log_queue.put("__DONE__")
            except Exception as exc:
                import traceback
                self.log_queue.put(f"\n[Failed] {exc}\n")
                self.log_queue.put(traceback.format_exc())
                self.log_queue.put("__FAILED__")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Log ----------

    def _log(self, s: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, s)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _poll_log(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item == "__DONE__":
                    self.run_btn.configure(state=tk.NORMAL)
                    self.status_label.configure(text="Done", foreground="#0a0")
                elif item == "__FAILED__":
                    self.run_btn.configure(state=tk.NORMAL)
                    self.status_label.configure(text="Run failed", foreground="#c00")
                else:
                    self._log(item)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_log)


def launch():
    root = tk.Tk()
    ToolGUI(root)
    root.mainloop()
