"""Tkinter GUI for the MLB prediction engine.

Dropdown-driven input: teams, date, stadium, starters, two ordered 9-man
lineups, day/night, and weather. Outputs per run: every batter's calibrated
HR probability (with fair odds), hit probability, both starters' projected
strikeouts with over-probabilities, and game totals.

Run:
    python Model/gui.py
"""

import datetime as dt
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
LOGO_PATH = Path(__file__).resolve().parents[1] / "MLB-Logo.png"

# MLB brand colors
NAVY = "#041E42"
RED = "#D50032"
WHITE = "#FFFFFF"
STRIPE = "#EAF0F8"      # light row stripe
TOPPICK = "#FBE3E9"     # light red tint for top HR picks
DISABLED = "#5A6B84"


def load_logo(height=64):
    """MLB logo as a tk PhotoImage (Pillow for smooth scaling if present)."""
    if not LOGO_PATH.exists():
        return None
    try:
        from PIL import Image, ImageTk
        img = Image.open(LOGO_PATH)
        w = int(img.width * height / img.height)
        return ImageTk.PhotoImage(img.resize((w, height), Image.LANCZOS))
    except Exception:
        try:
            img = tk.PhotoImage(file=str(LOGO_PATH))
            return img.subsample(max(1, img.height() // height))
        except Exception:
            return None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MLB Prediction Engine")
        self.geometry("1080x820")
        self._apply_style()
        self.pred = None
        self.pools = {}      # abbrev -> dict(batters={label: pid}, pitchers={...})
        self.abbrev_full = {}
        # worker threads only write these; the main thread polls them
        # (tkinter's after() must never be called from a worker thread)
        self._load_state = None
        self._load_msg = "starting..."
        self._pred_state = None
        self._build_layout()
        self.status.set("Loading data and models...")
        threading.Thread(target=self._load, daemon=True).start()
        self.after(200, self._poll_load)

    # ------------------------------------------------------------ setup

    def _load(self):
        try:
            from predict import Predictor

            def tick(msg):
                self._load_msg = msg
            self.pred = Predictor(progress=tick)
            self._load_msg = "building player pools..."
            self._build_pools()
            self._load_state = ("ok", None)
        except Exception as e:
            self._load_state = ("err", str(e))

    def _poll_load(self):
        if self._load_state is None:
            self.status.set(f"Loading: {self._load_msg}")
            self.after(200, self._poll_load)
            return
        state, err = self._load_state
        if state == "ok":
            self._on_ready()
        else:
            self.status.set(f"LOAD FAILED: {err}")
            messagebox.showerror("Load failed", err)

    def _build_pools(self):
        season = int(self.pred.stores.raw["games"]["Season"].max())
        bs = pd.read_csv(DATA_DIR / "mlb_batting_stats.csv",
                         encoding="utf-8-sig", usecols=["Year", "Team", "TeamName"])
        pairs = bs[bs["Year"] == bs["Year"].max()].drop_duplicates()
        self.abbrev_full = dict(zip(pairs["Team"], pairs["TeamName"]))
        full_abbrev = {v: k for k, v in self.abbrev_full.items()}

        ros = self.pred.stores.raw["rosters"]
        for _, r in ros.iterrows():
            ab = full_abbrev.get(r["Team"])
            if ab is None:
                continue
            pool = self.pools.setdefault(ab, {"batters": {}, "pitchers": {}})
            label = f'{r["Name"]} ({r["Position"]})'
            if r["Position"] in ("Rotation", "Bullpen"):
                pool["pitchers"][label] = r["PlayerId"]
            else:
                pool["batters"][label] = r["PlayerId"]

        # Depth-chart rosters lag trades and call-ups (e.g. a player dealt
        # mid-season). Anyone who actually appeared in current-season game
        # logs is added to the pool of the team they last played for.
        gb = self.pred.stores.raw["gb"]
        cur = gb[gb["Season"] == season].sort_values("Date")
        for pid, r in cur.groupby("PlayerId").last().iterrows():
            pool = self.pools.setdefault(r["Team"], {"batters": {}, "pitchers": {}})
            if pid in pool["batters"].values() or pid in pool["pitchers"].values():
                continue
            label = f'{r["Name"]} ({r["Position"] or "?"})'
            if label in pool["batters"]:
                label = f'{r["Name"]} [{pid}]'
            pool["batters"][label] = pid
        gp = self.pred.stores.raw["gp"]
        curp = gp[gp["Season"] == season].sort_values("Date")
        agg = curp.groupby("PlayerId").agg(
            n=("GamePk", "size"), gs=("GS", "sum"))
        for pid, r in curp.groupby("PlayerId").last().iterrows():
            # skip position players with a mop-up appearance or two
            if agg.loc[pid, "n"] < 3 and agg.loc[pid, "gs"] == 0:
                continue
            pool = self.pools.setdefault(r["Team"], {"batters": {}, "pitchers": {}})
            if pid in pool["pitchers"].values():
                continue
            label = f'{r["Name"]} (P)'
            if label in pool["pitchers"]:
                label = f'{r["Name"]} [{pid}]'
            pool["pitchers"][label] = pid

        games = self.pred.stores.raw["games"]
        parks = self.pred.stores.raw["parks"]
        self.venues = sorted(set(parks["Ballpark"]) |
                             set(games.loc[games["Season"] == season, "Venue"]))
        self.wind_dirs = sorted(games["WindDir"].dropna().unique())
        self.conditions = sorted(games["Condition"].dropna().unique())
        # HP-umpire name -> HpUmpId, for the form's editable umpire field.
        umps = self.pred.stores.raw.get("umps")
        self.ump_name_to_id = {}
        if umps is not None:
            u = umps.dropna(subset=["HpUmp", "HpUmpId"])
            self.ump_name_to_id = {n: int(i) for n, i in
                                   zip(u["HpUmp"], u["HpUmpId"])}
        self.ump_names = sorted(self.ump_name_to_id)
        # home team -> default venue
        self.team_park = {full_abbrev.get(t): b for b, t in
                          zip(parks["Ballpark"], parks["Team"]) if full_abbrev.get(t)}

    def _on_ready(self):
        self._refresh_team_options()
        self.cb_venue["values"] = self.venues
        self.cb_wdir["values"] = self.wind_dirs
        self.cb_cond["values"] = self.conditions
        self.cb_ump["values"] = self.ump_names
        self.btn_predict["state"] = "normal"
        self.status.set("Ready. Pick teams, fill lineups (or auto-fill), Predict.")
        self._load_todays_file(silent=True)
        self._health_check()

    def _health_check(self):
        """Warn when predictions would be built on bad inputs: the morning
        data job failed (Scripts/update_all.py writes its outcome to
        Logs/last_run_status.json) or the game logs have gone stale
        mid-season. Without this, the only failure signal is a log line."""
        import json
        problems = []
        status_file = DATA_DIR.parent / "Logs" / "last_run_status.json"
        try:
            status = json.loads(status_file.read_text())
            if not status.get("ok"):
                jobs = ", ".join(status.get("failed_jobs", [])) or "unknown"
                problems.append(
                    f"The last data update FAILED (finished "
                    f"{status.get('finished', '?')}; failed: {jobs}).\n"
                    f"Data was restored from backups and the retrain was "
                    f"skipped — see the newest Logs/update_*.log.")
        except (OSError, ValueError):
            pass                    # no status yet: job hasn't run since setup
        try:
            games = self.pred.stores.raw["games"]
            newest = pd.to_datetime(games["Date"]).max()
            age = (pd.Timestamp.today().normalize() - newest.normalize()).days
            if 5 <= dt.date.today().month <= 9 and age > 6:
                problems.append(
                    f"Newest game in the data is {newest.date()} "
                    f"({age} days ago) — mid-season that means the daily "
                    f"update is not ingesting new games. Predictions will "
                    f"use stale form/rosters.")
        except Exception:           # noqa: BLE001 — health check never blocks
            pass
        if problems:
            messagebox.showwarning("Data health", "\n\n".join(problems))

    def _load_todays_file(self, silent=False):
        """Populate the slate from Data/todays_games.json (written by
        Scripts/get_todays_games.py) if it exists."""
        import json
        path = DATA_DIR / "todays_games.json"
        if not path.exists():
            if not silent:
                messagebox.showinfo(
                    "No file", "Data/todays_games.json not found.\nRun: "
                    "python Scripts/get_todays_games.py")
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            specs = payload.get("games", [])
        except Exception as e:
            if not silent:
                messagebox.showerror("Load failed", str(e))
            return
        if not specs:
            return
        self._clear_slate()
        for spec in specs:
            spec["away_lineup"] = [tuple(x) for x in spec.get("away_lineup", [])]
            spec["home_lineup"] = [tuple(x) for x in spec.get("home_lineup", [])]
            self.slate.append(spec)
            self.lb_slate.insert("end", self._slate_row_text(spec))
        scraped = str(payload.get("scraped_at", ""))[:16].replace("T", " ")
        self.status.set(f"Auto-loaded {len(specs)} games from "
                        f"todays_games.json (scraped {scraped}). "
                        f"Click a game to load/edit it; Predict runs the "
                        f"whole slate.")

    # ----------------------------------------------------------- layout

    def _apply_style(self):
        self.configure(bg=NAVY)
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=NAVY, foreground=WHITE,
                    font=("Segoe UI", 10))
        s.configure("Title.TLabel", font=("Segoe UI", 17, "bold"),
                    foreground=WHITE)
        s.configure("Sub.TLabel", foreground="#9FB3D1")
        s.configure("TLabelframe", background=NAVY, bordercolor=RED,
                    relief="solid")
        s.configure("TLabelframe.Label", background=NAVY, foreground=WHITE,
                    font=("Segoe UI", 10, "bold"))
        s.configure("TButton", background=RED, foreground=WHITE,
                    font=("Segoe UI", 10, "bold"), padding=(10, 5),
                    bordercolor=RED, focuscolor=RED)
        s.map("TButton",
              background=[("disabled", DISABLED), ("active", "#F0234F")],
              foreground=[("disabled", "#C4CDD9")])
        for w in ("TCombobox", "TSpinbox", "TEntry"):
            s.configure(w, fieldbackground=WHITE, foreground=NAVY,
                        bordercolor="#7A8CA8", arrowcolor=NAVY,
                        insertcolor=NAVY)
        s.map("TCombobox",
              fieldbackground=[("readonly", WHITE)],
              foreground=[("readonly", NAVY)])
        s.configure("Treeview", background=WHITE, fieldbackground=WHITE,
                    foreground=NAVY, rowheight=24, font=("Segoe UI", 10))
        s.configure("Treeview.Heading", background=RED, foreground=WHITE,
                    font=("Segoe UI", 10, "bold"), relief="flat")
        s.map("Treeview.Heading", background=[("active", "#F0234F")])
        self.option_add("*TCombobox*Listbox.background", WHITE)
        self.option_add("*TCombobox*Listbox.foreground", NAVY)
        self.option_add("*TCombobox*Listbox.selectBackground", RED)
        self.option_add("*TCombobox*Listbox.selectForeground", WHITE)

    def _make_vscroll(self, parent):
        """Return an inner frame inside a vertically scrollable canvas.
        The mouse wheel scrolls it only while the pointer is over it, so it
        never fights the scrollbars in the results windows."""
        container = tk.Frame(parent, bg=NAVY)
        canvas = tk.Canvas(container, bg=NAVY, highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=NAVY)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win_id, width=e.width))

        def _wheel(e):
            canvas.yview_scroll(int(-e.delta / 120), "units")
        container.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        container.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        container.pack(side="top", fill="both", expand=True)
        return inner

    @staticmethod
    def _tree_with_scroll(parent, **kw):
        """A Treeview paired with a vertical scrollbar in its own frame."""
        frame = tk.Frame(parent, bg=NAVY)
        tv = ttk.Treeview(frame, **kw)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        tv.pack(side="left", fill="both", expand=True)
        return frame, tv

    def _build_layout(self):
        header = tk.Frame(self, bg=NAVY)
        header.pack(fill="x", padx=8, pady=(10, 2))
        self._logo = load_logo(height=64)
        if self._logo is not None:
            tk.Label(header, image=self._logo, bg=NAVY).pack(side="left",
                                                             padx=(4, 14))
        titles = tk.Frame(header, bg=NAVY)
        titles.pack(side="left")
        ttk.Label(titles, text="MLB Prediction Engine",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(titles, text="Home runs · hits · strikeouts · totals — "
                  "calibrated probabilities with fair odds",
                  style="Sub.TLabel").pack(anchor="w")

        # fixed bottom bar first, so it stays pinned below the scrollable body
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=8, pady=6)
        self.btn_predict = ttk.Button(bottom, text="Predict", state="disabled",
                                      command=self._predict_clicked)
        self.btn_predict.pack(side="left")
        self.status = tk.StringVar()
        ttk.Label(bottom, textvariable=self.status, wraplength=820,
                  justify="left").pack(side="left", padx=12)

        body = self._make_vscroll(self)

        top = ttk.LabelFrame(body, text="Game")
        top.pack(fill="x", padx=8, pady=6)

        def add(col, text, widget, width=14):
            ttk.Label(top, text=text).grid(row=0, column=col, sticky="w", padx=4)
            widget.grid(row=1, column=col, sticky="w", padx=4, pady=2)
            return widget

        self.cb_away = add(0, "Away team", ttk.Combobox(top, width=6, state="readonly"))
        self.cb_home = add(1, "Home team", ttk.Combobox(top, width=6, state="readonly"))
        self.e_date = add(2, "Date (YYYY-MM-DD)", ttk.Entry(top, width=12))
        self.e_date.insert(0, dt.date.today().isoformat())
        self.cb_venue = add(3, "Stadium", ttk.Combobox(top, width=28))
        self.cb_dn = add(4, "Day/Night", ttk.Combobox(
            top, width=7, state="readonly", values=["day", "night"]))
        self.cb_dn.set("day")
        self.sp_temp = add(5, "Temp °F", ttk.Spinbox(top, from_=20, to=115, width=5))
        self.sp_temp.set(72)
        self.sp_wind = add(6, "Wind mph", ttk.Spinbox(top, from_=0, to=45, width=5))
        self.sp_wind.set(6)
        self.cb_wdir = add(7, "Wind dir", ttk.Combobox(top, width=14))
        self.cb_cond = add(8, "Condition", ttk.Combobox(top, width=14))
        # Editable; leave blank for a neutral-ump prediction. Known names
        # resolve to an HpUmpId in _collect_spec; an unknown name -> no id.
        self.cb_ump = add(9, "HP Umpire", ttk.Combobox(top, width=18))

        self.cb_away.bind("<<ComboboxSelected>>", lambda e: self._team_changed("away"))
        self.cb_home.bind("<<ComboboxSelected>>", lambda e: self._team_changed("home"))

        mid = ttk.Frame(body)
        mid.pack(fill="both", expand=True, padx=8)
        self.side_widgets = {}
        for i, side in enumerate(("away", "home")):
            f = ttk.LabelFrame(mid, text=f"{side.title()} lineup")
            f.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            mid.columnconfigure(i, weight=1)
            w = {"lineup": []}
            ttk.Label(f, text="Starting pitcher").grid(row=0, column=0, sticky="w")
            w["starter"] = ttk.Combobox(f, width=34)
            w["starter"].grid(row=0, column=1, pady=2, sticky="w")
            for slot in range(1, 10):
                ttk.Label(f, text=str(slot)).grid(row=slot, column=0, sticky="w")
                cb = ttk.Combobox(f, width=34)
                cb.grid(row=slot, column=1, pady=1, sticky="w")
                cb.bind("<<ComboboxSelected>>",
                        lambda e, s=side: self._refresh_lineup_options(s))
                w["lineup"].append(cb)
            b = ttk.Button(f, text="Auto-fill from last game",
                           command=lambda s=side: self._autofill(s))
            b.grid(row=10, column=1, sticky="e", pady=4)
            self.side_widgets[side] = w

        slate_f = ttk.LabelFrame(body, text="Slate (click a game to load it "
                                            "into the form and edit; Predict "
                                            "runs them all)")
        slate_f.pack(fill="x", padx=8, pady=4)
        self.slate = []
        self._loaded_idx = None   # slate index currently loaded into the form
        # exportselection=False: keep the row selected while the user edits
        # form fields (otherwise clicking a combobox clears the selection)
        self.lb_slate = tk.Listbox(slate_f, height=8, bg=WHITE, fg=NAVY,
                                   selectbackground=RED, exportselection=False,
                                   font=("Segoe UI", 10))
        self.lb_slate.pack(side="left", fill="x", expand=True, padx=(6, 0), pady=4)
        self.lb_slate.bind("<<ListboxSelect>>", self._slate_selected)
        lb_vsb = ttk.Scrollbar(slate_f, orient="vertical",
                               command=self.lb_slate.yview)
        self.lb_slate.configure(yscrollcommand=lb_vsb.set)
        lb_vsb.pack(side="left", fill="y", pady=4)
        sb = ttk.Frame(slate_f)
        sb.pack(side="left", padx=6)
        ttk.Button(sb, text="Add game to slate",
                   command=self._add_to_slate).pack(fill="x", pady=1)
        ttk.Button(sb, text="Update selected game",
                   command=self._update_selected).pack(fill="x", pady=1)
        ttk.Button(sb, text="Remove selected",
                   command=self._remove_from_slate).pack(fill="x", pady=1)
        ttk.Button(sb, text="Clear slate",
                   command=self._clear_slate).pack(fill="x", pady=1)
        ttk.Button(sb, text="Load today's file",
                   command=self._load_todays_file).pack(fill="x", pady=1)

    # ------------------------------------------------------- interaction

    def _team_changed(self, side):
        team = (self.cb_away if side == "away" else self.cb_home).get()
        pool = self.pools.get(team, {"batters": {}, "pitchers": {}})
        w = self.side_widgets[side]
        w["starter"]["values"] = sorted(pool["pitchers"])
        w["starter"].set("")
        for cb in w["lineup"]:
            cb.set("")
        self._refresh_lineup_options(side)
        self._refresh_team_options()
        if side == "home" and team in self.team_park:
            self.cb_venue.set(self.team_park[team])

    def _refresh_team_options(self):
        """Each team dropdown hides the team picked on the other side."""
        teams = sorted(self.pools)
        away, home = self.cb_away.get(), self.cb_home.get()
        self.cb_away["values"] = [t for t in teams if t != home]
        self.cb_home["values"] = [t for t in teams if t != away]

    def _refresh_lineup_options(self, side):
        """Hide already-selected players from the other lineup slots."""
        team = (self.cb_away if side == "away" else self.cb_home).get()
        pool = sorted(self.pools.get(team, {"batters": {}})["batters"])
        w = self.side_widgets[side]
        chosen = {cb.get().strip() for cb in w["lineup"] if cb.get().strip()}
        for cb in w["lineup"]:
            own = cb.get().strip()
            cb["values"] = [p for p in pool if p not in chosen or p == own]

    def _autofill(self, side):
        team = (self.cb_away if side == "away" else self.cb_home).get()
        if not team or self.pred is None:
            return
        gb = self.pred.stores.raw["gb"]
        gp = self.pred.stores.raw["gp"]
        rows = gb[(gb["Team"] == team) & gb["BattingOrder"].notna()].copy()
        rows["bo"] = pd.to_numeric(rows["BattingOrder"], errors="coerce")
        rows = rows[rows["bo"] % 100 == 0]
        if rows.empty:
            return
        last_date = rows["Date"].max()
        last = rows[rows["Date"] == last_date].sort_values("bo")
        w = self.side_widgets[side]
        pool = self.pools.get(team, {"batters": {}})
        pid_label = {v: k for k, v in pool["batters"].items()}
        for cb, (_, r) in zip(w["lineup"], last.iterrows()):
            label = pid_label.get(r["PlayerId"], f'{r["Name"]} [{r["PlayerId"]}]')
            cb.set(label)
        st = gp[(gp["Team"] == team) & (gp["GS"] == 1)].sort_values("Date")
        if len(st):
            sp = st.iloc[-1]
            plabel = {v: k for k, v in pool.get("pitchers", {}).items()}.get(
                sp["PlayerId"], f'{sp["Name"]} [{sp["PlayerId"]}]')
            w["starter"].set(plabel)
        self._refresh_lineup_options(side)
        self.status.set(f"{side} lineup auto-filled from {last_date.date()} "
                        f"(edit as needed)")

    def _resolve(self, team, label, kind):
        """Combobox label -> PlayerId (supports 'Name [id]' fallback labels)."""
        pool = self.pools.get(team, {})
        pid = pool.get(kind, {}).get(label)
        if pid is not None:
            return int(pid)
        if "[" in label and label.endswith("]"):
            return int(label.rsplit("[", 1)[1][:-1])
        raise ValueError(f"unknown player: {label!r}")

    def _label_for(self, team, pid, kind, names=None):
        """PlayerId -> combobox label, the inverse of _resolve. Players not in
        the team's pool get a 'Name [id]' fallback label (which _resolve
        parses back), using the spec's scraped names when available."""
        pid = int(pid)
        for label, p in self.pools.get(team, {}).get(kind, {}).items():
            if int(p) == pid:
                return label
        name = (names or {}).get(str(pid))
        if not name and self.pred is not None:
            name = self.pred._name(pid)
        return f'{name or pid} [{pid}]'

    def _apply_spec(self, spec):
        """Fill every form field from a game spec — the inverse of
        _collect_spec, so a slate game can be loaded, edited, and saved back
        with 'Update selected game'."""
        self.cb_away.set(spec.get("away_team") or "")
        self._team_changed("away")
        self.cb_home.set(spec.get("home_team") or "")
        self._team_changed("home")            # sets the home park default...
        self.e_date.delete(0, "end")
        self.e_date.insert(0, spec.get("date") or dt.date.today().isoformat())
        self.cb_venue.set(spec.get("venue") or "")   # ...the spec venue wins
        self.cb_dn.set(spec.get("day_night") or "")
        for widget, v in ((self.sp_temp, spec.get("temp")),
                          (self.sp_wind, spec.get("wind_speed"))):
            widget.set("" if v is None else v)
        self.cb_wdir.set(spec.get("wind_dir") or "")
        self.cb_cond.set(spec.get("condition") or "")
        self.cb_ump.set(spec.get("hp_ump") or "")

        names = spec.get("names") or {}
        for side in ("away", "home"):
            team = spec.get(f"{side}_team")
            w = self.side_widgets[side]
            st = spec.get(f"{side}_starter")
            w["starter"].set(
                self._label_for(team, st, "pitchers", names) if st else "")
            for cb in w["lineup"]:
                cb.set("")
            for pid, slot in spec.get(f"{side}_lineup", []):
                if 1 <= int(slot) <= 9:
                    w["lineup"][int(slot) - 1].set(
                        self._label_for(team, pid, "batters", names))
            self._refresh_lineup_options(side)

    def _collect_spec(self):
        """Teams, date and at least one lineup player are required; anything
        else may be left blank — missing inputs become NaN features."""
        away, home = self.cb_away.get(), self.cb_home.get()
        if not away or not home or away == home:
            raise ValueError("pick two different teams")
        date = self.e_date.get().strip()
        dt.date.fromisoformat(date)

        def num(widget, label):
            v = str(widget.get()).strip()
            if not v:
                return None
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"{label} is not a number: {v!r}")

        spec = {"date": date, "away_team": away, "home_team": home,
                "venue": self.cb_venue.get().strip(),
                "day_night": self.cb_dn.get(),
                "temp": num(self.sp_temp, "temperature"),
                "wind_speed": num(self.sp_wind, "wind speed"),
                "wind_dir": self.cb_wdir.get(), "condition": self.cb_cond.get()}
        ump = self.cb_ump.get().strip()
        spec["hp_ump"] = ump or None
        spec["hp_ump_id"] = self.ump_name_to_id.get(ump)   # None if unknown
        for side, team in (("away", away), ("home", home)):
            w = self.side_widgets[side]
            st = w["starter"].get().strip()
            spec[f"{side}_starter"] = (self._resolve(team, st, "pitchers")
                                       if st else None)
            lineup = []
            for slot, cb in enumerate(w["lineup"], start=1):
                lab = cb.get().strip()
                if lab:
                    lineup.append((self._resolve(team, lab, "batters"), slot))
            if len({p for p, _ in lineup}) != len(lineup):
                raise ValueError(f"duplicate player in {side} lineup")
            spec[f"{side}_lineup"] = lineup
        if not spec["away_lineup"] and not spec["home_lineup"]:
            raise ValueError("fill in at least one lineup player")
        return spec

    @staticmethod
    def _slate_row_text(spec):
        n = len(spec.get("away_lineup", [])) + len(spec.get("home_lineup", []))
        return (f'{spec["date"]}  {spec["away_team"]} @ '
                f'{spec["home_team"]}  ({n} batters)')

    def _add_to_slate(self):
        try:
            spec = self._collect_spec()
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return
        self.slate.append(spec)
        self.lb_slate.insert("end", self._slate_row_text(spec))
        for side in ("away", "home"):
            w = self.side_widgets[side]
            w["starter"].set("")
            for cb in w["lineup"]:
                cb.set("")
        self.cb_away.set("")
        self.cb_home.set("")
        self._refresh_team_options()
        self.status.set(f"Slate: {len(self.slate)} game(s). Add more or Predict.")

    def _slate_selected(self, _event=None):
        """Clicking a slate game loads it into the form for editing."""
        sel = self.lb_slate.curselection()
        if not sel or not (0 <= sel[0] < len(self.slate)):
            return
        self._loaded_idx = sel[0]
        spec = self.slate[sel[0]]
        self._apply_spec(spec)
        self.status.set(
            f'Loaded {spec["away_team"]} @ {spec["home_team"]} into the form '
            f'— edit, then "Update selected game" to save it back. '
            f'Predict still runs the whole slate.')

    def _update_selected(self):
        """Write the form back over the selected (or last-loaded) slate game."""
        sel = self.lb_slate.curselection()
        idx = sel[0] if sel else self._loaded_idx
        if idx is None or not (0 <= idx < len(self.slate)):
            messagebox.showinfo(
                "No game selected",
                "Click a slate game first — it loads into the form. Edit it, "
                "then Update selected game saves your changes back.")
            return
        try:
            spec = self._collect_spec()
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return
        old = self.slate[idx]
        if old.get("names"):     # keep scraped display names for re-loading
            spec["names"] = old["names"]
        self.slate[idx] = spec
        self.lb_slate.delete(idx)
        self.lb_slate.insert(idx, self._slate_row_text(spec))
        self.lb_slate.selection_clear(0, "end")
        self.lb_slate.selection_set(idx)
        self.lb_slate.see(idx)
        self._loaded_idx = idx
        self.status.set(f'Updated game {idx + 1} of {len(self.slate)}: '
                        f'{spec["away_team"]} @ {spec["home_team"]}.')

    def _remove_from_slate(self):
        sel = self.lb_slate.curselection()
        if sel:
            self.slate.pop(sel[0])
            self.lb_slate.delete(sel[0])
            self._loaded_idx = None

    def _clear_slate(self):
        self.slate.clear()
        self.lb_slate.delete(0, "end")
        self._loaded_idx = None

    def _predict_clicked(self):
        if self.slate:
            specs = list(self.slate)
        else:
            try:
                specs = [self._collect_spec()]
            except Exception as e:
                messagebox.showerror("Input error", str(e))
                return
        self.btn_predict["state"] = "disabled"
        self.status.set(f"Predicting {len(specs)} game(s)...")
        self._pred_state = None
        threading.Thread(target=self._predict_run, args=(specs,),
                         daemon=True).start()
        self.after(200, self._poll_predict)

    def _predict_run(self, specs):
        try:
            from predict import save_excel_slate
            out = self.pred.predict_slate(specs)
            xlsx = save_excel_slate(specs, out)
            self._pred_state = ("ok", (specs, out, xlsx))
        except Exception as e:
            self._pred_state = ("err", str(e))

    def _poll_predict(self):
        if self._pred_state is None:
            self.after(200, self._poll_predict)
            return
        state, payload = self._pred_state
        self.btn_predict["state"] = "normal"
        if state == "ok":
            specs, out, xlsx = payload
            self.status.set(f"Saved: {xlsx}")
            self._show_results(specs, out, xlsx)
        else:
            self.status.set("Ready.")
            messagebox.showerror("Prediction failed", payload)

    # ---------------------------------------------------------- results

    @staticmethod
    def _make_sortable(tv, cols, top_iids=()):
        """Click a column header to sort by it; click again to flip.
        First click puts the best (highest) values on top. Stripes are
        re-applied after each sort; rows in top_iids keep their highlight."""
        state = {}

        def to_num(s):
            s = str(s).strip().rstrip("%")
            try:
                return float(s)
            except ValueError:
                return None

        def sort_by(col):
            desc = not state.get(col, False)  # first click = best on top
            state.clear()
            state[col] = desc
            rows = [(tv.set(iid, col), iid) for iid in tv.get_children("")]
            nums = {iid: to_num(v) for v, iid in rows}
            numeric = sum(n is not None for n in nums.values()) >= len(rows) / 2
            if numeric:  # blanks always sort last
                rows.sort(key=lambda t: (nums[t[1]] is None,
                                         (nums[t[1]] or 0)
                                         * (-1 if desc else 1)))
            else:
                rows.sort(key=lambda t: str(t[0]).lower(), reverse=desc)
            for i, (_, iid) in enumerate(rows):
                tv.move(iid, "", i)
                tag = ("top" if iid in top_iids
                       else ("stripe" if i % 2 else ""))
                tv.item(iid, tags=(tag,))

        for c in cols:
            tv.heading(c, text=c, command=lambda c=c: sort_by(c))

    def _show_results(self, specs, out, xlsx=None):
        multi = len(specs) > 1
        win = tk.Toplevel(self)
        if multi:
            title = f'{specs[0]["date"]} — slate of {len(specs)} games'
        else:
            s = specs[0]
            title = f'{s["date"]} {s["away_team"]} @ {s["home_team"]}'
        win.title(title)
        win.geometry("1180x880")
        win.configure(bg=NAVY)

        head = tk.Frame(win, bg=NAVY)
        head.pack(fill="x", padx=8, pady=(10, 2))
        self._result_logo = load_logo(height=44)
        if self._result_logo is not None:
            tk.Label(head, image=self._result_logo, bg=NAVY).pack(
                side="left", padx=(4, 12))
        ttk.Label(head, style="Title.TLabel", text=title).pack(side="left")

        # Bottom-pinned sections are packed BEFORE the expanding batter board
        # so they can never be pushed off-screen: export buttons, then the
        # starter table, then (top) the games table and batter board.

        # --- export buttons (pinned to the very bottom) ---
        # (no combined-copy button: Predict already writes the combined
        # workbook automatically — its path shows in the footer label)
        def export_per_game():
            folder = filedialog.askdirectory(
                title="Folder for the per-game Excel files")
            if folder:
                from predict import save_excel_per_game
                paths = save_excel_per_game(specs, out, folder)
                messagebox.showinfo(
                    "Exported", f"Wrote {len(paths)} files to {folder}")

        foot = tk.Frame(win, bg=NAVY)
        foot.pack(side="bottom", fill="x", padx=12, pady=6)
        if multi:
            ttk.Button(foot, text="Export one file per game...",
                       command=export_per_game).pack(side="right", padx=6)
        if xlsx:
            ttk.Label(foot, text=f"Saved to: {xlsx}", style="Sub.TLabel").pack(
                side="left")

        # The three boards below share the space above the buttons equally:
        # same base row height + fill/expand => identical rendered heights.
        COMMON_H = 6

        # --- game predictions (top) ---
        ttk.Label(win, text="GAME PREDICTIONS",
                  font=("Segoe UI", 11, "bold")).pack(side="top", anchor="w",
                                                      padx=12, pady=(4, 0))
        gcols = ["Game", "Venue", "Winner", "Expected score",
                 "xHR", "xRuns", "P>8.5", "P>9.5"]
        gframe, gv = self._tree_with_scroll(
            win, columns=gcols, show="headings", height=COMMON_H)
        for c, wpx in zip(gcols, [100, 195, 95, 135, 60, 60, 60, 60]):
            gv.heading(c, text=c)
            gv.column(c, width=wpx, anchor="center")
        gv.tag_configure("stripe", background=STRIPE)
        for i, (_, g) in enumerate(out["games"].iterrows()):
            away, home = g["Game"].split("@")
            gv.insert("", "end", tags=("stripe" if i % 2 else "",), values=(
                g["Game"], g["Venue"] or "—",
                f'{g["Winner"]} {g["WinProb"]:.0%}',
                f'{away} {g["exp_away_runs"]:.1f} — '
                f'{home} {g["exp_home_runs"]:.1f}',
                g["exp_lineup_HR"], g["exp_total_runs"],
                *[f'{g[f"P_runs_over_{ln}"]:.0%}' for ln in ["8.5", "9.5"]]))
        gframe.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 4))
        self._make_sortable(gv, gcols)

        # --- batter prop board ---
        ttk.Label(win, text="BATTER PROP BOARD",
                  font=("Segoe UI", 11, "bold")).pack(side="top", anchor="w",
                                                      padx=12)
        cols = (["Game"] if multi else []) + \
            ["Team", "Slot", "Name", "HR chance", "Fair odds", "Hit",
             "2+ hits", "2+ bases", "Scores run", "RBI", "Walk", "Steal"]
        widths = ([95] if multi else []) + \
            [55, 45, 195, 70, 75, 65, 65, 65, 65, 65, 65, 65]
        tvframe, tv = self._tree_with_scroll(
            win, columns=cols, show="headings", height=COMMON_H)
        for c, wpx in zip(cols, widths):
            tv.heading(c, text=c)
            tv.column(c, width=wpx, anchor="center")
        tv.tag_configure("stripe", background=STRIPE)
        tv.tag_configure("top", background=TOPPICK,
                         font=("Segoe UI", 10, "bold"))
        top_iids = set()
        for i, (_, r) in enumerate(out["batters"].iterrows()):
            tag = "top" if i < 3 else ("stripe" if i % 2 else "")
            name = r["Name"] + (" *" if "CareerG" in r
                                and r["CareerG"] < 50 else "")
            vals = ([r["Game"]] if multi else []) + [
                r["Team"], r["slot"], name, f'{r["P_HR"]:.1%}',
                r["HR_fair_odds"], f'{r["P_Hit"]:.0%}', f'{r["P_2Hits"]:.0%}',
                f'{r["P_TB2"]:.0%}', f'{r["P_Run"]:.0%}', f'{r["P_RBI"]:.0%}',
                f'{r["P_BB"]:.0%}', f'{r["P_SB"]:.0%}']
            iid = tv.insert("", "end", tags=(tag,), values=vals)
            if i < 3:
                top_iids.add(iid)
        tvframe.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 4))
        self._make_sortable(tv, cols, top_iids)

        # --- starter strikeouts ---
        ttk.Label(win, text="STARTER STRIKEOUTS",
                  font=("Segoe UI", 11, "bold")).pack(side="top", anchor="w",
                                                      padx=12)
        scols = (["Game"] if multi else []) + \
            ["Name", "Team", "xK", "P>3.5", "P>4.5", "P>5.5", "P>6.5", "P>7.5"]
        svframe, sv = self._tree_with_scroll(
            win, columns=scols, show="headings", height=COMMON_H)
        for c in scols:
            sv.heading(c, text=c)
            sv.column(c, width=100, anchor="center")
        sv.tag_configure("stripe", background=STRIPE)
        for i, (_, r) in enumerate(out["starters"].iterrows()):
            vals = ([r["Game"]] if multi else []) + [
                r["Name"], r["Team"], f'{r["xK"]:.1f}',
                *[f'{r[f"P_over_{ln}"]:.0%}' for ln in [3.5, 4.5, 5.5, 6.5, 7.5]]]
            sv.insert("", "end", tags=("stripe" if i % 2 else "",), values=vals)
        svframe.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 4))
        self._make_sortable(sv, scols)


if __name__ == "__main__":
    App().mainloop()
