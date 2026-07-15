"""Tkinter GUI for the MLB prediction engine.

Dropdown-driven input: teams, date, stadium, starters, two ordered 9-man
lineups, day/night, and weather. Outputs per run: every batter's calibrated
HR probability (with fair odds), hit probability, both starters' projected
strikeouts with over-probabilities, and game totals.

Run:
    python Tools/3_gui.py
"""

import datetime as dt
import re
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd

import sys
# the prediction engine (predict.py and friends) lives in Model/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Model"))

# Silence known-benign pandas noise from the prediction engine (mixed-type
# CSV column inference and fragmented-frame inserts in features.py). The
# engine sources are fingerprint-guarded, so the suppression lives here.
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

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
    # opening size — wide enough for the full top form row through the HP
    # Umpire box (col 10). _fit_minsize() raises it if the built form needs more.
    START_W, START_H = 1220, 920

    def __init__(self):
        super().__init__()
        self.title("MLB Prediction Engine")
        self.geometry(f"{self.START_W}x{self.START_H}")
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
        self._fit_minsize()
        self.status.set("Loading data and models...")
        threading.Thread(target=self._load, daemon=True).start()
        self.after(200, self._poll_load)

    # ------------------------------------------------------------ setup

    def _fit_minsize(self):
        """The window itself no longer scrolls, so it must never be shrinkable
        past what the fixed form needs — otherwise content would be silently
        clipped with no scrollbar to reach it. Measure the built layout once and
        pin that as the floor (clamped to the screen, so a small display still
        gets a usable window). Everything above the floor goes to the slate, the
        one elastic region, which scrolls internally."""
        self.update_idletasks()
        need_w = min(self.winfo_reqwidth(), self.winfo_screenwidth())
        need_h = min(self.winfo_reqheight(), self.winfo_screenheight())
        self.minsize(need_w, need_h)
        self.geometry(f"{max(self.START_W, need_w)}x"
                      f"{max(self.START_H, need_h)}")

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
        data job failed (Scrapers/update_all.py writes its outcome to
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
        Tools/1_get_todays_games.py) if it exists."""
        import json
        path = DATA_DIR / "todays_games.json"
        if not path.exists():
            if not silent:
                messagebox.showinfo(
                    "No file", "Data/todays_games.json not found.\nRun: "
                    "python Tools/1_get_todays_games.py")
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
        # earliest first pitch first — the workbook sheets keep this order
        specs.sort(key=lambda s: s.get("start_et") or "99:99")
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

        # fixed bottom bar first, so it stays pinned below the body
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=8, pady=6)
        self.btn_predict = ttk.Button(bottom, text="Predict", state="disabled",
                                      command=self._predict_clicked)
        self.btn_predict.pack(side="left")
        self.status = tk.StringVar()
        ttk.Label(bottom, textvariable=self.status, wraplength=820,
                  justify="left").pack(side="left", padx=12)

        # The window itself does NOT scroll — the form is always fully visible
        # and the Predict bar stays pinned. The SLATE is the only scrolling
        # region (its listbox owns the one scrollbar): it is also the only part
        # that grows without bound, so it absorbs the spare vertical space and
        # scrolls internally once the games outrun it. _fit_minsize() below
        # then forbids shrinking the window past the fixed form, which is what
        # makes dropping the global scrollbar safe (nothing can be clipped).
        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True)

        top = ttk.LabelFrame(body, text="Game")
        top.pack(fill="x", padx=8, pady=6)

        # Two stacked input rows (game identity, then conditions) so the
        # window stays narrow now that weather has more fields.
        def add(col, text, widget, row=0):
            ttk.Label(top, text=text).grid(row=row * 2, column=col,
                                           sticky="w", padx=4)
            widget.grid(row=row * 2 + 1, column=col, sticky="w", padx=4,
                        pady=2)
            return widget

        self.cb_away = add(0, "Away team", ttk.Combobox(top, width=6, state="readonly"))
        self.cb_home = add(1, "Home team", ttk.Combobox(top, width=6, state="readonly"))
        self.e_date = add(2, "Date (YYYY-MM-DD)", ttk.Entry(top, width=12))
        self.e_date.insert(0, dt.date.today().isoformat())
        # Optional — mlb.com drops the time once a game starts, so scraped
        # slates can arrive without it; sorts and slate order fall back
        # gracefully when blank. Accepts '7:10 PM' or '19:10'.
        self.e_start = add(3, "Start ET (opt.)", ttk.Entry(top, width=10))
        self.cb_venue = add(4, "Stadium", ttk.Combobox(top, width=28))
        self.cb_dn = add(5, "Day/Night", ttk.Combobox(
            top, width=7, state="readonly", values=["day", "night"]))
        self.cb_dn.set("day")
        self.sp_temp = add(0, "Temp °F", ttk.Spinbox(top, from_=20, to=115,
                                                     width=5), row=1)
        self.sp_temp.set(72)
        self.sp_wind = add(1, "Wind mph", ttk.Spinbox(top, from_=0, to=45,
                                                      width=5), row=1)
        self.sp_wind.set(6)
        self.cb_wdir = add(2, "Wind dir", ttk.Combobox(top, width=14), row=1)
        self.cb_cond = add(3, "Condition", ttk.Combobox(top, width=14), row=1)
        # Air-density inputs (scraped from the Open-Meteo forecast by
        # 1_get_todays_games.py; editable). Blank = NaN, the model imputes.
        # Under a closed roof (Condition = Dome) the model swaps humidity
        # for a fixed indoor value, so this field only matters outdoors.
        self.sp_hum = add(4, "Humidity %", ttk.Spinbox(top, from_=0, to=100,
                                                       width=5), row=1)
        self.e_pres = add(5, "Pressure hPa", ttk.Entry(top, width=7), row=1)
        # Precip (mm at start hour, Open-Meteo forecast; 2026-07-14): the
        # outs/total heads' rain-shortening signal. Blank = NaN.
        self.e_precip = add(6, "Precip mm", ttk.Entry(top, width=6), row=1)
        # Editable; leave blank for a neutral-ump prediction. Known names
        # resolve to an HpUmpId in _collect_spec; an unknown name -> no id.
        self.cb_ump = add(7, "HP Umpire", ttk.Combobox(top, width=18), row=1)
        # spread the two input rows evenly across the panel's full width
        # (equal weights share the leftover space; no `uniform`, which would
        # force every column as wide as the Stadium box and overflow)
        for i in range(8):
            top.columnconfigure(i, weight=1)

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

        # the ONLY scrolling region in the app: it takes every spare pixel
        # (fill both / expand) and scrolls internally once the games outgrow it
        slate_f = ttk.LabelFrame(body, text="Slate (click a game to load it "
                                            "into the form and edit; Predict "
                                            "runs them all)")
        slate_f.pack(fill="both", expand=True, padx=8, pady=4)
        self.slate = []
        self._loaded_idx = None   # slate index currently loaded into the form
        # exportselection=False: keep the row selected while the user edits
        # form fields (otherwise clicking a combobox clears the selection)
        self.lb_slate = tk.Listbox(slate_f, height=8, bg=WHITE, fg=NAVY,
                                   selectbackground=RED, exportselection=False,
                                   font=("Segoe UI", 10))
        self.lb_slate.pack(side="left", fill="both", expand=True,
                           padx=(6, 0), pady=4)
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
        ttk.Button(sb, text="🡅",
                   command=lambda: self._move_slate(-1)).pack(fill="x", pady=1)
        ttk.Button(sb, text="🡇",
                   command=lambda: self._move_slate(1)).pack(fill="x", pady=1)
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
        self.e_start.delete(0, "end")
        self.e_start.insert(0, spec.get("start_et") or "")
        self.cb_venue.set(spec.get("venue") or "")   # ...the spec venue wins
        self.cb_dn.set(spec.get("day_night") or "")
        for widget, v in ((self.sp_temp, spec.get("temp")),
                          (self.sp_wind, spec.get("wind_speed")),
                          (self.sp_hum, spec.get("humidity"))):
            widget.set("" if v is None else v)
        self.e_pres.delete(0, "end")
        if spec.get("pressure") is not None:
            self.e_pres.insert(0, spec["pressure"])
        self.e_precip.delete(0, "end")
        if spec.get("precip") is not None:
            self.e_precip.insert(0, spec["precip"])
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

        start = self.e_start.get().strip()
        if start:
            m = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)?$", start, re.I)
            if not m:
                raise ValueError(f"start time not understood: {start!r} "
                                 f"(use 7:10 PM or 19:10)")
            h = int(m.group(1))
            if m.group(3):
                h = h % 12 + (12 if m.group(3).upper() == "PM" else 0)
            if not 0 <= h <= 23:
                raise ValueError(f"start time hour out of range: {start!r}")
            start = f"{h:02d}:{m.group(2)}"
        spec = {"date": date, "away_team": away, "home_team": home,
                "start_et": start or None,
                "venue": self.cb_venue.get().strip(),
                "day_night": self.cb_dn.get(),
                "temp": num(self.sp_temp, "temperature"),
                "wind_speed": num(self.sp_wind, "wind speed"),
                "humidity": num(self.sp_hum, "humidity"),
                "pressure": num(self.e_pres, "pressure"),
                "precip": num(self.e_precip, "precip"),
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
        start = spec.get("start_et") or "--:--"
        return (f'{spec["date"]}  {start} ET  {spec["away_team"]} @ '
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
        for k in ("is_dh", "dh_game2"):   # DH flags have no form field —
            if old.get(k) is not None:    # carry the scraped values through
                spec[k] = old[k]
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

    def _move_slate(self, delta):
        """Shift the selected slate game up/down — the slate order IS the
        game order of every sheet in the output workbook."""
        sel = self.lb_slate.curselection()
        if not sel:
            return
        i = sel[0]
        j = i + delta
        if not (0 <= j < len(self.slate)):
            return
        self.slate[i], self.slate[j] = self.slate[j], self.slate[i]
        row = self.lb_slate.get(i)
        self.lb_slate.delete(i)
        self.lb_slate.insert(j, row)
        self.lb_slate.selection_clear(0, "end")
        self.lb_slate.selection_set(j)
        self.lb_slate.see(j)
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
            _specs, _out, xlsx = payload
            self.status.set(f"Saved: {xlsx}")
        else:
            self.status.set("Ready.")
            messagebox.showerror("Prediction failed", payload)


if __name__ == "__main__":
    App().mainloop()
