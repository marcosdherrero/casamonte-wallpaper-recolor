# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.launch
------------------------------
Maximize on the launch monitor and start the Tk mainloop.

Withdraw first so the window does not flash at a default 200×200, then
``state('zoomed')`` on the monitor under the cursor (taskbar stays visible).
``run()`` lazy-imports WallpaperRecolorApp so this file cannot cycle with app.py.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

def _pointer_screen_xy() -> tuple[int, int]:
    """Cursor position in virtual-screen pixels (launch monitor hint)."""
    try:
        import ctypes

        # POINT: Win32 cursor in virtual-screen pixels (multi-monitor)
        class _POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = _POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return int(pt.x), int(pt.y)
    except (AttributeError, OSError, ValueError):
        pass
    try:
        root = tk._get_default_root()  # type: ignore[attr-defined]
        if root is not None:
            return int(root.winfo_pointerx()), int(root.winfo_pointery())
    except (tk.TclError, TypeError, ValueError, AttributeError):
        pass
    return 0, 0


def _monitor_work_area(x: int, y: int) -> tuple[int, int, int, int]:
    """Work area (taskbar excluded) of the monitor containing ``(x, y)``."""
    try:
        import ctypes

        class _RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        class _MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_ulong),
                ("rcMonitor", _RECT),
                ("rcWork", _RECT),
                ("dwFlags", ctypes.c_ulong),
            ]

        user32 = ctypes.windll.user32
        # POINT + MONITOR_DEFAULTTONEAREST: work area of the display under (x, y)
        class _POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = _POINT(int(x), int(y))
        MONITOR_DEFAULTTONEAREST = 2
        handle = user32.MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST)
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if handle and user32.GetMonitorInfoW(handle, ctypes.byref(info)):
            work = info.rcWork
            left, top = int(work.left), int(work.top)
            return left, top, int(work.right) - left, int(work.bottom) - top
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return 0, 0, 0, 0


def _place_maximized_on_launch_monitor(root: tk.Tk) -> None:
    """Show the window maximized on the monitor under the cursor, without a small flash.

    Withdraw until geometry is on that display, then ``state('zoomed')`` so the
    taskbar stays visible.
    """
    try:
        root.withdraw()
    except tk.TclError:
        pass
    px, py = _pointer_screen_xy()
    mx, my, mw, mh = _monitor_work_area(px, py)
    if mw < 200 or mh < 200:
        try:
            mx = int(root.winfo_vrootx())
            my = int(root.winfo_vrooty())
            mw = int(root.winfo_vrootwidth())
            mh = int(root.winfo_vrootheight())
        except tk.TclError:
            mx = my = 0
            try:
                mw = int(root.winfo_screenwidth())
                mh = int(root.winfo_screenheight())
            except tk.TclError:
                mw, mh = 1280, 800
    try:
        root.geometry(f"{max(200, mw)}x{max(200, mh)}+{mx}+{my}")
        root.update_idletasks()
        root.state("zoomed")
    except tk.TclError:
        try:
            root.geometry(f"{max(200, mw)}x{max(200, mh)}+{mx}+{my}")
        except tk.TclError:
            pass
    try:
        root.deiconify()
    except tk.TclError:
        pass



def run() -> None:
    """Launch the desktop window maximized on the launch monitor."""
    from wallpaper_recolor.ui.app import WallpaperRecolorApp  # late: avoid import cycle

    root = tk.Tk()
    try:
        root.withdraw()  # hide until geometry is on the launch monitor
    except tk.TclError:
        pass
    try:
        ttk.Style().theme_use("vista")  # native-ish on Windows
    except tk.TclError:
        pass
    WallpaperRecolorApp(root)  # builds chrome + mixins; tests skip this run() path
    _place_maximized_on_launch_monitor(root)
    root.mainloop()
