#!/usr/bin/env python3
"""
JARVIS Desktop GUI
==================
Run:  python jarvis/desktop_app.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import customtkinter as ctk

# Ensure jarvis/ package root on sys.path
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
C_BG        = "#0d0d0d"
C_SIDEBAR   = "#111111"
C_SURFACE   = "#1a1a1a"
C_BORDER    = "#2a2a2a"
C_ACCENT    = "#3b82f6"       # blue-500
C_ACCENT_HV = "#2563eb"       # blue-600
C_USER_BG   = "#1e3a5f"
C_BOT_BG    = "#1a1a2e"
C_TEXT      = "#e2e8f0"
C_MUTED     = "#64748b"
C_GREEN     = "#22c55e"
C_RED       = "#ef4444"
C_YELLOW    = "#eab308"


# ---------------------------------------------------------------------------
# Async helper — run coroutines from the GUI thread
# ---------------------------------------------------------------------------
class AsyncHelper:
    """Bridges tkinter's event loop with asyncio."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, callback=None):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        if callback:
            future.add_done_callback(lambda f: callback(f))
        return future

    def stop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)


# ---------------------------------------------------------------------------
# Bubble widget
# ---------------------------------------------------------------------------
class MessageBubble(ctk.CTkFrame):
    def __init__(self, parent, role: str, text: str, timestamp: str, **kwargs):
        bg = C_USER_BG if role == "user" else C_BOT_BG
        super().__init__(parent, fg_color=bg, corner_radius=12, **kwargs)

        header_color = C_ACCENT if role == "user" else C_GREEN
        label = "You" if role == "user" else "JARVIS"

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(8, 2))

        ctk.CTkLabel(
            header,
            text=label,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=header_color,
        ).pack(side="left")

        ctk.CTkLabel(
            header,
            text=timestamp,
            font=ctk.CTkFont(size=10),
            text_color=C_MUTED,
        ).pack(side="right")

        ctk.CTkLabel(
            self,
            text=text,
            font=ctk.CTkFont(size=13),
            text_color=C_TEXT,
            wraplength=560,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(2, 10))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class JarvisApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("JARVIS — AI Desktop Assistant")
        self.geometry("900x650")
        self.minsize(700, 500)
        self.configure(fg_color=C_BG)

        self._async = AsyncHelper()
        self._agent = None          # JarvisAgent loaded lazily
        self._thinking = False
        self._api_key_ok = False

        self._build_ui()
        self._load_agent_async()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=200, fg_color=C_SIDEBAR, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)

        # Logo
        ctk.CTkLabel(
            sb,
            text="⬡  JARVIS",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=C_ACCENT,
        ).pack(pady=(24, 4), padx=16, anchor="w")

        ctk.CTkLabel(
            sb,
            text="AI Desktop Assistant",
            font=ctk.CTkFont(size=11),
            text_color=C_MUTED,
        ).pack(padx=16, anchor="w")

        ctk.CTkFrame(sb, height=1, fg_color=C_BORDER).pack(
            fill="x", padx=16, pady=16
        )

        # Status indicator
        self._status_dot = ctk.CTkLabel(
            sb, text="●  Starting…", font=ctk.CTkFont(size=12), text_color=C_YELLOW
        )
        self._status_dot.pack(padx=16, anchor="w")

        # API key entry
        ctk.CTkLabel(
            sb,
            text="OpenAI API Key",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C_MUTED,
        ).pack(padx=16, pady=(20, 4), anchor="w")

        self._api_entry = ctk.CTkEntry(
            sb,
            placeholder_text="sk-…",
            show="*",
            width=168,
            fg_color=C_SURFACE,
            border_color=C_BORDER,
            text_color=C_TEXT,
        )
        self._api_entry.pack(padx=16)

        # Pre-fill from env
        env_key = os.environ.get("OPENAI_API_KEY", "")
        if env_key:
            self._api_entry.insert(0, env_key)

        ctk.CTkButton(
            sb,
            text="Apply",
            width=168,
            height=30,
            fg_color=C_ACCENT,
            hover_color=C_ACCENT_HV,
            command=self._on_apply_key,
        ).pack(padx=16, pady=(6, 0))

        ctk.CTkFrame(sb, height=1, fg_color=C_BORDER).pack(
            fill="x", padx=16, pady=16
        )

        # Model selector
        ctk.CTkLabel(
            sb,
            text="Model",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C_MUTED,
        ).pack(padx=16, anchor="w")

        self._model_var = ctk.StringVar(value="gpt-4o")
        ctk.CTkOptionMenu(
            sb,
            values=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            variable=self._model_var,
            width=168,
            fg_color=C_SURFACE,
            button_color=C_ACCENT,
            button_hover_color=C_ACCENT_HV,
            dropdown_fg_color=C_SURFACE,
            text_color=C_TEXT,
        ).pack(padx=16, pady=(4, 0))

        ctk.CTkFrame(sb, height=1, fg_color=C_BORDER).pack(
            fill="x", padx=16, pady=16
        )

        # Clear button
        ctk.CTkButton(
            sb,
            text="🗑  Clear chat",
            width=168,
            height=32,
            fg_color=C_SURFACE,
            hover_color=C_BORDER,
            text_color=C_TEXT,
            command=self._clear_chat,
        ).pack(padx=16)

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Chat scroll area
        self._scroll = ctk.CTkScrollableFrame(
            main,
            fg_color=C_BG,
            scrollbar_button_color=C_BORDER,
            scrollbar_button_hover_color=C_MUTED,
        )
        self._scroll.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self._scroll.grid_columnconfigure(0, weight=1)

        # Thinking indicator (hidden initially)
        self._thinking_label = ctk.CTkLabel(
            main,
            text="JARVIS is thinking…",
            font=ctk.CTkFont(size=12),
            text_color=C_MUTED,
        )

        # Input bar
        input_bar = ctk.CTkFrame(main, fg_color=C_SURFACE, corner_radius=12)
        input_bar.grid(row=2, column=0, sticky="ew", padx=16, pady=12)
        input_bar.grid_columnconfigure(0, weight=1)

        self._input = ctk.CTkTextbox(
            input_bar,
            height=52,
            fg_color="transparent",
            border_width=0,
            text_color=C_TEXT,
            font=ctk.CTkFont(size=13),
            wrap="word",
        )
        self._input.grid(row=0, column=0, sticky="ew", padx=12, pady=8)
        self._input.bind("<Return>", self._on_enter)
        self._input.bind("<Shift-Return>", lambda e: None)  # allow newline

        self._send_btn = ctk.CTkButton(
            input_bar,
            text="Send ↵",
            width=90,
            height=36,
            fg_color=C_ACCENT,
            hover_color=C_ACCENT_HV,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_send,
        )
        self._send_btn.grid(row=0, column=1, padx=(0, 8), pady=8)

        # hint
        ctk.CTkLabel(
            main,
            text="Enter to send  •  Shift+Enter for new line",
            font=ctk.CTkFont(size=10),
            text_color=C_MUTED,
        ).grid(row=3, column=0, pady=(0, 6))

        # Welcome message
        self._add_bot_message(
            "Hello! I'm JARVIS, your AI desktop assistant. Enter your OpenAI API key in the sidebar and start chatting."
        )

    # ------------------------------------------------------------------
    # Agent loading
    # ------------------------------------------------------------------
    def _load_agent_async(self):
        self._async.run(self._load_agent(), callback=self._on_agent_loaded)

    async def _load_agent(self):
        try:
            from ai.agent import JarvisAgent
            from config.settings import settings
            self._agent = JarvisAgent()
            return True
        except Exception as e:
            return str(e)

    def _on_agent_loaded(self, future):
        result = future.result()
        if result is True:
            self.after(0, lambda: self._set_status("Ready", C_GREEN))
        else:
            self.after(0, lambda: self._set_status("Agent error", C_RED))

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def _set_status(self, text: str, color: str):
        self._status_dot.configure(text=f"●  {text}", text_color=color)

    # ------------------------------------------------------------------
    # Chat helpers
    # ------------------------------------------------------------------
    def _add_user_message(self, text: str):
        ts = datetime.now().strftime("%H:%M")
        bubble = MessageBubble(self._scroll, "user", text, ts)
        bubble.grid(
            sticky="e",
            padx=(80, 16),
            pady=(4, 4),
            row=self._next_row(),
            column=0,
        )
        self._scroll._parent_canvas.yview_moveto(1.0)

    def _add_bot_message(self, text: str):
        ts = datetime.now().strftime("%H:%M")
        bubble = MessageBubble(self._scroll, "assistant", text, ts)
        bubble.grid(
            sticky="w",
            padx=(16, 80),
            pady=(4, 4),
            row=self._next_row(),
            column=0,
        )
        self._scroll._parent_canvas.yview_moveto(1.0)

    def _next_row(self) -> int:
        children = self._scroll.winfo_children()
        return len(children)

    def _clear_chat(self):
        for w in self._scroll.winfo_children():
            w.destroy()
        if self._agent:
            self._agent.memory.clear()
        self._add_bot_message("Chat cleared. How can I help you?")

    # ------------------------------------------------------------------
    # API key
    # ------------------------------------------------------------------
    def _on_apply_key(self):
        key = self._api_entry.get().strip()
        if not key:
            return
        os.environ["OPENAI_API_KEY"] = key
        # Reload agent with new key
        self._set_status("Connecting…", C_YELLOW)
        if self._agent:
            try:
                self._agent.client.client.api_key = key
            except Exception:
                pass
        self._set_status("Ready", C_GREEN)
        self._api_key_ok = True

    # ------------------------------------------------------------------
    # Send message
    # ------------------------------------------------------------------
    def _on_enter(self, event):
        if event.state & 0x1:   # Shift held → insert newline
            return
        self._on_send()
        return "break"

    def _on_send(self):
        if self._thinking:
            return
        text = self._input.get("1.0", "end").strip()
        if not text:
            return
        self._input.delete("1.0", "end")
        self._add_user_message(text)
        self._start_thinking(text)

    def _start_thinking(self, text: str):
        self._thinking = True
        self._send_btn.configure(state="disabled", text="…")
        self._set_status("Thinking…", C_YELLOW)

        self._async.run(
            self._get_response(text),
            callback=lambda f: self.after(0, lambda: self._on_response(f)),
        )

    async def _get_response(self, text: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return "Please enter your OpenAI API key in the sidebar first."

        if self._agent is None:
            return "Agent is not loaded yet. Please wait a moment."

        try:
            model = self._model_var.get()
            # Update model if changed
            try:
                self._agent.client.model = model
            except Exception:
                pass

            response = await self._agent.process(text)
            return response.speech if hasattr(response, "speech") else str(response)
        except Exception as e:
            err = str(e)
            if "api_key" in err.lower() or "authentication" in err.lower():
                return "Invalid API key. Please check the key in the sidebar."
            return f"Error: {err}"

    def _on_response(self, future):
        self._thinking = False
        self._send_btn.configure(state="normal", text="Send ↵")
        self._set_status("Ready", C_GREEN)
        try:
            reply = future.result()
        except Exception as e:
            reply = f"Unexpected error: {e}"
        self._add_bot_message(reply)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------
    def on_close(self):
        self._async.stop()
        self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = JarvisApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
