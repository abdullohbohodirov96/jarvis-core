#!/usr/bin/env python3
"""
JARVIS Desktop GUI — O'zbek tilidagi AI yordamchi
==================================================
Ishga tushirish:  python jarvis/desktop_app.py
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import threading
import wave
from datetime import datetime
from pathlib import Path

import customtkinter as ctk

# jarvis/ papkasini sys.path ga qo'shamiz
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Ranglar ──────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

C_BG       = "#0d0d0d"
C_SIDEBAR  = "#111111"
C_SURFACE  = "#1c1c1c"
C_BORDER   = "#2a2a2a"
C_ACCENT   = "#3b82f6"
C_ACC_HV   = "#2563eb"
C_USER_BG  = "#1e3a5f"
C_BOT_BG   = "#1a1a2e"
C_TASK_BG  = "#182018"
C_TEXT     = "#e2e8f0"
C_MUTED    = "#64748b"
C_GREEN    = "#22c55e"
C_RED      = "#ef4444"
C_YELLOW   = "#eab308"
C_MIC_ACT  = "#dc2626"   # mikrofon yoqilganda qizil

# ── Tizim prompt — o'zbek tilida ─────────────────────────────────────────────
SYSTEM_PROMPT_UZ = """Sen JARVIS — foydalanuvchining shaxsiy AI yordamchisissan.

Qoidalar:
- Har doim O'ZBEK TILIDA javob ber.
- Qisqa, aniq va do'stona gapir.
- Vazifa qo'shish/ko'rish so'ralsa, JSON formatida quyidagicha qaytargin:
  {"action": "add_task", "title": "...", "priority": "yuqori|o'rta|past"}
  {"action": "list_tasks"}
- Salomlashganda foydalanuvchini ismi bilan murojaat qil.
- Texnik narsalar haqida so'ralsa tushuntirish berishdan tortinma."""


# ═══════════════════════════════════════════════════════════════════════════════
# Async ko'prik
# ═══════════════════════════════════════════════════════════════════════════════
class AsyncHelper:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

    def run(self, coro, callback=None):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        if callback:
            future.add_done_callback(callback)
        return future

    def stop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)


# ═══════════════════════════════════════════════════════════════════════════════
# Xabar bubble
# ═══════════════════════════════════════════════════════════════════════════════
class Bubble(ctk.CTkFrame):
    def __init__(self, parent, role: str, text: str, ts: str, **kw):
        bg = C_USER_BG if role == "user" else C_BOT_BG
        super().__init__(parent, fg_color=bg, corner_radius=14, **kw)

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(8, 0))

        ctk.CTkLabel(
            hdr,
            text="Siz" if role == "user" else "⬡ JARVIS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=C_ACCENT if role == "user" else C_GREEN,
        ).pack(side="left")

        ctk.CTkLabel(
            hdr, text=ts,
            font=ctk.CTkFont(size=10),
            text_color=C_MUTED,
        ).pack(side="right")

        ctk.CTkLabel(
            self, text=text,
            font=ctk.CTkFont(size=13),
            text_color=C_TEXT,
            wraplength=520,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(4, 10))


# ═══════════════════════════════════════════════════════════════════════════════
# Vazifa kartochkasi
# ═══════════════════════════════════════════════════════════════════════════════
class TaskCard(ctk.CTkFrame):
    PRIORITY_COLOR = {"yuqori": C_RED, "o'rta": C_YELLOW, "past": C_GREEN}

    def __init__(self, parent, task: dict, on_done, **kw):
        super().__init__(parent, fg_color=C_TASK_BG, corner_radius=8, **kw)
        self._task = task
        self._on_done = on_done

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=6)

        dot_color = self.PRIORITY_COLOR.get(task.get("priority", "o'rta"), C_YELLOW)
        ctk.CTkLabel(row, text="●", text_color=dot_color,
                     font=ctk.CTkFont(size=10)).pack(side="left", padx=(0, 6))

        ctk.CTkLabel(
            row, text=task["title"],
            font=ctk.CTkFont(size=12),
            text_color=C_TEXT, anchor="w",
        ).pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            row, text="✓", width=26, height=22,
            fg_color=C_SURFACE, hover_color=C_GREEN,
            text_color=C_MUTED,
            command=self._complete,
        ).pack(side="right")

    def _complete(self):
        self._on_done(self._task)
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# Asosiy ilova
# ═══════════════════════════════════════════════════════════════════════════════
class JarvisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("⬡ JARVIS — AI Yordamchi")
        self.geometry("1000x680")
        self.minsize(780, 520)
        self.configure(fg_color=C_BG)

        self._async      = AsyncHelper()
        self._thinking   = False
        self._recording  = False
        self._audio_buf  = []
        self._sd_stream  = None
        self._tasks: list[dict] = []
        self._history: list[dict] = []   # conversation history

        self._build_ui()
        self._greet()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_chat()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=220, fg_color=C_SIDEBAR, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_rowconfigure(5, weight=1)

        # Logo
        ctk.CTkLabel(sb, text="⬡  JARVIS",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=C_ACCENT).grid(
            row=0, column=0, padx=16, pady=(22, 2), sticky="w")
        ctk.CTkLabel(sb, text="AI Shaxsiy Yordamchi",
                     font=ctk.CTkFont(size=10), text_color=C_MUTED).grid(
            row=1, column=0, padx=16, sticky="w")

        sep = ctk.CTkFrame(sb, height=1, fg_color=C_BORDER)
        sep.grid(row=2, column=0, padx=16, pady=12, sticky="ew")

        # API kalit
        keys_frame = ctk.CTkFrame(sb, fg_color="transparent")
        keys_frame.grid(row=3, column=0, padx=16, sticky="ew")

        ctk.CTkLabel(keys_frame, text="OpenAI API Kalit",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=C_MUTED).pack(anchor="w", pady=(0, 4))

        self._api_entry = ctk.CTkEntry(
            keys_frame, placeholder_text="sk-…",
            show="*", height=32,
            fg_color=C_SURFACE, border_color=C_BORDER, text_color=C_TEXT)
        self._api_entry.pack(fill="x")

        env_key = os.environ.get("OPENAI_API_KEY", "")
        if env_key:
            self._api_entry.insert(0, env_key)

        ctk.CTkButton(keys_frame, text="Saqlash",
                      height=30, fg_color=C_ACCENT, hover_color=C_ACC_HV,
                      command=self._apply_key).pack(fill="x", pady=(6, 0))

        # Model
        ctk.CTkLabel(keys_frame, text="Model",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=C_MUTED).pack(anchor="w", pady=(14, 4))
        self._model_var = ctk.StringVar(value="gpt-4o")
        ctk.CTkOptionMenu(keys_frame,
                          values=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                          variable=self._model_var,
                          fg_color=C_SURFACE, button_color=C_ACCENT,
                          button_hover_color=C_ACC_HV,
                          dropdown_fg_color=C_SURFACE, text_color=C_TEXT,
                          ).pack(fill="x")

        sep2 = ctk.CTkFrame(sb, height=1, fg_color=C_BORDER)
        sep2.grid(row=4, column=0, padx=16, pady=12, sticky="ew")

        # Vazifalar
        task_frame = ctk.CTkFrame(sb, fg_color="transparent")
        task_frame.grid(row=5, column=0, padx=16, sticky="nsew")
        task_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(task_frame, text="📋  Vazifalar",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=C_TEXT).grid(row=0, column=0, sticky="w")

        self._task_scroll = ctk.CTkScrollableFrame(
            task_frame, fg_color="transparent",
            scrollbar_button_color=C_BORDER)
        self._task_scroll.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        task_frame.grid_columnconfigure(0, weight=1)

        # Holat
        self._status = ctk.CTkLabel(sb, text="●  Tayyor",
                                    font=ctk.CTkFont(size=11),
                                    text_color=C_GREEN)
        self._status.grid(row=6, column=0, padx=16, pady=(8, 4), sticky="w")

        ctk.CTkButton(sb, text="🗑  Suhbatni tozala",
                      height=30, fg_color=C_SURFACE, hover_color=C_BORDER,
                      text_color=C_TEXT, command=self._clear).grid(
            row=7, column=0, padx=16, pady=(0, 16), sticky="ew")

    # ── Chat panel ────────────────────────────────────────────────────────────
    def _build_chat(self):
        main = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Scroll
        self._scroll = ctk.CTkScrollableFrame(
            main, fg_color=C_BG,
            scrollbar_button_color=C_BORDER,
            scrollbar_button_hover_color=C_MUTED)
        self._scroll.grid(row=0, column=0, sticky="nsew")
        self._scroll.grid_columnconfigure(0, weight=1)

        # Fikrlash ko'rsatkichi
        self._think_lbl = ctk.CTkLabel(
            main, text="⏳  JARVIS o'ylamoqda…",
            font=ctk.CTkFont(size=12), text_color=C_MUTED)

        # Input panel
        bar = ctk.CTkFrame(main, fg_color=C_SURFACE, corner_radius=14)
        bar.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 4))
        bar.grid_columnconfigure(0, weight=1)

        self._input = ctk.CTkTextbox(
            bar, height=50, fg_color="transparent",
            border_width=0, text_color=C_TEXT,
            font=ctk.CTkFont(size=13), wrap="word")
        self._input.grid(row=0, column=0, sticky="ew", padx=12, pady=8)
        self._input.bind("<Return>", self._on_enter)

        btn_frame = ctk.CTkFrame(bar, fg_color="transparent")
        btn_frame.grid(row=0, column=1, padx=(0, 8))

        # Mikrofon tugmasi
        self._mic_btn = ctk.CTkButton(
            btn_frame, text="🎤", width=46, height=36,
            fg_color=C_SURFACE, hover_color=C_BORDER,
            font=ctk.CTkFont(size=18),
            command=self._toggle_mic)
        self._mic_btn.pack(side="left", padx=(0, 6))

        # Yuborish
        self._send_btn = ctk.CTkButton(
            btn_frame, text="Yuborish ↵",
            width=100, height=36,
            fg_color=C_ACCENT, hover_color=C_ACC_HV,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_send)
        self._send_btn.pack(side="left")

        ctk.CTkLabel(main,
                     text="Enter — yuborish  •  Shift+Enter — yangi qator  •  🎤 — ovoz",
                     font=ctk.CTkFont(size=10), text_color=C_MUTED).grid(
            row=3, column=0, pady=(0, 8))

    # ── Salomlashish ──────────────────────────────────────────────────────────
    def _greet(self):
        self._add_bot(
            "Salom! Men JARVIS — sizning shaxsiy AI yordamchingizman.\n\n"
            "Boshlash uchun:\n"
            "1. Chap panelda OpenAI API kalitingizni kiriting\n"
            "2. Yozing yoki 🎤 tugmasini bosib gapiringm\n"
            "3. O'zbek tilida so'rang — o'zbekcha javob beraman!"
        )

    # ── Holat ─────────────────────────────────────────────────────────────────
    def _set_status(self, text: str, color: str):
        self._status.configure(text=f"●  {text}", text_color=color)

    # ── Xabarlar ──────────────────────────────────────────────────────────────
    def _add_user(self, text: str):
        ts = datetime.now().strftime("%H:%M")
        b = Bubble(self._scroll, "user", text, ts)
        b.grid(sticky="e", padx=(60, 12), pady=3,
               row=len(self._scroll.winfo_children()), column=0)
        self._scroll_bottom()

    def _add_bot(self, text: str):
        ts = datetime.now().strftime("%H:%M")
        b = Bubble(self._scroll, "assistant", text, ts)
        b.grid(sticky="w", padx=(12, 60), pady=3,
               row=len(self._scroll.winfo_children()), column=0)
        self._scroll_bottom()

    def _scroll_bottom(self):
        self.after(80, lambda: self._scroll._parent_canvas.yview_moveto(1.0))

    # ── Suhbat tozalash ───────────────────────────────────────────────────────
    def _clear(self):
        for w in self._scroll.winfo_children():
            w.destroy()
        self._history.clear()
        self._add_bot("Suhbat tozalandi. Yangi savol bering!")

    # ── API kalit ─────────────────────────────────────────────────────────────
    def _apply_key(self):
        key = self._api_entry.get().strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
            self._set_status("Tayyor", C_GREEN)
            self._add_bot("API kalit saqlandi ✅ Endi gapira olasiz!")

    # ── Vazifalar ─────────────────────────────────────────────────────────────
    def _add_task(self, title: str, priority: str = "o'rta"):
        task = {"title": title, "priority": priority, "done": False}
        self._tasks.append(task)
        card = TaskCard(self._task_scroll, task, on_done=self._complete_task)
        card.pack(fill="x", pady=3)

    def _complete_task(self, task: dict):
        task["done"] = True

    def _refresh_tasks(self):
        for w in self._task_scroll.winfo_children():
            w.destroy()
        for t in self._tasks:
            if not t.get("done"):
                TaskCard(self._task_scroll, t, on_done=self._complete_task).pack(
                    fill="x", pady=3)

    # ── Mikrofon ──────────────────────────────────────────────────────────────
    def _toggle_mic(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            self._add_bot("❌ sounddevice o'rnatilmagan:\n`pip install sounddevice scipy`")
            return

        self._recording = True
        self._audio_buf = []
        self._mic_btn.configure(text="⏹", fg_color=C_MIC_ACT, hover_color=C_MIC_ACT)
        self._set_status("Tinglayapman…", C_RED)

        def callback(indata, frames, time, status):
            self._audio_buf.append(indata.copy())

        self._sd_stream = sd.InputStream(
            samplerate=16000, channels=1, dtype="float32", callback=callback)
        self._sd_stream.start()

    def _stop_recording(self):
        import numpy as np

        self._recording = False
        self._mic_btn.configure(text="🎤", fg_color=C_SURFACE, hover_color=C_BORDER)
        self._set_status("Ovoz tahlil qilinmoqda…", C_YELLOW)

        if self._sd_stream:
            self._sd_stream.stop()
            self._sd_stream.close()
            self._sd_stream = None

        audio = np.concatenate(self._audio_buf, axis=0) if self._audio_buf else None
        if audio is None or len(audio) < 3200:
            self._set_status("Tayyor", C_GREEN)
            return

        self._async.run(
            self._transcribe(audio),
            callback=lambda f: self.after(0, lambda: self._on_transcribed(f)),
        )

    async def _transcribe(self, audio) -> str:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return ""

        import numpy as np
        from openai import AsyncOpenAI

        # WAV faylga yozamiz
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        audio_int = (audio * 32767).astype(np.int16)
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int.tobytes())

        try:
            client = AsyncOpenAI(api_key=api_key)
            with open(tmp_path, "rb") as f:
                result = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="uz",        # O'zbek tili
                    response_format="text",
                )
            return str(result).strip()
        finally:
            os.unlink(tmp_path)

    def _on_transcribed(self, future):
        try:
            text = future.result()
        except Exception as e:
            self._set_status("Tayyor", C_GREEN)
            self._add_bot(f"❌ Ovoz tanib bo'lmadi: {e}")
            return

        if text:
            self._input.insert("end", text)
            self._set_status("Tayyor", C_GREEN)
            self._on_send()
        else:
            self._set_status("Tayyor", C_GREEN)

    # ── Xabar yuborish ────────────────────────────────────────────────────────
    def _on_enter(self, event):
        if event.state & 0x1:
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
        self._add_user(text)
        self._thinking = True
        self._send_btn.configure(state="disabled", text="…")
        self._think_lbl.grid(row=1, column=0, pady=4)
        self._set_status("O'ylamoqda…", C_YELLOW)
        self._async.run(
            self._chat(text),
            callback=lambda f: self.after(0, lambda: self._on_reply(f)),
        )

    async def _chat(self, user_text: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return "❌ Iltimos, chap panelda OpenAI API kalitingizni kiriting."

        from openai import AsyncOpenAI

        self._history.append({"role": "user", "content": user_text})

        messages = [{"role": "system", "content": SYSTEM_PROMPT_UZ}] + self._history

        client = AsyncOpenAI(api_key=api_key)
        try:
            resp = await client.chat.completions.create(
                model=self._model_var.get(),
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            reply = resp.choices[0].message.content or ""
            self._history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"❌ Xato: {e}"

    def _on_reply(self, future):
        self._thinking = False
        self._send_btn.configure(state="normal", text="Yuborish ↵")
        self._think_lbl.grid_forget()
        self._set_status("Tayyor", C_GREEN)

        try:
            reply = future.result()
        except Exception as e:
            reply = f"❌ Kutilmagan xato: {e}"

        # Vazifa buyrug'ini tekshirish
        import json, re
        m = re.search(r'\{[^}]+\}', reply)
        if m:
            try:
                cmd = json.loads(m.group())
                if cmd.get("action") == "add_task":
                    self._add_task(cmd["title"], cmd.get("priority", "o'rta"))
                    reply = reply.replace(m.group(), "").strip()
                    reply += f"\n✅ Vazifa qo'shildi: «{cmd['title']}»"
                elif cmd.get("action") == "list_tasks":
                    active = [t for t in self._tasks if not t.get("done")]
                    if active:
                        items = "\n".join(f"• {t['title']} [{t['priority']}]" for t in active)
                        reply = f"📋 Faol vazifalar:\n{items}"
                    else:
                        reply = "📋 Hozircha hech qanday vazifa yo'q."
            except Exception:
                pass

        self._add_bot(reply)

    # ── Yopish ────────────────────────────────────────────────────────────────
    def on_close(self):
        if self._recording:
            self._stop_recording()
        self._async.stop()
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = JarvisApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
