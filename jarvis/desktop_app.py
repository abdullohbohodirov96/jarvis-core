#!/usr/bin/env python3
"""
JARVIS — O'zbek tilidagi AI Yordamchi
======================================
Ishga tushirish:  python jarvis/desktop_app.py

Kerakli paketlar:
    pip install customtkinter openai sounddevice scipy edge-tts
    # Linux:  sudo apt-get install python3-tk libportaudio2
    # Mac:    brew install portaudio
"""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
import tempfile
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── customtkinter ─────────────────────────────────────────────────────────────
try:
    import customtkinter as ctk
except ImportError:
    print("Xato: pip install customtkinter")
    sys.exit(1)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Ranglar ───────────────────────────────────────────────────────────────────
BG        = "#111318"
SIDEBAR   = "#0d0f14"
SURFACE   = "#1c2028"
BORDER    = "#2a2f3a"
ACCENT    = "#4f8ef7"
ACC_HV    = "#3b7de8"
USER_BG   = "#1a3050"
BOT_BG    = "#151a2e"
TEXT      = "#dde3f0"
MUTED     = "#5a6480"
GREEN     = "#34d399"
RED       = "#f87171"
YELLOW    = "#fbbf24"
MIC_ON    = "#dc2626"

# ── Tizim prompt ──────────────────────────────────────────────────────────────
SYSTEM = """Sen JARVIS — foydalanuvchining shaxsiy O'zbek AI yordamchisissan.

MUHIM QOIDALAR:
- FAQAT O'ZBEK TILIDA gapir. Hech qachon boshqa tilda javob berma.
- Qisqa, aniq, do'stona va ishonchli gapir.
- "Siz" deb murojaat qil.
- Vazifa qo'shish so'ralsa: {"action":"vazifa","sarlavha":"...","muhimlik":"yuqori/o'rta/past"}
- Vazifalarni ko'rish so'ralsa: {"action":"vazifalar_korsatish"}

Misol javoblar:
  Foydalanuvchi: "Salom"
  JARVIS: "Salom! Men JARVIS, sizning yordamchingizman. Bugun qanday yordam kerak?"

  Foydalanuvchi: "Ertaga uchrashuv bor"
  JARVIS: "Tushundim! Uchrashuv vazifasini qo'shaymi? {"action":"vazifa","sarlavha":"Ertaga uchrashuv","muhimlik":"yuqori"}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Async ko'prik
# ═══════════════════════════════════════════════════════════════════════════════
class AsyncRunner:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._loop.run_forever, daemon=True)
        t.start()

    def run(self, coro, done=None):
        f = asyncio.run_coroutine_threadsafe(coro, self._loop)
        if done:
            f.add_done_callback(done)
        return f

    def stop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)


# ═══════════════════════════════════════════════════════════════════════════════
# Ovoz chiqarish — Edge TTS (Uzbek Sardor)
# ═══════════════════════════════════════════════════════════════════════════════
async def speak_uz(text: str):
    """Edge-TTS bilan o'zbek tilida ovoz chiqarish."""
    try:
        import edge_tts, tempfile, os, subprocess
        voice = "uz-UZ-SardorNeural"       # erkak ovoz
        # voice = "uz-UZ-MadinaNeural"     # ayol ovoz

        tts = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name

        await tts.save(path)

        # Platformaga qarab mp3 chalish
        if sys.platform == "darwin":
            subprocess.Popen(["afplay", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["mpg123", "-q", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[TTS xato]: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Xabar bubble
# ═══════════════════════════════════════════════════════════════════════════════
class Bubble(ctk.CTkFrame):
    def __init__(self, parent, role: str, text: str, **kw):
        bg = USER_BG if role == "user" else BOT_BG
        super().__init__(parent, fg_color=bg, corner_radius=16, **kw)

        # Sarlavha qatori
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=14, pady=(10, 0))

        if role == "user":
            icon, name, color = "👤", "Siz", ACCENT
        else:
            icon, name, color = "🤖", "JARVIS", GREEN

        ctk.CTkLabel(hdr, text=f"{icon} {name}",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=color).pack(side="left")

        ts = datetime.now().strftime("%H:%M")
        ctk.CTkLabel(hdr, text=ts,
                     font=ctk.CTkFont(size=10),
                     text_color=MUTED).pack(side="right")

        # Matn
        ctk.CTkLabel(self, text=text,
                     font=ctk.CTkFont(size=14),
                     text_color=TEXT,
                     wraplength=500,
                     justify="left",
                     anchor="w").pack(fill="x", padx=14, pady=(4, 12))


# ═══════════════════════════════════════════════════════════════════════════════
# Asosiy ilova
# ═══════════════════════════════════════════════════════════════════════════════
class JARVIS(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("JARVIS — O'zbek AI Yordamchi")
        self.geometry("1050x700")
        self.minsize(800, 550)
        self.configure(fg_color=BG)

        self._runner   = AsyncRunner()
        self._busy     = False
        self._mic_on   = False
        self._audio    = []
        self._stream   = None
        self._tasks: list[dict] = []
        self._history: list[dict] = []

        # UI ni kechiktirib quramiz — macOS bo'sh ekrandan qochish uchun
        self.after(50, self._build)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._sidebar()
        self._chat_panel()

        # Salomlashish
        self.after(200, lambda: self._bot_msg(
            "Assalomu alaykum! Men JARVIS — sizning shaxsiy AI yordamchingizman 🤖\n\n"
            "Boshlash uchun:\n"
            "① Chap tomonda OpenAI API kalitni kiriting va «Ulash» bosing\n"
            "② Yozing yoki 🎤 ni bosib gapiringm\n"
            "③ JARVIS o'zbekcha javob beradi va ovozda ham aytadi!"
        ))

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _sidebar(self):
        sb = ctk.CTkFrame(self, width=230, fg_color=SIDEBAR, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_rowconfigure(6, weight=1)

        # Logo
        ctk.CTkLabel(sb, text="⬡ JARVIS",
                     font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=ACCENT).grid(
            row=0, padx=18, pady=(24, 2), sticky="w")
        ctk.CTkLabel(sb, text="O'zbek AI Yordamchi",
                     font=ctk.CTkFont(size=11), text_color=MUTED).grid(
            row=1, padx=18, sticky="w")

        ctk.CTkFrame(sb, height=1, fg_color=BORDER).grid(
            row=2, padx=18, pady=14, sticky="ew")

        # API blok
        api_f = ctk.CTkFrame(sb, fg_color="transparent")
        api_f.grid(row=3, padx=18, sticky="ew")

        ctk.CTkLabel(api_f, text="OpenAI API Kalit",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=MUTED).pack(anchor="w", pady=(0, 5))

        self._key_entry = ctk.CTkEntry(
            api_f, placeholder_text="sk-...", show="*",
            height=34, fg_color=SURFACE, border_color=BORDER, text_color=TEXT,
            font=ctk.CTkFont(size=12))
        self._key_entry.pack(fill="x")

        env_k = os.environ.get("OPENAI_API_KEY", "")
        if env_k:
            self._key_entry.insert(0, env_k)

        ctk.CTkButton(api_f, text="Ulash ✓", height=32,
                      fg_color=ACCENT, hover_color=ACC_HV,
                      font=ctk.CTkFont(size=12, weight="bold"),
                      command=self._connect).pack(fill="x", pady=(8, 0))

        # Model
        ctk.CTkLabel(api_f, text="Model",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=MUTED).pack(anchor="w", pady=(14, 4))
        self._model = ctk.StringVar(value="gpt-4o")
        ctk.CTkOptionMenu(api_f,
                          values=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                          variable=self._model,
                          fg_color=SURFACE, button_color=ACCENT,
                          button_hover_color=ACC_HV,
                          dropdown_fg_color=SURFACE, text_color=TEXT,
                          font=ctk.CTkFont(size=12)).pack(fill="x")

        # Ovoz
        ctk.CTkLabel(api_f, text="JARVIS ovozi",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=MUTED).pack(anchor="w", pady=(14, 4))
        self._voice_on = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(api_f, text="Ovoz chiqarsin",
                      variable=self._voice_on,
                      oncolor=ACCENT, offcolor=SURFACE,
                      font=ctk.CTkFont(size=12),
                      text_color=TEXT).pack(anchor="w")

        ctk.CTkFrame(sb, height=1, fg_color=BORDER).grid(
            row=4, padx=18, pady=14, sticky="ew")

        # Vazifalar
        task_f = ctk.CTkFrame(sb, fg_color="transparent")
        task_f.grid(row=6, padx=18, sticky="nsew")
        task_f.grid_rowconfigure(1, weight=1)
        task_f.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(task_f, fg_color="transparent")
        top.grid(row=0, sticky="ew")

        ctk.CTkLabel(top, text="📋 Vazifalar",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=TEXT).pack(side="left")

        self._task_count = ctk.CTkLabel(top, text="0",
                                        font=ctk.CTkFont(size=11),
                                        text_color=MUTED)
        self._task_count.pack(side="right")

        self._task_list = ctk.CTkScrollableFrame(
            task_f, fg_color="transparent",
            scrollbar_button_color=BORDER, height=220)
        self._task_list.grid(row=1, sticky="nsew", pady=(8, 0))

        # Holat
        self._stat = ctk.CTkLabel(sb, text="● Kutmoqda",
                                  font=ctk.CTkFont(size=11),
                                  text_color=YELLOW)
        self._stat.grid(row=7, padx=18, pady=(10, 4), sticky="w")

        ctk.CTkButton(sb, text="🗑  Tozala", height=30,
                      fg_color=SURFACE, hover_color=BORDER,
                      text_color=TEXT, font=ctk.CTkFont(size=11),
                      command=self._clear).grid(
            row=8, padx=18, pady=(0, 18), sticky="ew")

    # ── Chat paneli ───────────────────────────────────────────────────────────
    def _chat_panel(self):
        pane = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        pane.grid(row=0, column=1, sticky="nsew")
        pane.grid_rowconfigure(0, weight=1)
        pane.grid_columnconfigure(0, weight=1)

        # Suhbat maydoni
        self._chat = ctk.CTkScrollableFrame(
            pane, fg_color=BG,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=MUTED)
        self._chat.grid(row=0, column=0, sticky="nsew", padx=0)
        self._chat.grid_columnconfigure(0, weight=1)

        # "O'ylamoqda" banner
        self._thinking = ctk.CTkLabel(
            pane, text="⏳  JARVIS javob tayyorlamoqda…",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=YELLOW, fg_color=SURFACE,
            corner_radius=8)

        # Input qutisi
        box = ctk.CTkFrame(pane, fg_color=SURFACE, corner_radius=18)
        box.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 4))
        box.grid_columnconfigure(0, weight=1)

        self._inp = ctk.CTkTextbox(
            box, height=54, wrap="word",
            fg_color="transparent", border_width=0,
            text_color=TEXT, font=ctk.CTkFont(size=14))
        self._inp.grid(row=0, column=0, sticky="ew", padx=14, pady=8)
        self._inp.bind("<Return>", self._enter)
        self._inp.bind("<Shift-Return>", lambda e: None)
        self._inp.focus()

        btns = ctk.CTkFrame(box, fg_color="transparent")
        btns.grid(row=0, column=1, padx=(0, 10))

        self._mic_btn = ctk.CTkButton(
            btns, text="🎤", width=46, height=40,
            fg_color=SURFACE, hover_color=BORDER,
            font=ctk.CTkFont(size=20),
            command=self._mic_toggle)
        self._mic_btn.pack(pady=4)

        self._send = ctk.CTkButton(
            btns, text="↑ Yuborish", width=110, height=40,
            fg_color=ACCENT, hover_color=ACC_HV,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._send_msg)
        self._send.pack(pady=4)

        ctk.CTkLabel(pane,
                     text="Enter — yuborish  ·  Shift+Enter — yangi qator  ·  🎤 — mikrofon",
                     font=ctk.CTkFont(size=10), text_color=MUTED).grid(
            row=3, column=0, pady=(0, 8))

    # ── Holat ─────────────────────────────────────────────────────────────────
    def _status(self, txt: str, color: str = MUTED):
        self._stat.configure(text=f"● {txt}", text_color=color)

    # ── API ulash ─────────────────────────────────────────────────────────────
    def _connect(self):
        k = self._key_entry.get().strip()
        if not k:
            return
        os.environ["OPENAI_API_KEY"] = k
        self._status("Ulandi ✓", GREEN)
        self._bot_msg("API kalit saqlandi ✅\nEndi yozing yoki 🎤 ni bosib gapiringm!")

    # ── Xabarlar ──────────────────────────────────────────────────────────────
    def _bot_msg(self, text: str):
        b = Bubble(self._chat, "assistant", text)
        b.grid(sticky="w", padx=(12, 60), pady=4,
               row=len(self._chat.winfo_children()), column=0)
        self.after(80, lambda: self._chat._parent_canvas.yview_moveto(1.0))

        # Ovoz chiqarish
        if self._voice_on.get() and len(text) < 400:
            clean = text.replace("🤖", "").replace("✅", "").replace("①②③", "")
            self._runner.run(speak_uz(clean))

    def _user_msg(self, text: str):
        b = Bubble(self._chat, "user", text)
        b.grid(sticky="e", padx=(60, 12), pady=4,
               row=len(self._chat.winfo_children()), column=0)
        self.after(80, lambda: self._chat._parent_canvas.yview_moveto(1.0))

    # ── Tozalash ──────────────────────────────────────────────────────────────
    def _clear(self):
        for w in self._chat.winfo_children():
            w.destroy()
        self._history.clear()
        self._bot_msg("Suhbat tozalandi. Yangi savol berish mumkin!")

    # ── Mikrofon ──────────────────────────────────────────────────────────────
    def _mic_toggle(self):
        if self._mic_on:
            self._mic_stop()
        else:
            self._mic_start()

    def _mic_start(self):
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            self._bot_msg("❌ Mikrofon uchun: pip install sounddevice scipy")
            return

        self._mic_on = True
        self._audio = []
        self._mic_btn.configure(text="⏹", fg_color=MIC_ON, hover_color=MIC_ON)
        self._status("🎤 Tinglayapman…", RED)

        def cb(data, frames, t, status):
            self._audio.append(data.copy())

        self._stream = sd.InputStream(
            samplerate=16000, channels=1, dtype="float32", callback=cb)
        self._stream.start()

    def _mic_stop(self):
        import numpy as np

        self._mic_on = False
        self._mic_btn.configure(text="🎤", fg_color=SURFACE, hover_color=BORDER)
        self._status("Ovoz tahlil qilinmoqda…", YELLOW)

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._audio:
            self._status("Tayyor", GREEN)
            return

        audio = np.concatenate(self._audio)
        if len(audio) < 4800:      # < 0.3 sekund — e'tiborsiz qoldirish
            self._status("Tayyor", GREEN)
            return

        self._runner.run(
            self._transcribe(audio),
            done=lambda f: self.after(0, lambda: self._after_transcribe(f))
        )

    async def _transcribe(self, audio) -> str:
        import numpy as np
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            return ""
        from openai import AsyncOpenAI

        audio_i16 = (audio * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
            wf.writeframes(audio_i16.tobytes())

        try:
            cl = AsyncOpenAI(api_key=key)
            with open(path, "rb") as fp:
                r = await cl.audio.transcriptions.create(
                    model="whisper-1", file=fp,
                    language="uz", response_format="text")
            return str(r).strip()
        finally:
            import os; os.unlink(path)

    def _after_transcribe(self, future):
        self._status("Tayyor", GREEN)
        try:
            text = future.result()
        except Exception as e:
            self._bot_msg(f"❌ Ovoz xatosi: {e}")
            return
        if text:
            self._inp.insert("end", text)
            self._send_msg()

    # ── Xabar yuborish ────────────────────────────────────────────────────────
    def _enter(self, e):
        if e.state & 0x1:
            return
        self._send_msg()
        return "break"

    def _send_msg(self):
        if self._busy:
            return
        text = self._inp.get("1.0", "end").strip()
        if not text:
            return
        self._inp.delete("1.0", "end")
        self._user_msg(text)

        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            self._bot_msg("❌ Iltimos, chap panelda API kalit kiriting va «Ulash» ni bosing.")
            return

        self._busy = True
        self._send.configure(state="disabled", text="…")
        self._thinking.grid(row=1, column=0, pady=6, padx=16, sticky="ew")
        self._status("O'ylamoqda…", YELLOW)

        self._runner.run(
            self._ai(text),
            done=lambda f: self.after(0, lambda: self._on_reply(f))
        )

    async def _ai(self, text: str) -> str:
        from openai import AsyncOpenAI
        key = os.environ.get("OPENAI_API_KEY", "")
        self._history.append({"role": "user", "content": text})
        msgs = [{"role": "system", "content": SYSTEM}] + self._history[-20:]

        try:
            cl = AsyncOpenAI(api_key=key)
            r = await cl.chat.completions.create(
                model=self._model.get(),
                messages=msgs,
                temperature=0.7,
                max_tokens=1024)
            reply = r.choices[0].message.content or ""
            self._history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"❌ Xato: {e}"

    def _on_reply(self, future):
        self._busy = False
        self._send.configure(state="normal", text="↑ Yuborish")
        self._thinking.grid_forget()
        self._status("Tayyor", GREEN)

        try:
            reply = future.result()
        except Exception as e:
            reply = f"❌ Kutilmagan xato: {e}"

        # Vazifa buyrug'i
        import json, re
        for m in re.finditer(r'\{[^{}]+\}', reply):
            try:
                cmd = json.loads(m.group())
                if cmd.get("action") == "vazifa":
                    self._add_task(cmd.get("sarlavha", "Nomsiz"),
                                   cmd.get("muhimlik", "o'rta"))
                    reply = reply[:m.start()].strip()
                elif cmd.get("action") == "vazifalar_korsatish":
                    active = [t for t in self._tasks if not t["bajarildi"]]
                    if active:
                        lst = "\n".join(f"• {t['sarlavha']} [{t['muhimlik']}]"
                                        for t in active)
                        reply = f"📋 Vazifalar ro'yxati:\n{lst}"
                    else:
                        reply = "📋 Hozircha hech qanday vazifa yo'q."
            except Exception:
                pass

        if reply:
            self._bot_msg(reply)

    # ── Vazifalar ─────────────────────────────────────────────────────────────
    def _add_task(self, sarlavha: str, muhimlik: str = "o'rta"):
        t = {"sarlavha": sarlavha, "muhimlik": muhimlik, "bajarildi": False}
        self._tasks.append(t)
        self._draw_task(t)
        active = sum(1 for x in self._tasks if not x["bajarildi"])
        self._task_count.configure(text=str(active))

    def _draw_task(self, t: dict):
        colors = {"yuqori": RED, "o'rta": YELLOW, "past": GREEN}
        dot = colors.get(t["muhimlik"], YELLOW)

        card = ctk.CTkFrame(self._task_list, fg_color=SURFACE, corner_radius=8)
        card.pack(fill="x", pady=3)

        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=6)

        ctk.CTkLabel(row, text="●", text_color=dot,
                     font=ctk.CTkFont(size=10)).pack(side="left", padx=(0, 6))

        ctk.CTkLabel(row, text=t["sarlavha"],
                     font=ctk.CTkFont(size=12), text_color=TEXT,
                     anchor="w").pack(side="left", fill="x", expand=True)

        def done(c=card, task=t):
            task["bajarildi"] = True
            c.destroy()
            active = sum(1 for x in self._tasks if not x["bajarildi"])
            self._task_count.configure(text=str(active))

        ctk.CTkButton(row, text="✓", width=28, height=22,
                      fg_color="transparent", hover_color=GREEN,
                      text_color=MUTED, font=ctk.CTkFont(size=12),
                      command=done).pack(side="right")

    # ── Yopish ────────────────────────────────────────────────────────────────
    def on_close(self):
        if self._mic_on:
            self._mic_stop()
        self._runner.stop()
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = JARVIS()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
