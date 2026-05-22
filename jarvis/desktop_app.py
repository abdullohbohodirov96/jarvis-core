#!/usr/bin/env python3
"""
JARVIS Desktop App — Voice + Chat GUI (CustomTkinter).
Microphone -> OpenAI Whisper API -> GPT-4o -> macOS TTS
O'zbek tilida javob beradi.
"""
from __future__ import annotations
import asyncio, json, os, subprocess, sys, threading, time, tempfile, wave, io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_JARVIS_ROOT = Path(__file__).parent
if str(_JARVIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_JARVIS_ROOT))

import customtkinter as ctk
import numpy as np
import sounddevice as sd

# --- Internal imports ---
try:
    from config.settings import settings
    from core.logger import get_logger
    log = get_logger("desktop_app")
except ImportError:
    class _S:
        OPENAI_API_KEY=""; OPENAI_MODEL="gpt-4o"; OPENAI_MAX_TOKENS=2048
        OPENAI_TEMPERATURE=0.7; VERSION="1.0.0"; DESKTOP_AUTOMATION_ENABLED=True
    settings = _S()
    import logging; log = logging.getLogger("desktop_app"); logging.basicConfig(level=logging.INFO)

_AI_AVAILABLE = False
try:
    from ai.agent import JarvisAgent, get_agent
    _AI_AVAILABLE = True
except ImportError:
    get_agent = None

# ── O'zbek tilda JARVIS system prompt ──
JARVIS_UZ_PROMPT = """Sen JARVIS — foydalanuvchining shaxsiy desktop AI assistantisan.
Sen Tony Stark ning JARVIS assistantigadek gapirasansan: professional, aqlli, biroz hazilkash.

Hozirgi vaqt: {current_time}
Foydalanuvchi: Sir (Xo'jayin)

Sen quyidagi ishlarni bajara olasan:
1. OPEN_APP — ilova ochish (Chrome, VS Code, Terminal, va boshqalar)
2. OPEN_URL — brauzerda URL ochish
3. SEARCH_WEB — Google da qidirish
4. TAKE_SCREENSHOT — ekran suratga olish
5. CREATE_TASK — vazifa yaratish
6. TYPE_TEXT — matn yozish
7. SPEAK — faqat javob berish

Doimo O'ZBEK TILIDA javob ber.
Javobingni JSON formatida ber:
{{
  "speech": "Foydalanuvchiga aytadigan gaping (O'ZBEK TILIDA)",
  "action": {{
    "type": "ACTION_TYPE",
    "params": {{ ... }}
  }} yoki null agar hech qanday amal kerak bo'lmasa,
  "follow_up": null
}}

Qoidalar:
- speech har doim bo'sh bo'lmasin va O'ZBEK TILIDA bo'lsin.
- Qisqa va aniq javob ber — 1-2 gap yetarli.
- Agar amal kerak bo'lmasa action ni null qil.
"""


# ── Colors ──
class C:
    BG="#0a0e17"; CARD="#111827"; INPUT="#1a2235"; SIDEBAR="#0d1220"
    ACCENT="#3b82f6"; ACCENT_H="#2563eb"; OK="#10b981"; WARN="#f59e0b"; ERR="#ef4444"
    T1="#f1f5f9"; T2="#94a3b8"; T3="#64748b"; BORDER="#1e293b"
    UBUB="#1e3a5f"; ABUB="#1a2235"; SCROLL="#334155"
    MIC_REC="#ef4444"


# ── Async bridge ──
class AsyncBridge:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=lambda: (asyncio.set_event_loop(self._loop), self._loop.run_forever()), daemon=True).start()
    def submit(self, coro, cb=None):
        f = asyncio.run_coroutine_threadsafe(coro, self._loop)
        if cb: f.add_done_callback(lambda fut: cb(fut.result()))
        return f
    def stop(self): self._loop.call_soon_threadsafe(self._loop.stop)


# ── OpenAI helpers ──
async def _openai_chat(messages, api_key, model="gpt-4o", max_t=2048, temp=0.7):
    from openai import AsyncOpenAI
    c = AsyncOpenAI(api_key=api_key, timeout=60.0)
    r = await c.chat.completions.create(model=model, messages=messages, temperature=temp, max_tokens=max_t)
    return r.choices[0].message.content or ""

async def _openai_stt(audio_bytes, api_key):
    from openai import AsyncOpenAI
    c = AsyncOpenAI(api_key=api_key, timeout=30.0)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes); tmp.close()
    try:
        with open(tmp.name, "rb") as f:
            t = await c.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
        return t if isinstance(t, str) else str(t)
    finally:
        os.unlink(tmp.name)

def _macos_say(text):
    try:
        # Milena — best available voice for non-English
        subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════

class JarvisDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("J.A.R.V.I.S — Desktop AI Assistant")
        self.geometry("1100x750")
        self.minsize(800, 550)
        self.configure(fg_color=C.BG)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._async = AsyncBridge()
        self._messages = []
        self._is_processing = False
        self._is_recording = False
        self._audio_chunks = []
        self._rec_stream = None

        self._agent = None
        if _AI_AVAILABLE and get_agent:
            try: self._agent = get_agent()
            except: pass

        self._build_ui()
        self.after(600, self._show_welcome)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ═══════════════════════ UI ═══════════════════════

    def _build_ui(self):
        # --- HEADER ---
        hdr = ctk.CTkFrame(self, height=60, fg_color=C.CARD, corner_radius=0)
        hdr.pack(fill="x", side="top"); hdr.pack_propagate(False)

        lf = ctk.CTkFrame(hdr, fg_color="transparent"); lf.pack(side="left", padx=20, pady=10)
        self._dot = ctk.CTkLabel(lf, text="*", font=ctk.CTkFont(size=18, weight="bold"), text_color=C.OK, width=20)
        self._dot.pack(side="left", padx=(0,8))
        ctk.CTkLabel(lf, text="J.A.R.V.I.S", font=ctk.CTkFont(size=20, weight="bold"), text_color=C.T1).pack(side="left")
        ctk.CTkLabel(lf, text=f"  v{settings.VERSION}", font=ctk.CTkFont(size=12), text_color=C.T3).pack(side="left")

        rf = ctk.CTkFrame(hdr, fg_color="transparent"); rf.pack(side="right", padx=20, pady=10)
        ctk.CTkLabel(rf, text=f"Model: {settings.OPENAI_MODEL}", font=ctk.CTkFont(size=12), text_color=C.ACCENT).pack(side="right")

        ctk.CTkFrame(self, height=1, fg_color=C.BORDER, corner_radius=0).pack(fill="x")

        # --- MAIN ---
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True)
        main.grid_columnconfigure(0, weight=0)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR ---
        sb = ctk.CTkFrame(main, width=250, fg_color=C.SIDEBAR, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nswe"); sb.grid_propagate(False)

        ctk.CTkButton(sb, text="+ Yangi Chat", height=42, font=ctk.CTkFont(size=14, weight="bold"),
                       fg_color=C.ACCENT, hover_color=C.ACCENT_H, corner_radius=10,
                       command=self._new_chat).pack(fill="x", padx=16, pady=(20,10))

        ctk.CTkFrame(sb, height=1, fg_color=C.BORDER).pack(fill="x", padx=16, pady=8)

        ctk.CTkLabel(sb, text="IMKONIYATLAR", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=C.T3, anchor="w").pack(fill="x", padx=20, pady=(10,6))

        for txt in ["[MIC] Ovozli buyruq", "[CHAT] Matnli suhbat", "[APP] Ilova ochish",
                     "[WEB] Internet qidirish", "[TASK] Vazifa yaratish", "[IMG] Screenshot olish"]:
            ctk.CTkLabel(sb, text=f"  {txt}", font=ctk.CTkFont(size=13),
                         text_color=C.T2, anchor="w").pack(fill="x", padx=16, pady=1)

        ctk.CTkFrame(sb, fg_color="transparent").pack(fill="both", expand=True)
        ctk.CTkFrame(sb, height=1, fg_color=C.BORDER).pack(fill="x", padx=16, pady=4)

        status = "AI Agent: Tayyor" if self._agent else "Rejim: Direct"
        ctk.CTkLabel(sb, text=status, font=ctk.CTkFont(size=12),
                     text_color=C.OK if self._agent else C.WARN).pack(padx=16, pady=(6,4))
        ctk.CTkLabel(sb, text="Ovoz + Chat rejimi", font=ctk.CTkFont(size=11),
                     text_color=C.T3).pack(padx=16, pady=(0,16))

        # --- CHAT AREA ---
        chat = ctk.CTkFrame(main, fg_color=C.BG, corner_radius=0)
        chat.grid(row=0, column=1, sticky="nswe")
        chat.grid_rowconfigure(0, weight=1)
        chat.grid_rowconfigure(1, weight=0)
        chat.grid_columnconfigure(0, weight=1)

        self._scroll = ctk.CTkScrollableFrame(chat, fg_color=C.BG,
            scrollbar_button_color=C.SCROLL, scrollbar_button_hover_color=C.ACCENT)
        self._scroll.grid(row=0, column=0, sticky="nswe")
        self._scroll.grid_columnconfigure(0, weight=1)

        # --- INPUT AREA ---
        inp_frame = ctk.CTkFrame(chat, fg_color=C.CARD, height=110, corner_radius=0)
        inp_frame.grid(row=1, column=0, sticky="swe")
        inp_frame.grid_propagate(False)
        ctk.CTkFrame(inp_frame, height=1, fg_color=C.BORDER, corner_radius=0).pack(fill="x")

        # Mic button (big, obvious)
        mic_row = ctk.CTkFrame(inp_frame, fg_color="transparent")
        mic_row.pack(fill="x", padx=16, pady=(8,4))

        self._mic_btn = ctk.CTkButton(mic_row, text="[ MIC ] Bosing va gapiring",
                                       height=36, font=ctk.CTkFont(size=13, weight="bold"),
                                       fg_color=C.ACCENT, hover_color=C.ACCENT_H,
                                       corner_radius=8, command=self._toggle_mic)
        self._mic_btn.pack(side="left", padx=(0,10))

        self._mic_label = ctk.CTkLabel(mic_row, text="", font=ctk.CTkFont(size=12), text_color=C.ERR)
        self._mic_label.pack(side="left")

        # Text input row
        txt_row = ctk.CTkFrame(inp_frame, fg_color="transparent")
        txt_row.pack(fill="x", padx=16, pady=(0,8))
        txt_row.grid_columnconfigure(0, weight=1)
        txt_row.grid_rowconfigure(0, weight=0)

        self._input = ctk.CTkEntry(txt_row, height=38, fg_color=C.INPUT, text_color=C.T1,
                                    font=ctk.CTkFont(size=14), border_width=1,
                                    border_color=C.BORDER, corner_radius=10,
                                    placeholder_text="Xabar yozing...")
        self._input.grid(row=0, column=0, sticky="we", padx=(0,8))
        self._input.bind("<Return>", lambda e: self._send_text())

        self._send_btn = ctk.CTkButton(txt_row, text="Yuborish >", width=100, height=38,
                                        font=ctk.CTkFont(size=13, weight="bold"),
                                        fg_color=C.ACCENT, hover_color=C.ACCENT_H,
                                        corner_radius=10, command=self._send_text)
        self._send_btn.grid(row=0, column=1, sticky="e")

    # ═══════════════════════ MESSAGES ═══════════════════════

    def _bubble(self, role, text):
        is_u = role == "user"
        w = ctk.CTkFrame(self._scroll, fg_color="transparent")
        w.grid(sticky="e" if is_u else "w", padx=16, pady=4)

        label_text = "Siz" if is_u else "JARVIS"
        label_color = C.ACCENT if is_u else C.OK
        ctk.CTkLabel(w, text=label_text, font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=label_color, anchor="e" if is_u else "w").pack(fill="x", padx=8, pady=(4,0))

        bub = ctk.CTkFrame(w, fg_color=C.UBUB if is_u else C.ABUB,
                            corner_radius=14, border_width=1, border_color=C.BORDER)
        bub.pack(fill="x", padx=0, pady=(2,0))
        ctk.CTkLabel(bub, text=text, font=ctk.CTkFont(size=14), text_color=C.T1,
                     wraplength=520, justify="left", anchor="w").pack(padx=14, pady=10, fill="x")

        ctk.CTkLabel(w, text=datetime.now().strftime("%H:%M"), font=ctk.CTkFont(size=10),
                     text_color=C.T3, anchor="e" if is_u else "w").pack(fill="x", padx=8, pady=(0,2))

        self.after(50, lambda: self._scroll._parent_canvas.yview_moveto(1.0))

    def _typing_on(self):
        self._tf = ctk.CTkFrame(self._scroll, fg_color="transparent")
        self._tf.grid(sticky="w", padx=16, pady=4)
        ctk.CTkLabel(self._tf, text="JARVIS", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=C.OK, anchor="w").pack(fill="x", padx=8)
        ind = ctk.CTkFrame(self._tf, fg_color=C.ABUB, corner_radius=14,
                            border_width=1, border_color=C.BORDER)
        ind.pack(padx=0, pady=(2,0))
        ctk.CTkLabel(ind, text="... o'ylayapman ...", font=ctk.CTkFont(size=13),
                     text_color=C.ACCENT).pack(padx=16, pady=8)
        self.after(50, lambda: self._scroll._parent_canvas.yview_moveto(1.0))

    def _typing_off(self):
        if hasattr(self, "_tf") and self._tf.winfo_exists(): self._tf.destroy()

    # ═══════════════════════ VOICE ═══════════════════════

    def _toggle_mic(self):
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self._is_processing: return
        self._is_recording = True
        self._mic_btn.configure(fg_color=C.MIC_REC, text="[ STOP ] To'xtating")
        self._mic_label.configure(text="Yozib olinmoqda... gapiring!")
        self._audio_chunks = []

        try:
            self._rec_stream = sd.InputStream(
                samplerate=16000, channels=1, dtype='int16',
                callback=self._audio_cb, blocksize=1024
            )
            self._rec_stream.start()
            log.info("Microphone recording started")
        except Exception as e:
            log.error("Microphone error: {}", e)
            self._mic_label.configure(text=f"Mikrofon xatosi: {e}")
            self._is_recording = False
            self._mic_btn.configure(fg_color=C.ACCENT, text="[ MIC ] Bosing va gapiring")

    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            log.warning("Audio status: {}", status)
        if self._is_recording:
            self._audio_chunks.append(indata.copy())

    def _stop_recording(self):
        self._is_recording = False
        self._mic_btn.configure(fg_color=C.ACCENT, text="[ MIC ] Bosing va gapiring")
        self._mic_label.configure(text="Tanib olinmoqda...")

        try:
            if self._rec_stream:
                self._rec_stream.stop()
                self._rec_stream.close()
                self._rec_stream = None
        except Exception as e:
            log.warning("Stream close error: {}", e)

        if not self._audio_chunks:
            self._mic_label.configure(text="Ovoz topilmadi")
            self.after(2000, lambda: self._mic_label.configure(text=""))
            return

        # Build WAV
        audio = np.concatenate(self._audio_chunks, axis=0)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio.tobytes())
        wav_bytes = buf.getvalue()
        log.info("Recorded {} bytes of audio", len(wav_bytes))

        # Transcribe
        self._async.submit(
            self._transcribe(wav_bytes),
            cb=lambda r: self.after(0, lambda: self._on_transcribed(r))
        )

    async def _transcribe(self, wav_bytes):
        api_key = settings.OPENAI_API_KEY
        if not api_key or "your" in api_key:
            return {"error": "API kalit sozlanmagan"}
        try:
            text = await _openai_stt(wav_bytes, api_key)
            text = text.strip()
            if not text:
                return {"error": "Ovoz aniqlanmadi"}
            return {"text": text}
        except Exception as e:
            return {"error": str(e)}

    def _on_transcribed(self, result):
        if "error" in result:
            self._mic_label.configure(text=f"Xato: {result['error']}")
            self.after(3000, lambda: self._mic_label.configure(text=""))
            return

        self._mic_label.configure(text="")
        text = result["text"]
        self._messages.append({"role": "user", "content": text})
        self._bubble("user", f"[Ovoz] {text}")
        self._process_ai(text, speak=True)

    # ═══════════════════════ TEXT INPUT ═══════════════════════

    def _send_text(self):
        raw = self._input.get().strip()
        if not raw or self._is_processing: return
        self._input.delete(0, "end")
        self._messages.append({"role": "user", "content": raw})
        self._bubble("user", raw)
        self._process_ai(raw, speak=False)

    # ═══════════════════════ AI ═══════════════════════

    def _process_ai(self, text, speak=False):
        self._is_processing = True
        self._send_btn.configure(state="disabled", text="...")
        self._dot.configure(text_color=C.WARN)
        self._typing_on()
        self._async.submit(
            self._call_ai(text),
            cb=lambda r: self.after(0, lambda: self._show_reply(r, speak))
        )

    async def _call_ai(self, user_text):
        try:
            api_key = settings.OPENAI_API_KEY
            if not api_key or "your" in api_key:
                return "API kalit sozlanmagan. .env faylga OPENAI_API_KEY yozing."

            ct = datetime.now(timezone.utc).strftime("%A, %B %d %Y at %H:%M UTC")
            system_prompt = JARVIS_UZ_PROMPT.format(current_time=ct)

            msgs = [{"role": "system", "content": system_prompt}]
            for m in self._messages[-20:]:
                msgs.append({"role": m["role"], "content": m["content"]})

            raw = await _openai_chat(msgs, api_key, settings.OPENAI_MODEL,
                                      settings.OPENAI_MAX_TOKENS, settings.OPENAI_TEMPERATURE)

            # Parse JSON response
            try:
                p = json.loads(raw)
                speech = p.get("speech", raw)

                # Execute action if present
                action = p.get("action")
                if action and isinstance(action, dict):
                    atype = action.get("type", "")
                    params = action.get("params", {})
                    if atype:
                        await self._execute_action(atype, params)
                        speech += f"\n\n[Amal: {atype}]"
                return speech
            except (json.JSONDecodeError, TypeError):
                return raw

        except Exception as e:
            return f"Xatolik: {e}"

    async def _execute_action(self, action_type, params):
        """Actually execute the desktop action."""
        import subprocess as sp
        at = action_type.upper()

        if at == "OPEN_APP":
            app = params.get("app", params.get("name", ""))
            if app:
                try:
                    sp.Popen(["open", "-a", app], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
                except: pass

        elif at == "OPEN_URL":
            url = params.get("url", "")
            if url:
                try:
                    sp.Popen(["open", url], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
                except: pass

        elif at == "SEARCH_WEB":
            query = params.get("query", "")
            if query:
                import urllib.parse
                url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
                try:
                    sp.Popen(["open", url], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
                except: pass

        elif at == "TAKE_SCREENSHOT":
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"/tmp/jarvis_screenshot_{ts}.png"
                sp.run(["screencapture", path], check=True)
            except: pass

        elif at == "TYPE_TEXT":
            text = params.get("text", "")
            if text:
                try:
                    sp.run(["osascript", "-e", f'tell application "System Events" to keystroke "{text}"'])
                except: pass

    def _show_reply(self, text, speak):
        self._typing_off()
        self._messages.append({"role": "assistant", "content": text})
        self._bubble("assistant", text)

        self._is_processing = False
        self._send_btn.configure(state="normal", text="Yuborish >")
        self._dot.configure(text_color=C.OK)

        # Speak the response
        if speak:
            clean = text.split("\n\n[Amal")[0].strip()
            if clean:
                threading.Thread(target=_macos_say, args=(clean,), daemon=True).start()

    # ═══════════════════════ MISC ═══════════════════════

    def _new_chat(self):
        self._messages.clear()
        for w in self._scroll.winfo_children(): w.destroy()
        self._bubble("assistant", "Yangi suhbat boshlandi. Sizga qanday yordam bera olaman?")

    def _show_welcome(self):
        self._bubble("assistant",
            "Assalomu alaykum, xo'jayin! Men JARVIS — sizning shaxsiy AI assistantingizman.\n\n"
            "Menga ovoz bilan buyruq bering: [MIC] tugmasini bosing va gapiring.\n"
            "Yoki pastdagi matn maydoniga yozing.\n\n"
            "Masalan: 'Chrome ochib ber', 'Google dan ob-havo qidir', 'Screenshot ol'")
        _macos_say("Assalomu alaykum xo'jayin. Men Jarvis. Sizga qanday yordam bera olaman?")

    def _on_close(self):
        if self._is_recording and self._rec_stream:
            try: self._rec_stream.stop(); self._rec_stream.close()
            except: pass
        self._async.stop()
        self.destroy()


if __name__ == "__main__":
    app = JarvisDesktopApp()
    app.mainloop()
