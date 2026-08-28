"""
tts_engine.py — Offline Text-to-Speech Engine (v2.0)

Uses pyttsx3 for instant, offline speech synthesis.

Why pyttsx3 over gTTS?
  - gTTS: requires internet, ~1-3 second HTTP delay, broken in no-wifi environments
  - pyttsx3: fully offline, <100ms latency, works in hospitals/schools/markets
"""

import pyttsx3
import threading
import logging

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Thread-safe, non-blocking offline Text-to-Speech engine.

    Usage:
        tts = TTSEngine()
        tts.speak("Hello")                     # speaks one word
        tts.speak_sentence(["Hello", "help"])  # speaks full sentence
    """

    def __init__(self, rate: int = 145, volume: float = 1.0):
        """
        Args:
            rate   : speech rate (words per minute). 145 is natural conversational speed.
            volume : volume 0.0–1.0
        """
        self._lock   = threading.Lock()
        self._busy   = False
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   rate)
            self._engine.setProperty("volume", volume)
            self._available = True
            logger.info("TTSEngine: pyttsx3 initialized (offline mode)")
        except Exception as e:
            self._available = False
            logger.warning(f"TTSEngine: pyttsx3 init failed — {e}. TTS disabled.")

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str):
        """
        Speak text asynchronously (non-blocking).
        If TTS is already speaking, the new request is skipped to avoid overlap.
        """
        if not self._available or not text.strip():
            return
        if self._busy:
            return  # skip — don't queue, avoid delay buildup
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def speak_sentence(self, words: list[str]):
        """
        Speak a list of words joined as a full sentence.

        Args:
            words: e.g. ["Hello", "I", "need", "help"]
                   → speaks "Hello I need help"
        """
        if not words:
            return
        self.speak(" ".join(words))

    def speak_word(self, word: str):
        """Alias for speak() — speaks a single detected sign word."""
        self.speak(word.replace("_", " ").title())

    def set_rate(self, rate: int):
        """Change speaking rate at runtime."""
        if self._available:
            with self._lock:
                self._engine.setProperty("rate", rate)

    def set_volume(self, volume: float):
        """Change volume at runtime (0.0–1.0)."""
        if self._available:
            with self._lock:
                self._engine.setProperty("volume", max(0.0, min(1.0, volume)))

    @property
    def is_available(self) -> bool:
        """Returns True if pyttsx3 initialized successfully."""
        return self._available

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self, text: str):
        with self._lock:
            self._busy = True
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                logger.warning(f"TTSEngine: speak failed — {e}")
            finally:
                self._busy = False


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    print("Testing offline TTS engine...")
    tts = TTSEngine(rate=145)

    if tts.is_available:
        print("Speaking: 'Hello, the offline voice is working.'")
        tts.speak("Hello, the offline voice is working.")
        time.sleep(4)

        print("Speaking sentence: ['I', 'need', 'help']")
        tts.speak_sentence(["I", "need", "help"])
        time.sleep(4)

        print("✓ TTS engine test complete — no internet was used.")
    else:
        print("✗ pyttsx3 not available. Install with: pip install pyttsx3")
