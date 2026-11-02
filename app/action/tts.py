# -*- coding: utf-8 -*-
import asyncio
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import config


class MockTTS:
    def speak(self, text, blocking=True):
        print(f"[TTS] {text}")

    def wait(self):
        pass

    def close(self):
        pass


class EdgeTTS:
    def __init__(self, voice=None):
        self.voice = voice or config.TTS_VOICE
        self.tmp_dir = Path(tempfile.gettempdir())
        self._threads = []

    def speak(self, text, blocking=False):
        if blocking:
            self._say(text, 0)
            return
        t = threading.Thread(target=self._say, args=(text, len(self._threads)), daemon=True)
        self._threads.append(t)
        t.start()

    def _say(self, text, idx):
        import edge_tts

        out = self.tmp_dir / f"produce_agent_tts_{idx}.mp3"

        async def _run():
            await edge_tts.Communicate(text, self.voice).save(str(out))

        asyncio.run(_run())
        _play(out)

    def wait(self):
        for t in list(self._threads):
            t.join()

    def close(self):
        self.wait()


class PiperTTS:
    def __init__(self, model=None):
        self.model = Path(model or config.PIPER_MODEL)
        self.tmp = Path(tempfile.gettempdir()) / "produce_agent_tts.wav"

    def speak(self, text, blocking=False):
        p = subprocess.run(
            ["piper", "--model", str(self.model), "--output_file", str(self.tmp)],
            input=text.encode("utf-8"),
            capture_output=True,
        )
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode("utf-8", "ignore"))
        if blocking:
            _play(self.tmp)
            return
        threading.Thread(target=_play, args=(self.tmp,), daemon=True).start()

    def wait(self):
        pass

    def close(self):
        pass


class Pyttsx3TTS:
    def __init__(self):
        import pyttsx3

        self.engine = pyttsx3.init()

    def speak(self, text, blocking=True):
        self.engine.say(text)
        self.engine.runAndWait()

    def wait(self):
        pass

    def close(self):
        pass


def _play(path):
    path = str(path)
    if sys.platform == "win32":
        if shutil.which(config.AUDIO_PLAYER_WIN):
            subprocess.run(
                [config.AUDIO_PLAYER_WIN, "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                check=False,
            )
            return
        if path.lower().endswith(".wav"):
            import winsound

            winsound.PlaySound(path, winsound.SND_FILENAME)
            return
        _play_win_media(path)
        return

    player = shutil.which(config.AUDIO_PLAYER) or shutil.which("mpv") or shutil.which("ffplay")
    if player:
        subprocess.run([player, "-nodisp", "-autoexit", path], capture_output=True, check=False)


def _play_win_media(path):
    """没有 ffplay 时用 WPF MediaPlayer 播 mp3，等播完再返回。"""
    script = (
        "Add-Type -AssemblyName PresentationCore;"
        f"$m = New-Object System.Windows.Media.MediaPlayer;"
        f"$m.Open([uri]'{path}');$m.Play();"
        "Start-Sleep -Milliseconds 400;"
        "while ($m.NaturalDuration.HasTimeSpan -eq $false) { Start-Sleep -Milliseconds 100 };"
        "$ms = $m.NaturalDuration.TimeSpan.TotalMilliseconds;"
        "Start-Sleep -Milliseconds ([int]$ms + 200);"
        "$m.Close()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
        capture_output=True,
        check=False,
    )


def build_tts():
    want = config.TTS_BACKEND
    order = {
        "auto": ["piper", "edge", "pyttsx3", "mock"],
        "edge": ["edge"],
        "piper": ["piper"],
        "pyttsx3": ["pyttsx3"],
        "mock": ["mock"],
    }.get(want, ["mock"])

    for name in order:
        try:
            if name == "piper" and Path(config.PIPER_MODEL).exists() and shutil.which("piper"):
                return PiperTTS()
            if name == "edge":
                import edge_tts  # noqa: F401

                return EdgeTTS()
            if name == "pyttsx3":
                return Pyttsx3TTS()
        except Exception:
            continue
    return MockTTS()
