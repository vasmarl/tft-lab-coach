# TFT Lab Coach

Live AI voice coaching on **recorded** Teamfight Tactics videos (Windows, single `TFTCoach.exe`).

## For the professor: how to run it

1. **Double-click `TFTCoach.exe`.** No Python, no install.
2. **First time only: connect a Google AI key.** Click **Get a free key**, sign in, click *Create API key*, copy it, paste it into the app and press **Connect**.
3. **Pick an option:**
   - **1 · Coach a video file**: choose an `.mp4/.mkv/.mov/.avi`; it plays in a window with the coach talking over it.
   - **2 · Coach a video on my screen**: pick the window that plays the video (for example YouTube in Chrome or VLC). A small caption panel appears bottom-right.

> If Windows SmartScreen says *"Windows protected your PC"* (the exe isn't code-signed): click **More info → Run anyway**.

**Use headphones.** The coach talks out loud, and your mic should only hear you.

## Controls

| | Video file window | Screen mode |
|---|---|---|
| Ask the coach a question | `M` to turn the mic on, speak, then `M` again | **Mic** button or `Ctrl+Shift+M` |
| Pause / resume | `Space` | **Pause** button or `Ctrl+Shift+P` |
| Stop | `Q` | **Stop** button or `Ctrl+Shift+Q` |

When the mic is on, the coach stops talking and listens. Nothing is sent from the mic while it is off.
The **Talk every (sec)** slider (4–20 s, default 8) sets how often the coach comments.

## Cost

- Live coaching (Gemini Live, about 1 frame/sec plus audio): **roughly $1–3 per full game** on a paid key. The free tier is enough to try it.
- Review mode (`TFTCoach.exe --review game.mp4`, gemini-2.5-flash, 1 frame every 5 s): **about $0.10 per game**.

## Where files are saved

Everything goes in the **same folder as `TFTCoach.exe`**:

- `.env`: your Google AI key (use *Change key* on the main screen to replace it)
- `coach_live_log_<date>.md` / `coach_screen_log_<date>.md`: transcripts (your questions + the coach's answers, frames streamed, session length)
- `coach_log_<date>.md`: review-mode tips + end-of-game review
- `tft_coach_output.log` (app output), `tft_coach_error.log` (crash details)

**Videos only:** it never coaches live matches. Screen mode hides League/Riot windows and refuses to run while a League of Legends game is open.

## Command line (optional)

```
TFTCoach.exe                      launcher
TFTCoach.exe --file game.mp4      live coaching on a file
TFTCoach.exe --screen             live coaching on a window
TFTCoach.exe --review game.mp4    cheaper round-by-round review
TFTCoach.exe --selftest           check that everything loads
  --every 8                       talk interval in seconds
  --model gemini-3.8-live         live model (or TFT_LIVE_MODEL=... in .env)
```

## Developers

- Python 3.11, tkinter, OpenCV, sounddevice, mss, pywin32, google-genai (Gemini Live API).
- `pip install -r requirements.txt pytest && python -m pytest tests` (Windows APIs are mocked, so tests also run on macOS).
- The exe is built by GitHub Actions on `windows-latest` (`.github/workflows/build.yml`). Download it from the run's **TFTCoach** artifact.
