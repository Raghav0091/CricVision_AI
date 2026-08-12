# Phone live testing (free, Option A)

Use your **phone camera** with CricVision while the **PC runs the AI**. No hosting cost.

## Before you start

1. Turn **off Norton VPN** on the PC.
2. Install cloudflared once: `winget install Cloudflare.cloudflared`

## Start (every session)

Double-click:

```text
scripts\start_phone_tunnel.bat
```

Three windows open — **do not close them**:

| Window | Role |
|--------|------|
| CricVision Backend | FastAPI + YOLO |
| CricVision Frontend | Next.js |
| CricVision Tunnel | HTTPS link for phone |

## On your phone

1. Copy the `https://....trycloudflare.com` link from the **Tunnel** window.
2. Open **`https://YOUR-LINK/live`** in Safari or Chrome.
3. Wait up to **60 seconds** on first load.
4. Allow **camera** access.

## What you can do on phone

| Page | Feature |
|------|---------|
| `/live` | Camera stump calibration |
| `/live` → Experimental Delivery Test | Motion → clip → ball detection |
| `/video-analysis` | Upload video from phone gallery |

## Troubleshooting

Run `scripts\check_phone_setup.bat` on the PC.

| Problem | Fix |
|---------|-----|
| Phone page never loads | PC must keep all 3 windows open; use a **new** tunnel link each session |
| PC `/live` fails | Read errors in Frontend window |
| Camera blocked | Must use **https://** tunnel link (not http) |
| Calibration fails | Backend running? Models in `Models/stump_detector/`? |

## Stop

Close all three windows or press Ctrl+C in each.
