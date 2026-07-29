# CricVision Pro web frontend

This is the future mobile/tablet-first Next.js frontend. It currently provides the application shell, browser camera preview, stump-alignment guides, and safe calibration request flow. It is not yet a replacement for Streamlit.

Use a second Command Prompt window:

```bat
cd /d C:\CricVision_AI\apps\web
set NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```

Open `http://localhost:3000/video-analysis`. The frontend defaults to
`http://127.0.0.1:8000` and accepts `NEXT_PUBLIC_API_BASE_URL`
(`NEXT_PUBLIC_API_URL` remains a compatibility alias). WebSockets derive their
address from the same API URL unless `NEXT_PUBLIC_WS_URL` is explicitly set.
Stop the frontend with `Ctrl+C`.
