# CricVision Pro web frontend

This is the future mobile/tablet-first Next.js frontend. It currently provides the application shell, browser camera preview, stump-alignment guides, and safe calibration request flow. It is not yet a replacement for Streamlit.

```powershell
npm install
npm run dev
```

The frontend expects the API at `http://localhost:8000` by default. Override it with `NEXT_PUBLIC_API_URL` when required. Browser camera access requires `localhost` or HTTPS.
