# React + FastAPI Web Demo

Single-port deployment: FastAPI backend serves React static files AND API endpoints on port **7860**.

## Structure

```
react_app/
├── backend/
│   └── main.py          # FastAPI server — API routes + serve React build
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main app with tabs
│   │   ├── components/
│   │   │   ├── Task1Tab.jsx # Multi-Exposure Fusion UI
│   │   │   ├── Task2Tab.jsx # Inpainting & Editing UI
│   │   │   └── Task3Tab.jsx # Outpainting UI
│   │   ├── index.css        # Styles
│   │   └── main.jsx         # Entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── start.sh                 # Build frontend + start server
└── README.md
```

## Run Locally (without Docker)

```bash
# 1. Build React frontend
cd react_app/frontend
npm install
npm run build

# 2. Start server (from Final-Web-Demo root)
cd ../..
python -m uvicorn react_app.backend.main:app --host 0.0.0.0 --port 7860
```

## Run with Docker

```bash
# Using the updated Dockerfile (new CMD uses React+FastAPI)
docker compose up --build

# Or use the dedicated compose file
docker compose -f docker-compose.react.yml up --build
```

## Dev Mode (hot reload React)

```bash
# Terminal 1: backend
cd Final-Web-Demo
python -m uvicorn react_app.backend.main:app --host 0.0.0.0 --port 7860 --reload

# Terminal 2: Vite dev server (proxies /api to :7860)
cd Final-Web-Demo/react_app/frontend
npm run dev
```

## Switch back to Gradio

Edit `Dockerfile`, uncomment the old CMD and comment the new one:
```dockerfile
CMD ["python", "app.py"]
# CMD ["python", "-m", "uvicorn", "react_app.backend.main:app", ...]
```
