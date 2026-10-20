# Frontend — HANA SQL Tuning Assistant

React 19 + Vite + TypeScript. One page: ask a question, read the answer, inspect the three knowledge-base
excerpts behind it, and see how the response time splits into retrieval, generation and browser/network.

```
npm install
npm run dev      # http://localhost:5173 — /api is proxied to the FastAPI backend on :8000
npm run build    # type-check + production build into dist/
```

Start the backend first (`uv run uvicorn backend.app:app --port 8000` from the project root).
