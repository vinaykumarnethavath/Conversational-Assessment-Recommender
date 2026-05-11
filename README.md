# SHL Conversational Assessment Recommender

This project is a conversational AI backend that helps recruiters pick SHL-style assessments based on hiring needs.

## Features

- `GET /health` returns `{"status":"ok"}`
- `POST /chat` supports:
  - follow-up clarification questions
  - recommendation generation (1-10 assessments)
  - recommendation refinement across message history
  - assessment comparison requests (for example, `difference between OPQ and GAT`)
- Retrieval is grounded in local catalog data (`data/assessments.json`)

## Project Structure

- `app.py` - FastAPI entrypoint and chat endpoint
- `rag.py` - requirement extraction, retrieval, comparison logic
- `embeddings.py` - sentence-transformers wrapper with safe fallback
- `models.py` - strict request/response Pydantic schemas
- `scraper.py` - SHL catalog scraper starter
- `data/assessments.json` - seed catalog used by recommender

## Using Your Full SHL Export

Place your exported SHL JSON file in one of these paths:

- `data/shl_catalog.json` (recommended)
- `data/assessments.json`
- `data/assessments_scraped.json`

The backend auto-normalizes your raw export fields like:
`entity_id`, `name`, `link`, `job_levels`, `languages`, `duration`,
`description`, `keys`, `remote`, `adaptive`.

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload
```

## Production Deploy Notes

- The repo includes a `Dockerfile`, `Procfile`, and `runtime.txt` to make deployment easier on platforms like Render/Railway/Fly/Heroku-style runners.
- After deployment, make sure these URLs work:
  - `GET /health` should return `{ "status": "ok" }` with HTTP 200
  - `POST /chat` should accept the full stateless conversation history and return a schema-compliant response.
- Optional: set `EMBEDDINGS_ENABLED=1` to enable sentence-transformers embeddings. If not set, the service uses a deterministic offline fallback.

## Deploy on Render (Step-by-step)

1) Push this repository to GitHub

2) Create a new Render Web Service
 
    - In Render: `New` -> `Web Service`
    - Connect your GitHub repo

    Note: Render does not use your local `.env` file. Add secrets (like `GROQ_API_KEY`) in Render's Environment settings.

3) Choose a deployment method

    Option A (recommended): Docker
    - Render will detect the `Dockerfile` automatically.

    Option B: Native Python
    - **Build command**: `pip install -r requirements.txt`
    - **Start command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

4) Configure environment variables (Render -> Environment)

    - `GROQ_API_KEY`: your Groq API key
    - `EMBEDDINGS_ENABLED`: `0` (recommended for Render free tier / 512Mi to avoid out-of-memory during model load) or `1` (enables sentence-transformers; requires more memory)
    - (Optional) `PORT`: Render provides this automatically

5) Deploy

    If your deploy fails with out-of-memory on the free tier, keep `EMBEDDINGS_ENABLED=0` or upgrade the Render instance size.

6) Verify

    After the deploy finishes, Render will give you a public base URL like:
    `https://conversational-assessment-recommender-1.onrender.com

    Test these URLs:
    - `GET https://conversational-assessment-recommender-1.onrender.com/health`
    - Swagger UI: `https://conversational-assessment-recommender-1.onrender.com/docs`
    - `POST https://conversational-assessment-recommender-1.onrender.com/chat`

    Example `POST /chat` payload:
    ```json
    {
      "messages": [
        {"role": "user", "content": "Hiring a mid-level Java developer. Need coding and problem solving."}
      ]
    }
    ```

## API Contract

### GET `/health`

Response:

```json
{
  "status": "ok"
}
```

### POST `/chat`

Request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a Java developer"
    }
  ]
}
```

Response (clarification):

```json
{
  "reply": "What experience level are you hiring for (entry, mid, or senior)?",
  "recommendations": [],
  "end_of_conversation": false
}
```

Response (recommendation):

```json
{
  "reply": "Here are the most relevant SHL assessments based on your requirements.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": true
}
```

## Notes

- The chatbot is stateless at API level and uses full `messages` history to keep conversational context.
- To use the entire real catalog, run `python scraper.py` and enrich the scraped entries (skills, category, duration, description) before indexing.
