# Wikipedia Semantic Search Engine 🧠🔎

A dynamic, web-based semantic search application that fetches Wikipedia articles and analyzes their conceptual meaning rather than relying on basic keyword matching.

By utilizing Google Gemini's embedding models and vector space math (cosine similarity), this engine understands the actual intent and context behind your queries — allowing you to discover relevant information even if the specific search terms don't explicitly appear in the text.

## 🚀 Features

- **True Semantic Vector Search**: Converts natural language queries into high-dimensional vectors to perform semantic proximity analysis via cosine similarity math.
- **Powered by Google Gemini**: Integrates the official `google-genai` SDK using `gemini-embedding-001` for language embedding generation.
- **Live Article Ingestion**: Fetches a fresh text corpus directly from Wikipedia's native API for every query, so results stay relevant to what you actually searched.
- **Wikitext Cleaning**: Uses `mwparserfromhell` to strip raw MediaWiki markup (templates, file embeds, wikilinks) from article text before embedding, so both the vectors and the displayed summaries are clean prose rather than raw markup.
- **Robust API Compliance**: Implements a custom `User-Agent` header to comply with Wikipedia API etiquette and reduce 429 throttling errors.
- **Interactive Web UI**: Built with a Gradio interface featuring processing step-trackers, variable result sliders, and match confidence scores.

## 🛠️ Tech Stack

- **UI Framework**: Gradio
- **Embedding API**: Google GenAI SDK (`gemini-embedding-001`)
- **Vector Math**: Scikit-learn (`cosine_similarity`), NumPy
- **Data Pipelines**: Pandas
- **Scrapers & Parsers**: Wikipedia API, `mwclient`, `mwparserfromhell`

## 📦 Installation & Setup

1. **Clone the Project & Navigate**
```bash
   git clone https://github.com/akshay-820/Wikipedia-Semantic-Search.git
   cd Wikipedia-Semantic-Search
```

2. **Configure Your Environment**

   Create a virtual environment to isolate dependencies cleanly, activate it, and install all libraries:
```bash
   # Setup Environment
   python3 -m venv venv
   source venv/bin/activate

   # Install Libraries
   pip install -r requirements.txt
```

3. **Setup Your API Key**

   Generate a developer API key inside [Google AI Studio](https://aistudio.google.com/). Create a `.env` file in the root folder of the project directory and paste your token inside:
```bash
   GEMINI_API_KEY=your_actual_gemini_api_key_here
```

## 🚀 Running the Application

Launch the backend listener and browser portal via the terminal command:
```bash
python app.py
```

Upon launching, the local server initialization logs will report:
Running on local URL:  `http://127.0.0.1:7860`

Open `http://127.0.0.1:7860` in your web browser to interact with the system interface!

## 💡 How It Works Under the Hood

1. **Live Query Search**: On every query, the app searches Wikipedia's live API for the most relevant article titles, then fetches their full text using a custom `User-Agent` header for compliance.
2. **Wikitext Cleaning**: Each article's raw content is parsed with `mwparserfromhell` to strip templates, file embeds, and link markup, leaving clean readable text.
3. **Gemini Vector Generation**: The cleaned article text is passed through Google's `gemini-embedding-001` model to generate high-dimensional embedding vectors.
4. **Similarity Proximity Ranking**: Your query is embedded the same way, and cosine similarity is calculated against all article vectors for that search, returning the closest contextual matches ranked by similarity score.

> **Note**: Because the corpus is rebuilt from live Wikipedia results on every search, there's no persistent local caching — each query re-fetches and re-embeds articles rather than reusing a prior index. This favors freshness over speed.
