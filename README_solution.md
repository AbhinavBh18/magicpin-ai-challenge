# magicpin AI Challenge Solution - Antigravity Vera

This is a complete solution for the magicpin AI Challenge, built using **FastAPI** and **Google's Gemini 1.5 Flash**.

## Features

- Fully implements the API contract (`/v1/healthz`, `/v1/metadata`, `/v1/context`, `/v1/tick`, `/v1/reply`).
- Uses `gemini-1.5-flash` for high-quality, highly specific, and customized message compositions matching category voice and merchant stats.
- Handles intent transitions properly by using the LLM to analyze conversation history and moving to action mode instead of qualifying further.
- Handles auto-replies using both heuristics (verbatim repeats) and LLM classification to gracefully end the conversation.

## Setup Instructions

1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Google API key for Gemini:
   - On Windows (Command Prompt): `set GOOGLE_API_KEY=your_api_key_here`
   - On Windows (PowerShell): `$env:GOOGLE_API_KEY="your_api_key_here"`
   - On macOS/Linux: `export GOOGLE_API_KEY=your_api_key_here`

3. Run the bot:
   ```bash
   python bot.py
   ```
   The bot will start on `http://localhost:8080`.

## Testing with Judge Simulator

With the bot running, open another terminal and run the judge simulator:

```bash
python judge_simulator.py
```

Ensure the configuration in `judge_simulator.py` is correctly pointed to `http://localhost:8080`. You might also need to configure your LLM provider in `judge_simulator.py` if the test suite requires it.
