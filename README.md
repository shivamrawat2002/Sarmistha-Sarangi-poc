# Sarmistha Sarangi POC

A small command-line proof of concept that reads a conversation transcript and asks an LLM to extract up to five actionable cards. The cards are validated and printed as JSON.

## Requirements

- Python 3.8 or later
- An API key for a supported hosted provider, or a local Ollama installation

The script uses only Python's standard library; no package installation is required.

## Setup

1. Copy `sample.env` to `.env`.
2. Set `PROVIDER` and `API_KEY` in `.env`. For example:

   ```env
   PROVIDER=gemini
   API_KEY=your_api_key
   ```

   Keep `.env` private; it is excluded from Git. Ollama runs locally and does not require an API key.

## Run

Process a transcript file:

```powershell
python prompt.py transcript.txt --date 2026-11-06
```

Pass transcript text directly:

```powershell
python prompt.py --text "Tomorrow I have a client meeting at 5 PM." --date 2026-11-12
```

For providers with smaller context limits, cap the number of transcript characters:

```powershell
python prompt.py transcript.txt --date 2026-11-06 --max-chars 15000
```

The date is the conversation date and is used to interpret relative dates such as “tomorrow.” The script prints the extracted cards and a validation result.

## Providers

Set `PROVIDER` in `.env` to one of:

| Provider | Configuration |
| --- | --- |
| Gemini | `PROVIDER=gemini`; set `API_KEY`. Defaults to `gemini-3.6-flash`. |
| Groq | `PROVIDER=groq`; set `API_KEY`. Defaults to `llama-3.3-70b-versatile`. |
| OpenRouter | `PROVIDER=openrouter`; set `API_KEY`. Defaults to `meta-llama/llama-3.3-70b-instruct:free`. |
| Ollama | `PROVIDER=ollama`; set `MODEL` if needed. Start Ollama and pull the model first (for example, `ollama pull llama3.1`). |

Set `MODEL` in `.env` to override a provider's default model.

## Output

The result is a JSON array with up to five cards. Each card contains `title`, `hashtag`, `body`, `related`, and `context`. Hashtags are limited to `task`, `research`, `follow_up`, `decision`, `planning`, `meeting`, and `review`.

## Self-test

Run the local validator checks without making an API call:

```powershell
python prompt.py --selftest
```
