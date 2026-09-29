---
description: Set up deeprecall for a notes folder (config + first index)
argument-hint: "[notes folder]"
allowed-tools: Bash(deeprecall:*)
---

Set up deeprecall for the user's notes.

1. Notes folder: use "$ARGUMENTS" if given, otherwise ask the user which folder holds their Markdown notes.
2. Ask whether they want the free local reranker (default, private) or a hosted one (`jev` needs
   `TYPESAFE_API_KEY`; `openai` works with any OpenAI-compatible endpoint that returns logprobs).
3. Run `deeprecall init --root "<folder>" [--backend <name>]`, then `deeprecall index`.
4. Run `deeprecall status` and report: config path, number of notes indexed, reranker, and caps.
5. Try one real lookup with `deeprecall recall "<a question about something in their notes>?"`.
