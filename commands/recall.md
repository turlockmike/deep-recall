---
description: Find the note that answers a question (deeprecall)
argument-hint: "<question>"
allowed-tools: Bash(deeprecall:*), Read
---

Run `deeprecall recall "$ARGUMENTS"` and answer the question from the winning section it prints.
Cite the file path. If the top score is low (the tool says "low confidence"), say the notes don't
clearly answer it and list the closest files instead of guessing.
