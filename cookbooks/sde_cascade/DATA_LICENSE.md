# Data licence

`example.json`, `calendar_answers.json`, `calendar_results.csv` and `calendar_cache.json` come from, or hold results on, the 1,000 calendar-scheduling tasks of [NATURAL PLAN](https://github.com/google-deepmind/natural-plan) (Copyright 2024 DeepMind Technologies Limited), whose data is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `example.json` is one of those tasks, unchanged. `calendar_sweep.py` reads the tasks from that repository at commit `ca76db33` and adds a JSON Schema for the answer to each task's zero-shot prompt. `calendar_answers.json` is the tasks' answer key, with every other valid meeting time added for the tasks that have more than one.

The other files this cookbook publishes to typesafe-public-examples are TypeSafe's work under that repository's MIT licence.
