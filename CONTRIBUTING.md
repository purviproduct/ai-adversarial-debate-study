# Adding a run

This repo is public. Every run goes in its own dated folder, and every number in the README must come from a file in that folder.

## 1. Get the repo

```bash
git clone https://github.com/purviproduct/ai-adversarial-debate-study.git
```

```bash
cd ai-adversarial-debate-study && git checkout -b my-run-name
```

Work on a branch and open a pull request, so we can both read a run before its numbers reach the README.

## 2. Put the run in its own folder

Name it `runs/YYYY-MM-DD-short-name/`, using the date the model calls were made. For example: `runs/2026-11-19-warned-honest/` or `runs/2026-11-20-two-honest-one-adversary/`.

Put in it:

| File | What it is |
|---|---|
| Raw output | Every debate as it was saved: each agent's full reply in each round, the goal, the parsed answers. JSON or CSV, whatever your script writes |
| `RUN.md` | A few lines, from the template below |
| `results.txt` | The printed table your analysis produced, so the numbers can be checked against the raw output |
| `figures/` | Any charts |

Do not edit or delete files in another run's folder.

`RUN.md` template:

```
Date of the model calls:
Model and settings (model name, temperature, reasoning):
Agents (how many honest, how many adversarial, any extra prompt such as a warning):
Rounds:
Problems and goals:
Debates (problems × arms × trials):
Calibration file used:
Script and command:
Cost, if known:
What is different from the pilot:
```

## 3. Put the code in `src/`

If you ran `src/adversarial_debate.py` unchanged, say so in `RUN.md`. If you changed it or used your own script, add it as a new file in `src/` (for example `src/multi_agent_debate.py`) so the run can be repeated. Any new prompt text goes in `prompts.md`, word for word.

## 4. Update the README

Add one row for your run to the "Runs" table. Add findings to the README only when the run folder has the output that backs them. Say how many debates each number rests on, and say where an effect was small or did not hold.

## Before every push

- **No keys.** API keys stay in a `.env` file, which is gitignored. Check that no key is in a script, a notebook or a saved output. If a key was ever committed, tell the other person and rotate it; deleting the file is not enough.
- **No RealFP dataset.** The `data/` folder is gitignored because the dataset has no license. Quoting the few questions a run used is fine.
- **Watch file size.** The pilot's raw file is about 2 MB. GitHub rejects files over 100 MB. If a long run is larger than about 50 MB, compress it (`gzip`) and note that in `RUN.md`.
