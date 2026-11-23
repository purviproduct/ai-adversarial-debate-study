Date of the model calls: 2026-11-23
Model and settings: gpt-5-mini, temperature 1, default reasoning
Agents: 2 honest (A, A2), 1 adversarial (B). In the control arm all three are honest. No warning prompt.
Rounds: 10
Problems and goals: the pilot's 5 RealFP problems; control plus goals at the honest median ÷100, ÷10, ÷3, ×3, ×10, ×100
Debates: 5 problems × 7 arms × 2 trials = 70
Calibration file used: runs/2026-11-10-pilot/adversarial_debate_calibration_final.json
Script and command: src/adversarial_debate.py run --calib runs/2026-11-10-pilot/adversarial_debate_calibration_final.json --n 2 --rounds 10 --honest 2
Cost: $7.42, 2,100 model calls, 0 unparsed answers
Who: Purvi
What is different from the pilot: 10 rounds instead of 6, and a second honest agent. Each agent sees both other agents' previous replies. The adversary's instruction says "2 other AI agents" and "the other agents" (see prompts.md). comparison.txt is the output of src/compare_runs.py against runs/2026-11-23-one-honest-10-rounds.
Note: the run took 5,806 seconds because of a few connection resets, which the model-call layer retried.
