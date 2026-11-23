Date of the model calls: 2026-11-23
Model and settings: gpt-5-mini, temperature 1, default reasoning
Agents: 1 honest (A), 1 adversarial (B). In the control arm both are honest.
Rounds: 10
Problems and goals: the pilot's 5 RealFP problems; control plus goals at the honest median ÷100, ÷10, ÷3, ×3, ×10, ×100
Debates: 5 problems × 7 arms × 2 trials = 70
Calibration file used: runs/2026-11-10-pilot/adversarial_debate_calibration_final.json
Script and command: src/adversarial_debate.py run --calib runs/2026-11-10-pilot/adversarial_debate_calibration_final.json --n 2 --rounds 10
Cost: $4.64, 1,400 model calls, 0 unparsed answers
Who: Purvi
What is different from the pilot: 10 rounds instead of 6. Everything else is the same.
