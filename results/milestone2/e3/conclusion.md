**Result: passed. The pipeline memorizes one batch; the loss on fixed draws goes below 0.01.**
- 2,000 steps (61 min on Apple M4, MPS). The loss on fixed draws fell from 2.41 (the E2 value) to **0.0057**
  (~420× lower). It first dipped below 0.01 at step 1,200 and stayed below it at every log point from step 1,800
  (0.0100, 0.0066, 0.0054, 0.0067, 0.0057). The sample PSNR against the 32 targets rose from 6.7 dB (noise) to
  **36.5 dB** (29.5 dB at step 500).
- Grid (last context | target | 2-step sample): the samples reproduce each window's layout, score digits, bullets,
  shields and player position. The alien the 500-step model kept in row 5 (an alien the target has already lost) is
  now only a faint smudge, and a faint background speckle remains.
- No sign of a training-loop bug: the loss drops steadily from the expected initial value, nothing is stuck
  or NaN. A diffusion loss is noisier to drive to zero than a classifier's, since every step draws a new
  noise level (single-draw training loss still jumps, e.g. 0.015 at step 1,800); at small σ the model must predict
  the added noise exactly, which needs the target exactly.
