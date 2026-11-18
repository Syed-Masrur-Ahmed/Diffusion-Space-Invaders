**Result: the pipeline can memorize, but 500 steps are not enough for loss < 0.01.**
- The loss on fixed draws fell from 2.41 (the E2 value) to **0.031** (78× lower) and was still falling
  (0.048 → 0.033 → 0.031 over the last 100 steps). The sample PSNR against the 32 targets rose from 6.7 dB
  (noise) to **29.8 dB** and was still rising.
- Grid (last context | target | 2-step sample): the samples reproduce each window's layout, score digits, bullets
  and shields. Two signs it is not fully memorized yet: the samples are slightly dim with faint background
  speckle, and in row 5 the sample keeps an alien that the target has already lost (it leans on the context).
- No sign of a training-loop bug: the loss drops steadily from the expected initial value, nothing is stuck
  or NaN. A diffusion loss is noisier to drive to zero than a classifier's, since every step draws a new
  noise level; at small σ the model must predict the added noise exactly, which needs the target exactly.
- To meet the < 0.01 criterion: continue the same run for more steps (≈ 1,500 more on MPS, ~45 min).
