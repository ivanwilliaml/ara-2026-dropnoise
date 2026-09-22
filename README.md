# ARA 2026 — Segmentation Debugging

A prior rewrite of a working 0.82-dice segmentation recipe changed the model,
augmentation, resize strategy, and post-processing all at once — and the score dropped
to 0.74. This notebook rebuilds the recipe with discipline: one change at a time, each
one measured before it's allowed to ship.

## Open this first
- [`ara-dropnoise-final.ipynb`](./ara-dropnoise-final.ipynb) — the notebook, with every
  toggle (mislabeled-mask exclusion, a FocalLoss double-sigmoid bug fix) gated off by
  default and logged in its own experiment table.
- [`improvement-roadmap.txt`](./improvement-roadmap.txt) — the phased roadmap for what to
  try next, in dependency order.

## What it found
- A training mask (`train_298`) that's actually 93% one real object plus 51 tiny speckle
  components — a measurable noise signature.
- A loss-function bug: the model's `FocalLoss` applies sigmoid twice, compressing its
  effective input range to `[0.50, 0.73]` (~5x smaller loss magnitude) — verified
  numerically, not just suspected.

No dataset is committed here.
