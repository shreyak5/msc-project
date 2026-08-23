# Occlusion handling training strategy

Companion to `implementation-plan.md` Sec 7, which lists "synthetic occlusion
training" under Pass C's original "out of scope for now" list.

## The problem

Pass C trains the TemporalTransformer (TT) to refine each frame's per-frame
encoding using its neighbors in time. Left alone, TT only ever practices this
on whatever real occlusion happens to occur in the training data - which is
sparse and uncontrolled. But at inference time, TT needs to reliably handle a
frame where the face is genuinely occluded (a hand passing in front of the
mouth, motion blur, a sign-language hand covering part of the face) by
leaning on nearby good frames instead. Without deliberate practice at this,
there's no guarantee TT actually learns to do it well.

## The idea

During Pass C, artificially pretend that some real, fully-visible frames in a
clip are occluded - but only for what TT is allowed to see. The frame's real,
true appearance is still known (it wasn't actually occluded), so the model's
prediction for that frame can be checked against genuine ground truth. This
turns "recover a frame from its temporal context" into a supervised task with
a reliable, dense training signal, rather than relying on incidental real
occlusion to teach the same skill.

Concretely, for a chosen frame:
- Its visual input to TT is corrupted - replaced with an average of its
  nearest good neighboring frames, and its visibility score is zeroed - so TT
  has to reconstruct that frame's parameters mostly from temporal context
  rather than its own (hidden) appearance.
- Everything used to compute the loss - the real photo, landmarks, mesh
  targets - stays untouched. The model is still graded against what actually
  happened in that frame.
- The frame's contribution to the loss is weighted higher than a normal
  frame's, so this "fill in the blank" signal isn't diluted by the rest of
  the clip.

Frames are chosen independently at random, scattered through the clip rather
than in clusters - each occluded frame is meant to look like an isolated gap
surrounded by good context, which is the case the averaging-based corruption
is designed for.

## Why only synthetic occlusion, not real partial occlusion

There's a second, superficially similar case: a frame that's genuinely
detected but partially occluded in reality (e.g. visibility ratio 0.3,
already low). It's tempting to treat this "the same way" and upweight it too
- but its real pixels show the occluder, not the true face. Supervising a
reconstruction loss against that image at any raised weight risks teaching
the model to faithfully reproduce the occlusion rather than see past it,
which is the opposite of the intended effect. Synthetic occlusion avoids this
entirely because the ground truth is known and clean by construction - the
frame wasn't actually occluded, only pretended-occluded for TT's input. Real
partial occlusion is a genuinely different, harder problem and is
deliberately left untouched by this strategy for now.

Frames that are real and totally undetected (no face found at all) are a
third case, and don't need special handling here either - they already
contribute nothing to reconstruction losses today, since there's no valid
target to supervise against in the first place.

## Toggling and tuning

The whole strategy is a training-time switch, off by default. Turning it on
exposes two more knobs: how often a frame gets pretended-occluded, and how
much extra weight an occluded frame's loss gets relative to a normal frame.
Both are starting guesses meant to be tuned empirically once real training
curves are available, same as every other loss weight in this project.
