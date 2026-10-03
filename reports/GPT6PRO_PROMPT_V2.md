# Prompt for GPT-6-PRO, round 2: four questions while the LoRA run finishes

Style: ASD-STE100 Simplified Technical English. Short sentences. Active voice.
Present tense.

## 0. Context

Read the companion report `20261003_v10_exploration_report.md`. It gives the
full numbers. The short state is this:

- The score formula is F_v = 2 x IoU-sum / (N_pred + N_gt), macro over videos.
- The decision layer holds +0.116 binary temporal F1. Two harvest routes
  failed. The layer is closed for now.
- Fourteen ranking variants sit in one band: held-out AP 0.677 to 0.696.
  Pooling, structure, motion features, label protocol, and training breadth do
  not move it.
- One unconfirmed signal exists: LoRA r=8 on the vision tower. A smoke test at
  step 12 read 0.7067. We do not trust this point. The 900-step run decides.
- The platform scores are flat near 34 for eight rounds. Declared model size
  changes do not move the score.

## 1. A warning from the operator

The step-12 LoRA point is probably noise. The paired TCN has 12 steps of
training. The frozen baselines have 2500. One seed gives the point. The gain
mechanism is not understood. Treat the 0.7067 as unconfirmed. Design all
answers below so they do not depend on this point. Ask for the confirmation
protocol results first.

## 2. Question 1: close the data side

The operator holds all public data that fits the task. The set is: PHD2 (GIF
intervals, per-user selections, 13,939 videos), QVHighlights (cross-domain;
transfer failed), RetargetVid and LIVE (spatial GT), and 32B teacher points
(distillation is flat on the platform). More manual annotation is possible but
the operator does not want it: highlight choice is subjective, and one
annotator adds one more bias.

Questions:

- Data acquisition is at its end. Can label strategy still give gains on this
  fixed data? Give mechanisms, not hopes. The tests of OR-union,
  vote-fraction, and any-strict labels all landed in one AP band. What label
  transformations remain untested?
- The teacher point set (46,360 queries, 100 percent parse) is idle. Is there
  a use that is not distillation-into-GT and not a keep mask?
- Self-training: the best ranker can label the unused 8,000 PHD2 videos. The
  labels then come from a model at AP 0.69. Does noise of this size help or
  hurt a ranker that already trains on that model's own distribution? Give
  the decision rule.
- Give one cheap test that tells us if the data side is truly closed. The
  test must use only data on disk.

## 3. Question 2: the model hypothesis space at 500M

The present model is a frozen Siglip2 tower (86M) plus a 1.5M spatial head
plus a 0.2M temporal TCN. The platform limit is 500M parameters. The scores
sit near 34.

Questions:

- What is a realistic score ceiling for a model at or below 500M on this
  task, with these labels? Give the estimate as a number with reasoning, not
  as a range without a basis.
- Is the present blocking point the capacity, the architecture (an image
  encoder reads frames one at a time and knows no motion), or the labels? The
  ten-variant band says the representation is a candidate. Confirm or reject
  with a test design.
- A base-model swap is on the table. Candidates: a small video-native encoder
  (VideoMAE class, V-JEPA class, InternVideo small), or a stronger image
  tower. Give the trade study: expected AP gain, deployment cost, risk of the
  domain gap, Note: the k_size lever is dead. The size of the new tower does not
  change the coefficient.
- If we swap the base, the spatial head must retrain. Give the full migration
  order and the acceptance gates.

## 4. Question 3: training method, including RL

The operator asks: after the model is final, do we need reinforcement
learning, not only supervised gradient descent? The model is small (500M or
less), so RL runs fast.

Background facts for this question:

- The reward that matters is the real F per video. The formula is known and
  cheap to compute on training videos. The supervised loss (MSE on sigmoid
  outputs) optimizes a proxy.
- The decision layer failure (Section 4 of the report) shows that a better
  proxy fit does not give a better decision. RL on the true reward is one way
  around that.
- The platform labels are noisy (GIF user behavior). A reward on noisy labels
  invites reward hacking.

Questions:

- Is RL the right tool here? Compare, for this specific task: GRPO,
  reward-weighted regression, REINFORCE with baseline, and direct F-max
  optimization through a differentiable surrogate. Give the decision
  criteria.
- Give a full reward design: the reward function, the group baseline, the
  length normalization, the per-video normalization, and the guards against
  reward hacking on noisy labels. Be specific. A reward that says "use the
  F1" is not enough.
- The policy space: what does the policy control? Candidate actions: the keep
  decision per second, the score itself, the k selection. Recommend one and
  say why.
- Give the pipeline. The operator lists LoRA, partial freeze, and full
  fine-tune as options. Give the order, the switch conditions between
  stages, and the evaluation gate after each stage. Example shape: supervised
  warm start, then LoRA RL, then wider. Confirm or change this shape.

## 5. Question 4: metrics and the protocol against false conclusions

This is the most important question. The operator wants two things:

First: a short list of metrics that decide progress. Not ten metrics. The
list must say which metric is the promotion gate, which are diagnostics, and
which numbers are forbidden as evidence.

Second: a protocol that stops the coding agent from two failure modes. Mode
one: a false positive report (the leak incident: +0.075 that came from
reading the answer). Mode two: a false "this line is dead" report (an
untested hypothesis marked as closed, or three weak variants used to close a
whole class).

Questions:

- Give the promotion gate as a checklist. The gate must be computable from
  files on disk.
- Give the "line is dead" rule. How many variants, which controls, what CI,
  what documentation must exist before an agent may write "closed"?
- Give the audit list: the checks that catch train-on-eval, checkpoint
  mismatch, proxy metric confusion, aggregation mismatch, and seed count
  errors. Each check must be automatic where possible.
- Give the reporting template. It must make a false claim hard and a true
  claim easy to check.

## 6. Output format

Answer in this order: Q1, Q2, Q3, Q4. For each question give:

1. The direct answer, in five sentences or fewer.
2. The mechanism, with the score formula terms it moves.
3. The experiment design: data on disk, the split, the metric, the stop rule,
   the cost in hours.
4. The failure condition: the observation that closes this direction.

Keep the ASD-STE100 style. Short sentences. Active voice. No idioms. Mark
every estimate as an estimate.
