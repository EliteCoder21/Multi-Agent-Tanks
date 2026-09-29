# How other multi-agent systems plan at scale — and what this project borrows

A short survey of the approaches that have produced coordinated, board-wide play in
multi-agent reinforcement learning, and where each one shows up (or deliberately
doesn't) in this codebase.

## 1. Centralised training, decentralised execution (CTDE)

The most common frame. Each unit acts on what it can see, but during training the
learner may use everything — the whole state, every unit's observation — to judge
actions, because that information is only needed for the *gradient*, not for play.

- **MAPPO** (Yu et al., 2021) — plain PPO with one change: the **value function sees
  the global state** while the policy sees only local observations. On StarCraft
  micromanagement, Hanabi and Google Football it matched or beat purpose-built
  methods. The lesson: most of the credit-assignment problem in a team game is the
  critic not knowing what the rest of the team is doing.
- **QMIX / VDN** (Rashid et al., 2018; Sunehag et al., 2017) — value decomposition:
  one team value factorised into per-unit values through a mixing network so that a
  greedy choice per unit is a greedy choice for the team. Off-policy Q-learning
  rather than PPO; strong on small-team benchmarks (SMAC), harder to scale to
  dozens of heterogeneous units with many action heads.
- **MAT — Multi-Agent Transformer** (Wen et al., 2022) — treats the team's joint
  action as a *sequence* and decodes it unit by unit with a transformer, so each unit
  conditions on what earlier units decided. Effective, but inherently sequential per
  step and centralised at execution.

*Here:* the policy is decentralised (each unit acts on its own senses and the team
map), training is PPO, and — as of this version — the **critic is centralised**: it
also sees both teams' maps pooled over the whole board and the game-wide counts.
That is the MAPPO idea applied to a game where the "global state" is naturally the
two maps.

## 2. Shared reward and *team spirit*

- **OpenAI Five** (Berner et al., 2019) — five heroes, each with its own LSTM copy of
  one network, no explicit communication. Coordination came from two things: a
  shared observation of the game state, and **team spirit** — a hyperparameter
  blending each hero's own reward with the team average, *annealed from 0.3 toward
  1* over training. Early on, individual reward gives a faster, cleaner learning
  signal; later, only team reward produces sacrifice plays. They also found long
  horizons (γ-annealed) and a huge batch essential.
- **Hide-and-seek** (Baker et al., 2020) — team reward only, entity-centric attention
  policies; six distinct strategy phases emerged over ~500 million episodes. The
  cautionary lesson: emergent strategy at that level took orders of magnitude more
  experience than a single GPU session provides.

*Here:* team spirit has been in the reward since the base game and is now
**annealed 0.3 → 0.7** over the first 1 500 iterations, for OpenAI Five's reason.
Earlier in this project a *fixed* 0.5 together with generous economy rewards taught
whole armies to farm; annealing lets combat be learned individually first.

## 3. Learned communication

- **CommNet** (Sukhbaatar et al., 2016) — every agent broadcasts a continuous vector,
  every agent receives the mean; trained end-to-end by the listeners' gradient.
- **TarMAC / ATOC / IC3Net** — the same with attention over senders, or a learned
  gate deciding *whether* to talk.
- **Neural Map** (Parisotto & Salakhutdinov, 2018) — a single agent's structured
  memory: a 2-D grid over the world that the agent writes at its own position and
  reads with a global summary plus a context lookup. Solved maze tasks that flat
  LSTM memory could not, because the memory has the geometry of the problem.

*Here:* the project went through exactly this sequence — discrete words (learned
nothing: no gradient through sampling), CommNet's mean, attention over teammates —
and then replaced *all* of them with a **shared team map**: Neural Map's write-where-
you-stand grid, but per team rather than per agent, so every unit reads what every
other unit wrote. Part of each sector's vector is a fixed sighting report from the
writer's sensors (so the map means something before any language is learned); the
rest is learned. See `docs/HISTORY.md` for what went wrong along the way.

## 4. Hierarchy: commanders and orders

- **Feudal networks** (Vezhnevets et al., 2017) — a manager sets goals in a latent
  space at a slower timescale; a worker is rewarded for moving toward them.
- **Hierarchical MARL for StarCraft** (Pang et al., 2019; Lee et al.) — a high-level
  policy picks macro-actions or targets for groups; low-level policies fight.
- **AlphaStar** (Vinyals et al., 2019) — one centralised controller over every unit,
  with a transformer over the unit list and a pointer network to select which units
  an order applies to; plus a **league** of exploiters to keep self-play honest.

*Here:* bases are the natural commanders — stationary, seeing five times farther,
and holding the economy. In this version each base writes an **orders layer** over
the *whole board* (an 8 × 8 plan upsampled onto the map's 32 × 32 sectors) that every
unit reads alongside the sightings in its own sector. Units are never told what an
order means; the bases' output and the units' response are trained together by the
same team-level reward. It is Feudal networks' manager/worker split with the goal
space made spatial and shared.

## 5. Keeping self-play honest

- **Fictitious self-play / PSRO / leagues** — play against a distribution of past
  and specialised opponents so the current policy cannot forget how to beat an old
  style or be exploited by a narrow one.

*Here:* a quarter of the games put one side under a frozen past snapshot or a
scripted rushing bot. Without the bot, an earlier policy lost every game to a
strategy that simply charges.

## What actually moved the needle in this project

In rough order of impact, measured by the behaviour probes in `report/collect.py`:

1. Fixing the game before tuning rewards — scouts stepping over hearts, blocks
   placed at random because of an entropy bonus, an order head that never had a
   heart to spend.
2. Reward *shape* over reward *size* — per-turn penalties that are mild enough that a
   fresh policy still moves and shoots; potential-based pulls that guide without
   changing what is worth doing.
3. Opponent diversity (the bot and past snapshots).
4. Memory with the geometry of the problem (the map) over flat messages.
5. Training on three board sizes and four terrains at once, so tactics generalise.
