# Multi-Agent Tanks

Two armies of tanks fight over a maze. Every tank and every base on the board is
controlled by the **same neural network**, trained from scratch by self-play on a
single GPU. Nothing about tactics is scripted: the network decides where each tank
drives, when it shoots, when a scout ferries supplies or patches up a wounded ally,
where barricades go up, what the bases build, what each unit writes onto the team's
shared **map of the war**, and what orders the bases lay over that map for the rest
of the team to follow.

The point of the project is to find out how much coordinated, board-wide play —
squads, base defence, sieges, supply lines, medics, fortifications — can *emerge*
from reward design and self-play alone, and to make the loop fast enough that a
reward idea can be tested in an hour rather than a day.

![overview](docs/overview.png)

---

## Contents

- [Quick start](#quick-start)
- [The game](#the-game)
- [What an agent senses](#what-an-agent-senses)
- [The network](#the-network)
- [Training](#training)
- [Rewards](#rewards)
- [The viewer](#the-viewer)
- [Results and current status](#results-and-current-status)
- [Collecting results](#collecting-results)
- [Performance notes](#performance-notes)
- [Further reading](#further-reading)

---

## Quick start

```bash
pip install -r requirements.txt      # PyTorch with CUDA, numpy, scipy, pygame-ce, tensorboard

python viewer.py --policy models/tank_policy_it0800.pt   # watch the shipped policy play (CPU only)

python train.py                      # train from scratch; writes tank_policy.pt every iteration
tensorboard --logdir runs            # watch the curves
python viewer.py                     # watch (or play against) the policy being trained, live
```

Training needs a CUDA GPU; it was developed on an NVIDIA DGX Spark (GB10). The
viewer runs on the CPU, so it doesn't steal time from training, and it reloads
`tank_policy.pt` whenever `train.py` saves a new one.

| file | what it is |
|---|---|
| `arena.py` | the game: map generation in four terrain styles, physics, sensors, rewards, and a scripted benchmark bot |
| `train.py` | the policy (encoder → shared team map with orders → GRU memory) and self-play PPO across three board sizes |
| `viewer.py` | live pygame viewer: top-down with the team map and markers overlaid, and a first-person mode to drive a tank yourself |
| `report/collect.py` | plays the saved checkpoints and writes every number a write-up needs to `report/data/` |
| `models/` | a trained policy to try without training |
| `docs/HISTORY.md` | how the design got here: what was tried, what broke, and why |
| `docs/SURVEY.md` | how other multi-agent systems plan at scale, and what this project borrows from them |

---

## The game

The board is a **1125 × 1125** field of rock in one of four terrain styles —
*rubble* (many small rocks), *boulders* (fewer, bigger), *corridors* (long thin
walls) or *open* (sparse). Each team has **5 bases**, spread at least 250 tiles
apart, and starts with **28 tanks** in a ring around them, with room to build up to
40. Board size and army size are parameters: training uses three sizes at once and
the viewer will play any.

**Winning.** A team that loses all of its bases loses. If both still have bases when
the 1500-turn clock runs out, the team with more bases wins (more tanks breaks a
tie). Losing every tank doesn't end the game — a base with supplies can build more.

| unit | per team | hp | speed | what it does |
|---|---|---|---|---|
| scout | 6 | 4 | 3 | collects hearts (carries 3); unloads them at a friendly base, or spends one to heal a teammate within 4 tiles |
| soldier | 18 | 6 | 2 | fires bullets |
| heavy | 4 | 8 | 1.5 | fires bullets, plus a slow missile that flies over walls |
| base | 5 | 40 | — | long-range missile (5× a tank's reach), sees 5× farther, spends stored hearts, issues orders |

**Hearts** are the only resource. They lie scattered around the map, only scouts can
carry them, and a base spends them on one of four orders, at most one every 8
turns: **repair** (5 hp to itself and 2 to every tank beside it, 1 heart), **build** a
scout / soldier / heavy (2 / 5 / 10 hearts), or **raise a wall** — a row of 5 blocks,
14 tiles out in the direction its turret points (1 heart).

**Blocks.** Any tank can drop a 2 × 2 block just ahead of it, once every 4 turns.
Blocks stop tanks and bullets like rock does, but break after 3 hits (missiles fly
over both). A team can have at most 60 standing. Friendly fire is on — for bullets,
and for shooting your own team's blocks.

## What an agent senses

Each agent gets a 212-number observation, laid out the same way for tanks and bases:

- **itself** — role, health, hearts carried, cooldowns, position (as a fraction of
  the board), heading, and how many tanks and bases each side still has;
- **radar** — 16 sectors all the way round, reaching 160 tiles, *through walls*. Per
  sector: the nearest rock and placed block along the sector's centre line, the
  nearest friendly and enemy tank, the most badly hurt friendly tank, the nearest
  friendly and enemy base (bases show at any distance), how many enemy tanks are
  closing on a friendly base in that direction, how damaged the enemy base there
  is, and the nearest heart;
- **vision** — 9 sectors in a forward cone, 48 tiles, *blocked by walls*. Per
  sector: rock, placed blocks, the nearest enemy, and the nearest heart.

Bases see five times farther on both. Every distance is normalised, so a policy
trained on the small board plays the big one unchanged.

## The network

One network is shared by every agent on both teams; agents are told apart only by
what they observe.

```
                team map, one per team: 32 × 32 sectors × (24 sightings + 8 orders)
                     ▲ write my sector (gated)   ▲ orders over the whole board (bases only)
                     │                          │                    │ read: 5 × 5 sectors round me
                     │                          │                    │       + the whole map max-pooled to 8 × 8
                     │                          │                    ▼
                     │                          │               digest (256)
observation ─► encoder (2 × 512) ─► [encoded observation, digested read] ─► GRU memory (512) ─► 5 action heads
                                                                                  │
                                              both teams' maps pooled ─► overview ─┴─► value (critic, training only)
```

**The team map.** Each team keeps a grid of **32 × 32 sectors** laid over the board
— whatever the board's size, so the same network plays a 400-tile skirmish and an
1125-tile war — and every sector holds a vector. Every turn each living unit
**writes** to the sector it is standing in and **reads** the 5 × 5 sectors around it
plus the whole board max-pooled to 8 × 8: its picture of the entire war, not just its
corner of it. Sectors nobody visits fade slowly, so stale reports don't linger. This
map is the *only* channel between teammates; it replaced a radio in which everyone
broadcast to everyone, because what a war needs to share is *where* things are, and
it should outlive the scout that saw them.

Each sector's vector has three parts:

- **a sighting report** (8 numbers) copied straight from the writer's sensors — here
  I am, enemies in how many radar directions, how close the nearest enemy tank and
  enemy base are, my health, the worst threat I see to a friendly base — so the map
  means something from the first turn;
- **a learned message** (16 numbers), written through a gate the writer controls;
  what it means is whatever the readers' policy gradient teaches the writers to say;
- **orders** (8 numbers) written by the **bases only**. A base is the natural
  commander — stationary, seeing five times farther, holding the economy — and each
  one lays a gated 8 × 8 plan over the *whole board*. Every unit reads the order for
  its own sector alongside the sightings. Nobody is told what an order means: a
  base's plan gets its gradient from every unit that acted on it. It is the
  manager / worker split of Feudal networks with the goal space made spatial and
  shared.

The read is squeezed through a 256-unit **digest** before it joins the agent's own
senses; fed raw it was four times wider than the encoded observation and drowned it.

**A centralised critic.** The value head, used only in training, also sees *both*
teams' maps pooled over the board — the MAPPO observation that most of the
credit-assignment problem in a team game is a critic that can't see what the rest of
the team is doing.

**Memory.** A GRU lets an agent remember recent turns — an enemy that went behind a
wall, which way the squad was heading. It is wiped when an agent dies or its game
ends.

**Actions** (all discrete): move (3 throttle × 3 steering), fire, special (missile /
unload / heal), base order (6 choices), and place block.

## Training

`train.py` is self-play PPO, built around keeping the GPU busy.

- **Hundreds of games at once.** Every agent of every game is one row of a single
  `(games, agents, …)` tensor, and `torch.compile` fuses the sensors and physics
  into a handful of GPU kernels. Games reset independently, so training is a
  continuous stream of 32-turn rollouts.
- **Three board sizes, four terrains, as a curriculum.** Training starts on the
  small board alone (400 tiles, 14 tanks and 3 bases a side, 1024 games — the
  cheapest iteration, where fights are learned fastest), adds the medium board
  (560, 28, 5, 768 games) at iteration 300 and the large one (800, 40, 7, 384 games)
  at 900, then rotates through all three. Each game is on one of sixteen maps in the
  four terrain styles, so what is learned has to be about tanks and terrain in
  general, not one arena.
- **Recurrent PPO.** The update replays each game's rollout in order through the
  GRU *and the team map* (backpropagation through time), starting from the memory
  the rollout began with: 64 whole games per minibatch, 2 epochs, in bf16.
- **Team spirit is annealed** from 0.3 to 0.5 over the first 1 500 iterations,
  the way OpenAI Five did it: individual reward learns to fight fastest; team
  reward buys the plays that cost the individual.
- **Opponents.** Both sides of most games are the current policy. In a quarter of
  the games one side is played by someone else — half by a frozen **past snapshot**
  (a pool of the last 10, refreshed every 50 iterations), half by the scripted
  **raider** bot — so the policy can't forget how to beat older styles and has to
  handle an all-out rush.

Every 100 iterations it saves a snapshot to `checkpoints/` and plays full games on
the medium board against the raider and against a copy of itself **cut off from the
team map** — if the map carries anything useful, that copy should lose.

```bash
python train.py --minutes 120            # stop after two hours (default: run until Ctrl-C)
python train.py --resume                 # continue from tank_policy.pt
python train.py --arenas medium,large    # a subset of the board sizes
```

An iteration takes about 6 s on the small board and 10–12 s on the medium one; the
PPO update is the larger half (a 3.9 M-parameter recurrent policy replayed over 32
turns for every agent of every game). The log prints one line per iteration;
`moving` — the share of tank turns that aren't "stand still" — comes first because
a policy that stops moving is the most common way a reward change goes wrong.

## Rewards

This is where most of the work went. Every term below is there because, without it,
self-play found something degenerate instead — [docs/HISTORY.md](docs/HISTORY.md)
tells those stories. All the weights are named constants near the top of `arena.py`.

**Individual credit.** +1 per kill, +0.2 per damage dealt, −0.05 per shot that hits
nothing, −1 for dying. A **blind shot** — fired with no enemy anywhere within radar
range — costs −0.1, so spraying at nothing is a loss; a shot with an enemy about is
never charged beyond the miss, however badly aimed, because charging every shot
taught fresh policies to stop shooting before they could aim. Hitting a teammate
costs what hitting an enemy earns (−0.2 per damage); killing one, −2.

**Team spirit.** Every agent's reward is blended with its team's average — 30 % at
the start of training, rising to 50 %. Above 50 %, soldiers learned to live off the
scouts' deliveries.

**Bases.** Losing has to hurt more than winning pays, or teams happily trade bases.

- *Attacking:* +3 to the tank that destroys a base, +0.2 extra per damage to a base,
  and **+0.2 more per allied tank also at that base** (beyond the first, up to four),
  so a massed siege on one target beats trickling in; +2 to every member of the team
  that takes it.
- *Defending:* −4 to every member per base lost (−8 to the base itself), and damage
  to an enemy within 60 tiles of one of your bases is worth three times as much.
- *The game:* +3 for winning outright, −6 for losing. **The clock is no refuge**: a
  game decided on the time limit pays its winner only +0.5, costs its loser −6 all
  the same, and costs *both* sides −2 when it is a tie.

**Squads.** A soldier or heavy pays up to −0.01 a turn for drifting away from its
comrades (nothing while its second-nearest ally is within 12 tiles, all of it 40
tiles beyond), and −0.015 a turn once it has gone 60 turns without hitting an enemy
— so a group has to fight together, not just huddle. It is also **drawn toward the
nearest enemy**, tank or base, by a potential-based pull that reaches across the
board and nets to zero over the approach, so it guides without being farmable.
Every tank that hit an enemy in the 20 turns before it died gets +1, as much as the
killer, so focusing fire pays. A comrade dying close by costs a little (−0.05, or
−0.15 for a heavy). Only actually overlapping another tank is penalised (−0.1 a
turn). These terms are deliberately mild: early in training, when nothing can aim
yet, every one of them lands on *doing something*, and at twice these values fresh
policies learned to stand still.

**Barricades.** A block dropped with no enemy in radar range and no base of yours
nearby is pointless and costs −0.15; anywhere else placing is free, and a block that
extends a wall of your team's blocks earns +0.15. The tank that placed a block earns
+0.5 for every enemy bullet it stops and +0.2 for every enemy tank it stops at one of
its bases.

**Economy and medics.** A scout gets +0.2 per heart picked up and +0.6 per heart
delivered, with the same kind of potential-based pull toward the nearest heart while
it has room and toward the nearest base while carrying; +0.5 per hp it heals on a
teammate within reach, and +0.5 more for reaching one below 35 % health. A scout
that goes 60 turns without delivering or healing pays the idle penalty like a
soldier. A base gets +0.05 per hp a repair restores, and +0.1 per heart spent on a
tank and per wall block raised. All of this stays short of the combat rewards:
shared through team spirit, generous economy rewards twice taught entire armies to
stay home and farm.

## The viewer

```bash
python viewer.py                                  # the full 1125 board, 28 tanks and 5 bases a side
python viewer.py --grid 200 --tanks 10 --bases 1  # a small skirmish
python viewer.py --opponent raider                # the policy (blue) against the scripted bot (red)
python viewer.py --policy models/tank_policy_it0800.pt
```

Zoomed in, tanks are drawn with treads, hull, turret and barrel (a yellow turret is
a heavy; a scout carrying a heart shows it); zoomed out they are arrowheads. Bases
are big squares with a health bar, their stored hearts, and a turret showing where
a wall order would go. Dark rock is permanent, sandy blocks are placed barricades
tinted by team, bullets leave a trail, and units that die burst. The HUD shows the
board size and terrain style. When a game ends, the next one starts three seconds
later with the latest weights.

**The team map.** Press **M** to overlay what blue's units have written about each
sector of the board, again for red's, and again to hide it. Colour comes from the
first three numbers of each sector's vector and opacity from how much has been
written there, so you can watch the team's picture of the war build up and fade —
and, in first person, the same overlay on the minimap.

**Markers.** Every base on either side is always marked: on screen it is visible
itself, off screen a square in its team's colour is pinned to the edge of the view
with an arrow and its distance (grey for a destroyed base). Once you select or drive
a tank, every tank within its radar range gets a triangle tag with its distance,
off-screen ones at the edge too. In first person the markers sit on the horizon
like waypoints, and the ones behind you stack down the sides. **K** hides them.

**Watching.** The mouse wheel zooms, drag pans, click selects a unit, **F** follows
it, **Space** pauses, **N** skips to a new game, **+ / −** change the speed.

**Playing.** Press **C** to take over the selected tank (or a random blue one). The
game slows to 8 turns a second and the network keeps controlling everyone else,
your teammates included. **V** switches between two views:

- **top-down** — the camera follows you, and your tank turns to face the mouse;
- **first person** — a raycast 3D view from the tank, with towering rock, chest-high
  barricades, the other tanks and bases, bullets and hearts, and a minimap. The mouse
  is captured: move it to look around, and the tank turns to follow.

**W / S** drive, **click** or **Space** fires, **right-click** or **E** is the special,
**Q** drops a block, **P** pauses, and **C** or **Esc** hands the tank back.

![first person](docs/first_person.png)

## Results and current status

The numbers below come from `report/collect.py` on the latest run (920 iterations,
about two hours on one GPU, curriculum over three board sizes), measured in **full
self-play games on the medium board with no scripted units** — 128 games per
checkpoint, 512 for the head-to-heads. The shipped model, `models/tank_policy_it0800.pt`,
is the strongest checkpoint of that run: it beats the final one 68–30 and does three
times better against the scripted rusher, so the last hundred iterations of self-play
had drifted rather than improved.

| per game, medium board | it 100 | it 300 | it 600 | it 800 | it 920 |
|---|---|---|---|---|---|
| games ending by elimination | 0 % | 0 % | 0 % | **94 %** | 85 % |
| enemy bases destroyed | 0 | 1.1 | 1.8 | **7.9** | 6.7 |
| kills | 0.8 | 34 | 36 | 39 | 37 |
| accuracy | 0.02 | 0.19 | 0.25 | 0.33 | 0.33 |
| shots fired with nothing in sight | — | 5 % | 1 % | 2 % | 3 % |
| hearts picked up / delivered | 39 / 15 | 159 / 122 | 177 / 134 | 93 / 57 | 101 / 57 |
| tanks built / base repairs | 1.7 / 1.3 | 26 / 13 | 25 / 24 | 8.5 / 20 | 9.1 / 22 |
| heals / rescues of dying allies | 4.5 / 0.5 | 16 / 4.1 | 18 / 6.1 | 21 / 6.5 | 22 / 6.5 |
| enemy moves stopped by placed blocks | 25 | 28 | 23 | 29 | 49 |
| tanks massed on a base under siege | 1.9 | 1.8 | 1.9 | 2.2 | 2.3 |

What the run shows, in order of how convincingly:

- **Bases fall and games are won.** Until the base-repair rebalance at iteration
  ~640 not one game ended before the clock; after it, nine in ten do, and a game
  lasts 650 turns instead of the full 747.
- **Fire discipline is learned, not scripted.** A third of shots land, and only 2–3 %
  are fired with no enemy anywhere in radar range.
- **The economy runs itself** — pickups, deliveries, repairs, builds, heals and
  rescues all from zero — and it settles at a level that doesn't crowd out fighting
  once the economy rewards were cut back (the it 600 column is the farming peak).
- **Barricades matter more than they look:** ~130 blocks a game stop 25–50 enemy
  moves, though only a fifth extend a wall.
- **Sieges are still small** (2.2 tanks at a base under attack) and **base defence
  hasn't emerged**: no more friendly tanks are near a threatened base than a calm
  one. These are the open problems.
- **The team map is not yet pulling its weight.** Over 512 games the copy that reads
  and writes its map beats a copy cut off from it only 52–45, and what teams write
  tracks where their *own* units are (correlation 0.40) far more than where the
  enemy is (0.08). The sighting-report channels and the bases' orders layer are the
  current bet for fixing that; the collector measures both.
- **The scripted rusher still wins** (86 % against the shipped model). It sees
  everything and charges; beating it needs the defence that hasn't emerged yet.

Throughput on the GB10: one game-turn for 1024 medium games takes 54 ms of physics
plus 31 ms of sensors — 1.1 million agent-steps per second before the network, 0.5
million with it (`report/data/speed.json`).

## Collecting results

```bash
python report/collect.py
```

This plays the saved checkpoints and writes `report/data/`:

- `scalars.csv` — every TensorBoard curve of the run;
- `behaviour.csv` — behaviour per checkpoint, measured over full games with no
  scripted units: fire discipline, whether hurt tanks back off, how often soldiers
  are grouped, how many defenders turn up at a threatened base versus a quiet one,
  siege group sizes, economy, heals, barricades;
- `ladder.csv` — the final policy against earlier checkpoints, from both sides;
- `radio.json` — the team-map test (a copy cut off from its map, 512 games) and what
  the map encodes: how well the amount written in each sector tracks where enemy
  tanks and own tanks actually are;
- `raider.json` and `speed.json` — the scripted-bot benchmark and throughput.

`report/baseline_mean_radio/` keeps the code and training log of an earlier
generation — a radio that averaged every teammate's message — for comparison.

## Performance notes

- On an idle GB10 one game-turn for all 1024 medium games takes about 85 ms
  (physics ~54 ms, sensors ~31 ms): roughly 1.1 million agent-steps per second
  before the network.
- The PPO update, not the simulation, is the larger cost. Profiling shows it spread
  evenly over hundreds of small kernels per turn — GPU-bound, not launch-bound — so
  bigger minibatches don't help; fewer epochs and the small-board curriculum do.
- The team map costs little: reads are one gather (25 sectors per agent) and one
  max-pool; writes are three scatter-adds. All of it is fused by `torch.compile`,
  compiled once per board size (dynamic shapes mis-specialised a stride the moment
  a second size joined).
- Cost grows with the square of the army size (radar, bullets and the pairwise
  reward terms compare every agent with every other), which is why armies are 28 a
  side rather than 280.
- Placed blocks live in one grid cell per 2 × 2 tiles — 80 million cells across 1024
  games — so every block update happens in place. Copying that grid each turn once
  cost more than the rest of the physics put together.
- The viewer's single game runs on 8 CPU threads: with all 20, thread hand-offs made
  every turn three times slower.

## Further reading

- [docs/HISTORY.md](docs/HISTORY.md) — the design's history: every reward term and
  most of the architecture exist because an earlier version without them produced
  something dumb, and the dumb things are worth knowing about.
- [docs/SURVEY.md](docs/SURVEY.md) — how OpenAI Five, AlphaStar, MAPPO, QMIX,
  CommNet, Neural Map and Feudal networks approach large-scale coordination, and
  which of their ideas are in this code.
