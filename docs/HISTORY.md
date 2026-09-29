# How the design got here

Almost every rule and reward term in this project exists because an earlier version
without it produced something dumb. This is the short version of that story, roughly
in order — useful if you want to change a reward and would rather not rediscover
the same failure modes.

## The game

**From "Tank Trouble" to a strategy game.** It started as a small arena in the
spirit of Tank Trouble: a few tanks, bouncing bullets, walls. Ricochets went early
(they made fights a lottery), then fixed turrets went, and rectangular scattered
rock replaced long barrier walls — thick barriers trapped tanks in pockets, while
lots of small obstacles give cover to dodge behind. Bases, a heart economy, three
tank roles, placed blocks and base-built walls came later, one at a time.

**Board size is a speed/interest trade-off.** The board went through 450, 10 000,
4 500, 2 250 and finally 1 125 tiles a side. Very large boards meant fresh policies
almost never met an enemy, and very large armies (410 agents a game) were slow: the
simulator compares every agent with every other, so cost grows with the square of the
army. The compromise is a 1 125 board for watching and a 560 board for training,
with every distance the network sees normalised so one policy plays both.

**Where tanks start matters more than it looks.** A fresh network placed around its
own bases on a big board never found the enemy, so for a long time tanks started at
random spots. Once policies were strong enough, spawning at bases made the game about
bases again — but packing six tanks into a 10-tile disc round each base taught a
fresh policy to *freeze*: any move risked an overlap penalty and any shot hit a
friend. Tanks now start evenly spaced on a ring round their base.

## Rewards — the failure modes

**A fresh policy pays for every penalty the moment it acts.** Before it can aim,
moving risks bumping a teammate, shooting inside a group mostly hits friends, and
straying from the group costs too — while an "idle" penalty hits everyone equally, so
it never favours acting. With the overlap penalty at −0.25, friendly damage at −0.4
and the squad penalties at −0.02, freshly trained tanks moved on 66 % of turns at
first and 9 % a few hundred iterations later, and never learned to fight. Halving
them fixed it: tanks kept moving on ~75 % of turns and were landing kills within
80 iterations.

**Dense penalties swamp everything.** Early per-turn penalties for clustering,
hugging corners and bumping walls were bigger in total than the combat reward; tanks
learned to avoid everything, including the enemy. Lesson: keep per-turn terms small
next to what a kill is worth, and measure behaviour after every change.

**Punishing death too hard makes cowards.** A death penalty close to the value of
losing the game (−2.5 against −3) collapsed play into passivity. It is −1 now.

**Scripted reflexes break learning.** For a while tanks had hard-coded dodge,
retreat and "keep your distance from allies" reflexes layered on top of the network.
The separation reflex in particular broke up every firefight. All reflexes were
removed; the network controls the tanks directly.

**"Stay in formation" bonuses teach clumping; exemptions teach camping.** A reward
for being near allies taught tanks to pile up. A penalty for being out of formation
worked better — but when base guards were exempt from the idle penalty, whole armies
learned to sit at home. The version that works charges a soldier for drifting from
its comrades *and* for not fighting, at the same rate, so a group has to move and
fight together.

**Economy rewards shared by the team become the whole game.** With team spirit
blending everyone's reward, generous rewards for delivering hearts and building tanks
meant a soldier earned more by staying safe at home than by fighting. Kills fell from
about 44 a game to 5 while both sides built impressive walls around their bases and
farmed. The economy rewards are now small; hearts matter because of what they buy.

**Symmetric base rewards make teams trade bases.** If losing a base costs what
taking one earns, racing for the enemy's bases while ignoring your own is fine. Losing
a base now costs every teammate −4 against +2 for taking one, losing the game costs
twice what winning pays, and damage to intruders near your bases is worth triple.

**Some actions collapse to "never".** Placing a block costs a little immediately and
pays off only later, and only sometimes. Twice the place-block output collapsed to
exactly zero probability, after which no reward could ever teach it — tanks simply
never tried. That head now gets a larger entropy bonus (0.05 instead of 0.01), and a
fresh network starts it near, but not at, "never".

**Caps need to be per purpose.** With one block cap per team, tanks' scattered
blocks filled it and crowded out the walls bases build with hearts. Scattered blocks
also have to cost something (−0.15), while blocks that extend a wall are free, so
barricades rather than confetti are what pays.

## The radio, then the map

The first radio let each tank pick one of a few discrete "words" every turn. It
carried exactly zero information: a sampled word gets no gradient, so nothing ever
taught a speaker what to say. It was replaced by a CommNet-style continuous channel —
each agent broadcasts a vector and hears the average of its teammates' — which is
differentiable end to end, and a copy with the radio muted started losing to the
talking copy (by up to 80–18). Attention over teammates came next, so an agent could
listen selectively rather than to everyone at once.

Both of those are *who-to-who* channels: a message is tied to the agent that sent
it, and it is gone next turn. What a war actually needs to share is *where* things
are, and it should outlive the scout that saw them. So the current design is a
shared **map**: each team has a 24 × 24 grid of sectors over the board, each holding
a learned 16-number vector; a unit writes only to the sector it stands in and reads
the sectors round it plus a coarse view of the whole board. Reports persist (fading
slowly) after the writer has moved on or died, and the same map fits any board size
because it is defined in fractions of the board. `report/collect.py` measures how
well what a team has written tracks where the enemy really is.

The first version fed the read — 2 136 numbers for the 5 × 5 window plus the 8 × 8
pooled board — straight into the GRU beside the 512-number encoded observation. Over
512 games a team did *better* with its map switched off (54–40): early in training
the map is mostly noise, and four times as much of it as signal. A 256-unit digest
layer between the read and the GRU fixed the proportions — but not the content: 600
iterations later what teams had written tracked where their own units were
(correlation 0.39) and hardly at all where the enemy was (0.09), and the map still
made no difference to who won. Waiting for writers to invent a language that
readers can't yet use is slow. So eight of each sector's 24 numbers are now a fixed
sighting report from the writer's own sensors — enemies on its radar, the nearest
enemy tank and base, its health, threats to friendly bases — that readers can use
from the first turn, with the other sixteen left for whatever the network learns to
add.

## The network

The first policies were small MLPs (256 units). Swapping in a 2-layer 512-unit encoder
and a 512-unit GRU memory, trained with backpropagation through time, made a large
difference: the recurrent policy won half its games against the scripted raider
after about 400 iterations, where the MLP had needed about 4 000.

## Planning at the scale of the board

The map gave units somewhere to *read* the state of the war, but nothing in the
design let anyone *decide* for the team — every unit still chose its own action from
its own corner of the map, and the value that trained those choices saw only that
corner too. Two ideas from the literature (see `docs/SURVEY.md`) fixed both halves:

- **A centralised critic** (MAPPO): the value head, which only training uses, now
  sees both teams' maps pooled over the whole board, so a unit's action is judged
  against what the rest of the war was doing.
- **Commanders writing orders** (Feudal networks, made spatial): each base writes an
  8 × 8 plan over the entire board into a separate orders layer of the team map,
  and every unit reads the order for the sector it stands in. What an order means
  is never specified; the bases' plans and the units' responses are trained
  together by the same team reward.

Team spirit is annealed from 0.3 to 0.7 rather than fixed, as OpenAI Five did:
early on, individual reward is the faster teacher.

**No battles.** With the economy finally working and team spirit heading for 0.7,
a soldier earned more from the team's deliveries than from fighting, and games
drifted to the clock with the armies apart. Three changes, none of them a bigger
kill reward: a potential-based pull toward the nearest enemy for soldiers and
heavies (the device that got scouts to hearts, and just as unfarmable), a clock that
pays a timeout winner almost nothing and costs a tie both sides, and team spirit
capped at 0.5.

## Speed, again

With the map and the GRU, an iteration had grown to 15 s, 11 s of it the PPO update.
Profiling showed the update evenly spread over hundreds of small kernels per turn —
GPU-bound, not launch-bound — so bigger minibatches changed nothing; two epochs
instead of three and a curriculum that starts on the small board alone (half the
agents, fights learned soonest) roughly halved the time to a fighting policy.

## One arena is not enough

Trained on one board, the policy learned that board: where the bases sit, how far
the enemy is, how long the game lasts. Training now rotates through three board
sizes (400, 560 and 800 tiles a side, with armies to match) and sixteen maps in four
terrain styles, so what it learns has to be about tanks and terrain in general.

**Cheap shots get sprayed — but charging every shot stops the shooting.** With only
wasted shots penalised, and lightly (−0.03), tanks fired constantly. Charging every
shot (−0.02) and every miss (−0.08) fixed the spraying and broke learning: at 5 %
accuracy a shot has negative expected value, so within 30 iterations the policy fired
a tenth as often and never learned to aim. Charging only shots with no enemy in the
forward cone was gentler but still slowed combat learning threefold from scratch: a
fresh policy can't yet tell "dead ahead" from "nearby", so it learns to hold fire.
What works is charging only shots with no enemy anywhere in radar range — shooting
at nothing — and leaving every shot near a fight to the small miss penalty.

**The scouts were stepping over the hearts.** When scouts were sped up from 2.4 to
3 tiles a turn, the pickup radius stayed at 1 tile — so a scout driving straight at a
heart went from 1 tile short of it to 2 tiles past it and never picked it up. Pickups
fell to about one a game and everything downstream (deliveries, repairs, builds,
heals) died with them, and it took a staged test with one scout, one heart and no
walls to see it. The radius is 2 now.

**Three action heads that never learned.** Four hundred iterations into the map
runs, the scouts' unload/heal choice was still a coin flip, the bases' order head
uniform over its six options, and the place-block head 50/50 — the last held there
by the extra entropy bonus meant to stop it collapsing, which now that placing is
free near fights just made blocks confetti (130 a game, 9 % of them extending a
wall). The special and order heads had nothing to learn from: scouts picked up one
heart a game, so no base ever had a heart to spend. The fix is a bootstrap for the
chain — a dense, potential-based pull toward the nearest heart for a scout with room
(it nets to zero over the trip), bigger pickup and delivery rewards, and the block
head's entropy bonus back to normal.

**...and a ceiling after all.** With the pickup bug fixed and the pull in place,
the doubled rewards (pickup 0.4, delivery 1.0) sent deliveries up sixfold in an hour
— and kills down by two thirds, with half the soldiers idle: half of every reward is
the team's average, so a scout's delivery pays a soldier for standing still. The
pulls stay (they net to zero); the pickup and delivery rewards went back down.

**Economy rewards need a floor, not just a ceiling.** After the farming episode the
economy rewards were cut so far that scouts stopped bothering: a few deliveries a
game, no repairs, no builds. The fix was not to raise them back but to make idling
cost scouts the way it costs soldiers, and to pay a scout for heading home while
carrying (a potential-based term that nets to zero over the trip).

## Self-play

Pure self-play drifts into styles that beat themselves but not much else — at one
point a well-trained policy lost every game to the dumb scripted raider, which just
charges. Mixing in opponents fixed it: a quarter of the games put one side under
either the raider or a frozen past snapshot of the policy.

## Speed

- `torch.compile` on the sensors made ray-marching about 40× faster than eager PyTorch.
- A shared pool of bullet slots let team 0 grab them first and biased every game;
  each agent now owns its own slots.
- Copying the placed-block grid (80 million cells across 1024 games) every turn cost
  more than all the other physics; block updates are now in place, with a spare cell
  that soaks up the writes of agents that aren't placing.
- The viewer got three times faster by using 8 CPU threads instead of 20.
