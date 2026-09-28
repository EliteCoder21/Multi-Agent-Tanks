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

## The radio

The first radio let each tank pick one of a few discrete "words" every turn. It
carried exactly zero information: a sampled word gets no gradient, so nothing ever
taught a speaker what to say. It was replaced by a CommNet-style continuous channel —
each agent broadcasts a vector and hears the average of its teammates' — which is
differentiable end to end, and a copy with the radio muted started losing to the
talking copy (by up to 80–18). The current version replaces the average with
attention, so an agent can listen selectively rather than to everyone at once;
`report/collect.py` measures who it actually listens to.

## The network

The first policies were small MLPs (256 units). Swapping in a 2-layer 512-unit encoder
and a 512-unit GRU memory, trained with backpropagation through time, made a large
difference: the recurrent policy won half its games against the scripted raider
after about 400 iterations, where the MLP had needed about 4 000.

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
