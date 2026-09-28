"""Tank arena: a 1125x1125 maze of rectangular walls and two teams, each with 5
stationary bases (spread at least 250 tiles apart) and up to 40 tanks. Bases are
agents too -- the same network steers their turret, fires their long-range missile,
and decides what to do with the hearts scouts bring home: repair, build a tank, or
raise a wall. Every tank starts at one of its team's bases. A team that loses all its
bases loses; if neither has by the turn limit, the one with more bases wins.

Nothing is scripted between the policy and the agents. Every agent of every game is
one row of a (games, agents, ...) tensor, so thousands of games step together on the
GPU; the viewer is just batch=1 on the CPU. Team radio lives in the policy network
(see train.py).

Each agent sees through two kinds of rays (bases see 5x farther):
  * radar  -- 16 sectors all the way round, passes through walls. Per sector: the
              first wall on the sector's centre ray, the nearest friendly and enemy
              tank, how badly hurt the neediest friendly tank is, and the nearest
              friendly and enemy base (bases show at any distance).
  * vision -- 9 sectors in a forward cone, blocked by walls. Per sector: the wall,
              and the nearest enemy (tank or base) and heart in front of it.
"""
import math
import os
import numpy as np
import torch
from scipy import ndimage

# torch.compile has Triton write compiled kernels to ~/.triton/cache by default; on this
# machine that folder is owned by root, which crashes the compile, so use one we own
os.environ.setdefault('TRITON_CACHE_DIR', os.path.expanduser('~/.cache/tank_arena/triton'))

GRID = 1125                               # full board (the viewer); training uses a smaller one
TANKS = 40                                # tank slots per team; bases fill empty ones  } defaults -- any
START = (6, 18, 4)                        # scouts, soldiers, commanders alive at start } Arena can use its
BASES = 5                                 # per team                                    } own army sizes
N = TANKS + BASES                         # agents per team: tank slots, then bases
A = 2 * N
LIMIT = 1500
N_MAPS = 8
N_RECTS = 3440
BASE_GAP = 250                            # minimum distance between any two bases
BASE_CLEAR = 15                           # open ground kept around every base

ROLE_NAMES = ('scout', 'soldier', 'commander', 'base')
SCOUT, SOLDIER, COMMANDER, BASE = range(4)
SPEED = torch.tensor([3.0, 2.0, 1.5, 0.])    # scouts are quick: fetch hearts, reach the wounded
MAX_HP = torch.tensor([4., 6., 8., 60.])
SIGHT = torch.tensor([1., 1., 1., 5.])    # multiplier on vision and radar range
CARRY = torch.tensor([3., 1., 1., 30.])   # hearts a scout can carry / a base can store
TURN = .4

VISION_SECTORS, VISION_SPAN, VISION_RANGE = 9, .9, 48.
RADAR_SECTORS, RADAR_RANGE = 16, 160.
DIAG = GRID * math.sqrt(2)                # base distances are scaled by this on any board size

K = 5                                     # bullet slots per agent
BLOCK = 2                                 # a placed block covers BLOCK x BLOCK tiles
BLOCK_HP, BLOCK_CD, BLOCK_REACH = 3., 4, 2.5   # 3 hits to break; a tank can place one every 4 turns
MAX_BLOCKS, TANK_BLOCKS = 60, 30          # standing placed blocks per team; at most 30 of them placed by tanks
                                          # (with a single cap, tanks' scattered blocks filled it and crowded out base walls)
WALL_RADIUS, WALL_WIDTH = 14., 5          # a base's wall order: 5 blocks side by side, 14 tiles out where its turret points
BULLET_SPEED, BULLET_LIFE, GUN_CD = 3.5, 14, 5
MISSILE_SPEED, MISSILE_LIFE, MISSILE_CD = 3., 18, 15
BASE_MISSILE_SPEED, BASE_MISSILE_LIFE, BASE_GUN_CD = 5., 50, 8     # 5x a tank's reach
HIT_RADIUS, BASE_HIT_RADIUS = .9, 3.

HEARTS, HEART_RESPAWN = 400, 250         # only scouts can pick them up; plenty, or scouts can't find them
SPAWN_POOL = 4096                         # open spots sampled at reset for hearts to respawn on
PICK_RADIUS, HEAL_RADIUS, HEAL_AMOUNT, HEAL_CD = 1., 4., 2., 4
DEPOSIT_RADIUS = BASE_CLEAR + 3
ORDERS = ('none', 'repair', 'build scout', 'build soldier', 'build heavy', 'build wall')
BUILD_COST = {SCOUT: 2, SOLDIER: 5, COMMANDER: 10}
REPAIR_SELF, REPAIR_NEAR, ORDER_CD = 10., 2., 5
WALL_COST = 1                             # hearts per wall order

# reward -- mostly individual credit, so the agent that lands the kill, wastes the
# shot, carries the heart, builds the tank or dies is the one that feels it
KILL_W, BASE_KILL_W, DAMAGE_W, MISS_W = 1., 3., .2, -.03
DEATH_W, BASE_DEATH_W = -1., -8.
# losing hurts more than winning pays, so nobody trades their own bases for the enemy's:
# a lost base costs every teammate 4, taking one pays each 1; losing the game costs twice
# what winning earns. And defending pays: damage to an enemy within DEFEND_RADIUS of one of
# your bases is worth 3x, so tanks that stay to meet a raid out-earn tanks that charge
BASE_LOST_W = -4.                         # to every living member of a team, per base it loses
LOSS_SCALE = 2.
DEFEND_RADIUS, DEFEND_W = 60., .4         # extra per damage dealt to an intruder
FRIENDLY_DAMAGE_W, FRIENDLY_KILL_W = -.4, -2.   # friendly fire is on and costs more than enemy fire earns
STACK_RADIUS, STACK_W = 1., -.25          # per turn while overlapping another tank (only actual overlap);
                                          # -.05 was too weak: tanks learned to brawl point-blank on top of enemies
# team spirit: every agent's reward is blended 30% with its team's average, so helping
# the team pays even when someone else gets the credit -- the glue for team-wide plans
TEAM_SPIRIT = .3
BASE_WON_W = 1.                           # to every living member of a team, per enemy base it destroys
# economy: a heart is worth ~1.5 to the scout that brings it home (pickup + delivery) and
# again to the base that spends it -- on a tank (per heart of its cost) or a wall block.
# CARRY_PULL is potential-based shaping, +0.5 x hearts carried x how close the scout is to
# a friendly base: it pays for heading home as you go and nets to zero over a round trip
# (delivery was +1 and building +.25/heart: with team spirit, everyone shared it and whole
# armies learned to sit safely at home farming the economy -- kills fell 44 -> 5 a game)
HEAL_W, PICKUP_W, DEPOSIT_W, CARRY_PULL = .5, .1, .4, .2
RESCUE_W, RESCUE_FRAC = .5, .35           # extra for a heal that reaches an ally below 35% health
BUILD_W, WALL_W = .1, .1                 # base: per heart spent on a tank; per wall block raised (3 per heart)
# teamwork: every tank that hit an enemy in the 20 turns before it died gets as much as the
# killer; damage landed from behind is worth double (one tank holds attention, another flanks)
ASSIST_W, ASSIST_WINDOW = 1., 20
FLANK_W, FLANK_COS = .2, .3               # extra per damage when the bullet hits the target's rear half
# comrades: a soldier or heavy pays up to AWAY_W a turn for drifting away from the group --
# nothing while its second-nearest allied tank is within FORM_MAX, the full amount by
# FORM_MAX + AWAY_SCALE (scouts roam for hearts and are exempt). Being close is never
# penalised, only actually overlapping another tank (STACK_W)
FORM_MAX, AWAY_SCALE, AWAY_W = 12., 20., -.02   # at -.01 over 40 tiles only ~10% of turns were grouped
# ...and a comrade dying within COMRADE_NEAR stings a little (a heavy more): cover each other
COMRADE_NEAR, COMRADE_DEATH_W, COMMANDER_DEATH_W = 25., -.05, -.15
# no sitting out the war: a soldier or heavy that hasn't hit an enemy for IDLE_TURNS pays
# every turn, wherever it is (exempting base guards taught everyone to camp at home;
# real defenders earn DEFEND_W on intruders instead)
IDLE_TURNS, IDLE_W = 60, -.02          # as strong as AWAY_W, or groups just huddle; small next to death (-1)
BLOCK_STOP_W = .1                         # to a block's placer, per enemy move it stops near their base or them
STOP_NEAR = 20.                           # "near the placer"
# blocks: placing one costs a little, so scattered singles never pay; one that extends a
# wall of your team's blocks is free; and a block pays its placer every time it stops an
# enemy bullet or an enemy tank -- so barricades go up where the fighting is
BLOCK_PLACE_W, BLOCK_WALL_W, BLOCK_SAVE_W = -.15, .15, .25
TANK_SAVE_W = .5                          # per enemy bullet stopped by a block a tank placed (bases' walls get BLOCK_SAVE_W)
WIN_BONUS, TIMEOUT_BONUS = 3., 1.5          # enemy lost every base / had fewer bases at the limit

OBS = 4 + 8 + 2 + 2 + 4 * VISION_SECTORS + 7 * RADAR_SECTORS
FAR = 1e5
_maps = {}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _map(k, grid=GRID, bases=BASES):
    """Map k (deterministic) on a grid x grid board: scattered rectangles, 10 bases
    spread apart (BASE_GAP on the full board, scaled down with it) with open ground
    round each, and every pocket not connected to the big open area walled off.
    Returns (walls, base positions: first half team 0, second half team 1)."""
    if (k, grid, bases) in _maps:
        return _maps[k, grid, bases]
    f = grid / GRID
    gap = BASE_GAP * f * math.sqrt(BASES / bases)                # more bases -> proportionally closer
    rng = np.random.default_rng(k)
    while True:
        spots = []
        while len(spots) < 2 * bases:
            p = rng.uniform(30 * f, grid - 30 * f, 2)
            if all(np.hypot(*(p - q)) >= gap for q in spots):
                spots.append(p)
            elif rng.random() < .001:
                spots = []
        spots = np.array(spots)[rng.permutation(2 * bases)]
        walls = np.zeros((grid, grid), bool)
        wh = rng.uniform(3, 10, (int(N_RECTS * f * f), 2))
        xy = rng.uniform(2, grid - 2 - wh)
        for (x, y), (w, h) in zip(xy.astype(int), (xy + wh).astype(int)):
            walls[x:w + 1, y:h + 1] = True
        for bx, by in spots.astype(int):
            walls[bx - BASE_CLEAR:bx + BASE_CLEAR + 1, by - BASE_CLEAR:by + BASE_CLEAR + 1] = False
        labels, _ = ndimage.label(~walls)
        biggest = np.bincount(labels.ravel())[1:].argmax() + 1
        if all(labels[bx, by] == biggest for bx, by in spots.astype(int)):
            _maps[k, grid, bases] = (walls | (labels != biggest), spots.astype(np.float32))
            return _maps[k, grid, bases]


class Arena:
    def __init__(self, batch=1, device='cpu', seed=0, auto_reset=True, grid=GRID, tanks=sum(START), bases=BASES):
        """tanks: alive per side at the start (same scout/soldier/heavy mix as START);
        room to build up to 40/28 as many. bases: per side."""
        d = self.device = torch.device(device)
        self.B, self.auto_reset, self.grid = batch, auto_reset, grid
        mix = [round(tanks * s / sum(START)) for s in START]
        mix[1] += tanks - sum(mix)                                 # rounding goes to soldiers
        self.start, self.nb = tuple(mix), bases
        self.slots = max(tanks, round(tanks * TANKS / sum(START)))
        self.N = self.slots + bases
        self.A = 2 * self.N
        f = grid / GRID                                            # hearts and game length scale with the board
        self.H, self.limit = max(20, round(HEARTS * f * f)), round(LIMIT * f)
        torch.manual_seed(seed)
        ids = torch.randint(N_MAPS, (batch,), device=d)
        self.maps = ids.unique().tolist()                          # only build the maps in use
        self.map_id = torch.searchsorted(torch.tensor(self.maps, device=d), ids)
        self.walls = torch.stack([torch.from_numpy(_map(k, grid, bases)[0]) for k in self.maps]).to(d).reshape(-1)
        self.base_pos = torch.stack([torch.from_numpy(_map(k, grid, bases)[1]) for k in self.maps]).to(d)[self.map_id]
        self.team = torch.arange(self.A, device=d) // self.N
        self.is_base = torch.arange(self.A, device=d) % self.N >= self.slots
        self.same = self.team[:, None] == self.team[None]
        self.eye = torch.eye(self.A, dtype=torch.bool, device=d)
        self.base_slots = self.is_base.nonzero().squeeze(1)       # team 0's five, then team 1's
        self.vision_centers = (torch.arange(VISION_SECTORS, device=d) + .5) / VISION_SECTORS * 2 * VISION_SPAN - VISION_SPAN
        self.radar_centers = (torch.arange(RADAR_SECTORS, device=d) + .5) / RADAR_SECTORS * 2 * math.pi - math.pi
        self.vision_steps = torch.arange(1., VISION_RANGE + 1e-3, 1., device=d)
        self.radar_steps = torch.arange(2.5, RADAR_RANGE + 1e-3, 2.5, device=d)
        self.tables = {k: v.to(d) for k, v in dict(speed=SPEED, max_hp=MAX_HP, sight=SIGHT, carry=CARRY).items()}

        B = batch
        z = lambda *s, dt=torch.float32: torch.zeros(B, *s, dtype=dt, device=d)
        self.pos, self.heading, self.hp, self.supply = z(self.A, 2), z(self.A), z(self.A), z(self.A)
        self.gun_cd, self.special_cd = z(self.A), z(self.A)
        self.role = torch.zeros(B, self.A, dtype=torch.long, device=d)
        self.b_pos, self.b_vel = z(self.A, K, 2), z(self.A, K, 2)
        self.b_life, self.b_dmg = z(self.A, K), z(self.A, K)
        self.b_pierce, self.b_alive = z(self.A, K, dt=torch.bool), z(self.A, K, dt=torch.bool)
        self.heart_pos, self.heart_timer = z(self.H, 2), z(self.H)
        self.heart_alive = z(self.H, dt=torch.bool)
        self.spawn_pool = z(SPAWN_POOL, 2)
        self.BG = -(-grid // BLOCK)                                # placed-block grid, one cell per block
        # hp of the placed block in each cell (a block stands while hp > 0) and the agent who
        # placed it (only meaningful while it stands). One extra cell at the end soaks up the
        # writes of agents that aren't placing, so every update is in place, never a full copy
        self.blocks = z(self.BG * self.BG + 1)
        self.block_owner = torch.full((B, self.BG * self.BG + 1), -1, dtype=torch.int16, device=d)
        self.block_cd = z(self.A)
        self.phi = z(self.A)                                       # carry shaping potential
        self.born = z(self.A)                                      # turn each agent (re)appeared, for idleness
        self.last_hit = z(self.A, self.A)                          # [target, shooter] turn of the last enemy hit
        self.t = torch.zeros(B, dtype=torch.long, device=d)
        self.winner = torch.full((B,), -1, dtype=torch.long, device=d)
        self.reset_stats()
        self.reset_rows(torch.ones(B, dtype=torch.bool, device=d))
        # the ray marching is a long chain of elementwise ops over ~100M samples;
        # fusing it into one kernel is ~40x faster on the GPU than running it eagerly
        cuda = d.type == 'cuda'
        self.observe = torch.compile(self._observe, dynamic=False) if cuda else self._observe
        self._step_fn = torch.compile(self._step, dynamic=False) if cuda else self._step

    # role-dependent properties, looked up per agent
    speed = property(lambda s: s.tables['speed'][s.role])
    max_hp = property(lambda s: s.tables['max_hp'][s.role])
    sight = property(lambda s: s.tables['sight'][s.role])
    carry = property(lambda s: s.tables['carry'][s.role])

    # ---- setup -----------------------------------------------------------
    def reset_stats(self):
        if not hasattr(self, 'stats'):
            self.stats = {k: torch.zeros((), device=self.device) for k in (
                'shots', 'hits', 'misses', 'kills', 'base_kills', 'deaths', 'heals', 'pickups',
                'deposits', 'builds', 'repairs', 'games', 'decisive', 'timeouts', 'turns',
                'friendly_hits', 'stacked', 'bases_lost', 'blocks_placed', 'blocks_broken', 'assists',
                'grouped', 'rescues', 'block_saves', 'wall_blocks', 'base_walls', 'flank_hits', 'defend_hits',
                'block_stops', 'idle')}
        for v in self.stats.values():
            v.zero_()

    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device)

    def _random_open(self, rows, n):
        """n random open positions on each listed game's map: resample anything that
        landed in a wall a few times (about 1 in 10 does, so 1 in 10^6 is left)."""
        pos = self._rand(len(rows), n, 2) * (self.grid - 2) + 1
        for _ in range(6):
            bad = self._walls_at(pos, rows)
            pos = torch.where(bad.unsqueeze(-1), self._rand(len(rows), n, 2) * (self.grid - 2) + 1, pos)
        return pos

    def _near(self, centre, radius):
        ang = self._rand(*centre.shape[:-1]) * 2 * math.pi
        r = self._rand(*centre.shape[:-1]) * radius
        return centre + torch.stack((ang.cos(), ang.sin()), -1) * r.unsqueeze(-1)

    def reset_rows(self, mask):
        rows = mask.nonzero().squeeze(1)
        if len(rows) == 0:
            return
        R = len(rows)
        role = torch.full((self.A,), SOLDIER, dtype=torch.long, device=self.device)
        start = torch.tensor([SCOUT] * self.start[0] + [SOLDIER] * self.start[1] + [COMMANDER] * self.start[2], device=self.device)
        for team in range(2):
            role[team * self.N:team * self.N + len(start)] = start
        role[self.is_base] = BASE
        self.role[rows] = role
        # every tank starts in the open ground round one of its team's bases (dealt out in turn)
        local = torch.arange(self.A, device=self.device) % self.N
        home = self.team * self.nb + local % self.nb
        pos = self._near(self.base_pos[rows][:, home], 10.)
        pos[:, self.base_slots] = self.base_pos[rows]
        self.pos[rows] = pos
        self.heading[rows] = (self._rand(R, self.A) * 2 - 1) * math.pi
        alive = torch.zeros(self.A, dtype=torch.bool, device=self.device)
        for team in range(2):
            alive[team * self.N:team * self.N + len(start)] = True
        alive |= self.is_base
        self.hp[rows] = self.tables['max_hp'][role] * alive
        for x in (self.supply, self.gun_cd, self.special_cd, self.heart_timer, self.blocks, self.block_cd, self.phi, self.born):
            x[rows] = 0
        self.last_hit[rows] = -1e9
        self.block_owner[rows] = -1
        self.b_alive[rows] = False
        self.heart_pos[rows] = self._random_open(rows, self.H)
        self.spawn_pool[rows] = self._random_open(rows, SPAWN_POOL)
        self.heart_alive[rows] = True
        self.t[rows] = 0

    # ---- geometry ----------------------------------------------------------
    def _walls_at(self, xy, rows=None):
        c = xy.floor().to(torch.int32)
        G = self.grid
        inside = ((c >= 0) & (c < G)).all(-1)
        c = c.clamp(0, G - 1)
        m = (self.map_id if rows is None else self.map_id[rows]).view(-1, *[1] * (xy.dim() - 2)).to(torch.int32)
        return self.walls[m * G * G + c[..., 0] * G + c[..., 1]] | ~inside

    def _blocks_at(self, xy):
        """Is there a placed block at each point? xy is (B, ..., 2)."""
        c = (xy / BLOCK).floor().to(torch.int32).clamp(0, self.BG - 1)
        flat = (c[..., 0] * self.BG + c[..., 1]).long()
        return torch.gather(self.blocks, 1, flat.reshape(self.B, -1)).view(flat.shape) > 0

    def _solid_at(self, xy):
        return self._walls_at(xy) | self._blocks_at(xy)

    def _rays(self, angle, steps, scale):
        """Distance along each ray to the first permanent wall and to the first placed
        block (ray length scales with the agent's sight); the ray's full length if none."""
        u = torch.stack((angle.cos(), angle.sin()), -1)
        reach = steps.view(1, 1, 1, -1) * scale[..., None, None]
        pts = self.pos[:, :, None, None] + u.unsqueeze(-2) * reach.unsqueeze(-1)
        def first(hit):
            i = torch.where(hit.any(-1), hit.float().argmax(-1), torch.full_like(hit[..., 0], len(steps) - 1, dtype=torch.long))
            return steps[i] * scale[..., None]
        return first(self._walls_at(pts)), first(self._blocks_at(pts))

    @staticmethod
    def _sector_min(idx, value, valid, sectors):
        v = torch.where(valid, value, torch.full_like(value, FAR))
        out = torch.full((*value.shape[:-1], sectors), FAR, device=value.device)
        return out.scatter_reduce(-1, idx.clamp(0, sectors - 1), v, 'amin')

    def _radar_idx(self, rel):
        return ((rel + math.pi) / (2 * math.pi) * RADAR_SECTORS).long()

    def _vision_idx(self, rel):
        return ((rel + VISION_SPAN) / (2 * VISION_SPAN) * VISION_SECTORS).floor().long()

    # ---- observation --------------------------------------------------------
    def _observe(self):
        B, d = self.B, self.device
        alive = self.hp > 0
        sight = self.sight
        radar_r, vision_r = RADAR_RANGE * sight, VISION_RANGE * sight
        delta = self.pos[:, None] - self.pos[:, :, None]           # (B,A,A,2): j seen from i
        dist = delta.norm(dim=-1)
        rel = wrap(torch.atan2(delta[..., 1], delta[..., 0]) - self.heading[..., None])
        seen = alive[:, None] & ~self.eye
        friend, enemy = seen & self.same, seen & ~self.same
        tank_j = ~self.is_base.view(1, 1, self.A)

        # radar: long range, through walls
        r_idx = self._radar_idx(rel)
        in_range = dist < radar_r[..., None]
        r_wall, r_block = self._rays(self.heading[..., None] + self.radar_centers, self.radar_steps, sight)
        r_friend = self._sector_min(r_idx, dist, friend & tank_j & in_range, RADAR_SECTORS)
        r_enemy = self._sector_min(r_idx, dist, enemy & tank_j & in_range, RADAR_SECTORS)
        need = (1 - self.hp / self.max_hp)[:, None].expand(-1, self.A, -1)
        r_need = torch.zeros(B, self.A, RADAR_SECTORS, device=d).scatter_reduce(
            -1, r_idx.clamp(0, RADAR_SECTORS - 1), torch.where(friend & tank_j & in_range, need, torch.zeros_like(need)), 'amax')
        r_own = self._sector_min(r_idx, dist, friend & ~tank_j, RADAR_SECTORS)
        r_foe = self._sector_min(r_idx, dist, enemy & ~tank_j, RADAR_SECTORS)

        # vision: forward cone, stops at the first wall or placed block
        v_wall, v_block = self._rays(self.heading[..., None] + self.vision_centers, self.vision_steps, sight)
        v_block = torch.where(v_block < v_wall, v_block, torch.full_like(v_block, FAR))
        v_stop = v_wall.minimum(v_block)
        v_idx = self._vision_idx(rel)
        cone = (rel.abs() < VISION_SPAN) & (dist < vision_r[..., None])
        v_enemy = self._sector_min(v_idx, dist, enemy & cone, VISION_SECTORS)
        h_delta = self.heart_pos[:, None] - self.pos[:, :, None]      # (B,A,H,2)
        h_dist = h_delta.norm(dim=-1)
        h_rel = wrap(torch.atan2(h_delta[..., 1], h_delta[..., 0]) - self.heading[..., None])
        h_ok = self.heart_alive[:, None] & (h_rel.abs() < VISION_SPAN) & (h_dist < vision_r[..., None])
        v_heart = self._sector_min(self._vision_idx(h_rel), h_dist, h_ok, VISION_SECTORS)
        v_enemy = torch.where(v_enemy < v_stop, v_enemy, torch.full_like(v_enemy, FAR))
        v_heart = torch.where(v_heart < v_stop, v_heart, torch.full_like(v_heart, FAR))

        by_team = alive.view(B, 2, self.N)
        tanks_alive = by_team[..., :self.slots].float().sum(-1) / sum(self.start)
        bases_alive = by_team[..., self.slots:].float().mean(-1)
        is_b = self.role == BASE
        clip = lambda x, r: (x / r.unsqueeze(-1)).clamp(max=1)
        me = torch.stack((self.hp / self.max_hp, self.supply / self.carry,
                          self.gun_cd / torch.where(is_b, BASE_GUN_CD, GUN_CD),
                          self.special_cd / torch.where(is_b, ORDER_CD, MISSILE_CD),
                          tanks_alive[:, self.team], tanks_alive[:, 1 - self.team],
                          bases_alive[:, self.team], bases_alive[:, 1 - self.team]), -1)
        diag = torch.full_like(sight, DIAG)
        return torch.cat((
            torch.nn.functional.one_hot(self.role, 4).float(), me,
            self.pos / self.grid, torch.stack((self.heading.cos(), self.heading.sin()), -1),
            clip(v_wall, vision_r), clip(v_block, vision_r), clip(v_enemy, vision_r), clip(v_heart, vision_r),
            clip(r_wall, radar_r), clip(r_block, radar_r), clip(r_friend, radar_r), clip(r_enemy, radar_r), r_need,
            clip(r_own, diag), clip(r_foe, diag),
        ), -1)

    # ---- dynamics -----------------------------------------------------------
    def step(self, action):
        """action (B,A,6) = throttle, steer in [-1,1]; fire, special as >0.5; base
        order index (see ORDERS); place a block as >0.5. Returns per-agent reward,
        per-agent terminal flag, per-game done, winner."""
        with torch.no_grad():
            out = self._step_fn(action)
            if self.auto_reset:
                self.reset_rows(out[2])
            return out

    def _step(self, action):
        B, d = self.B, self.device
        alive0 = self.hp > 0
        hp0 = self.hp.clone()
        throttle, steer, fire, special, order, place = action.unbind(-1)
        role = self.role
        is_b = role == BASE

        self.heading = torch.where(alive0, wrap(self.heading + steer.clamp(-1, 1) * TURN), self.heading)
        u = torch.stack((self.heading.cos(), self.heading.sin()), -1)
        prop = (self.pos + u * (throttle.clamp(-1, 1) * self.speed).unsqueeze(-1)).clamp(.5, self.grid - .5)
        blocked = self._solid_at(prop)
        wanted, bumped = prop, blocked.clone()        # where it tried to go, before sliding along the obstacle
        for axis in range(2):                       # slide along a wall instead of stopping dead
            alt = self.pos.clone()
            alt[..., axis] = prop[..., axis]
            ok = blocked & ~self._solid_at(alt)
            prop = torch.where(ok.unsqueeze(-1), alt, prop)
            blocked &= ~ok
        # a placed block that stops an enemy tank's move near the placer's base (or near
        # the placer) earns its placer something: blocks that hold a line, not just stand
        foe_base = ~self.same & (is_b & alive0)[:, None]
        intruding = torch.where(foe_base, torch.cdist(self.pos, self.pos), FAR).amin(-1) < DEFEND_RADIUS
        c = (wanted / BLOCK).floor().long().clamp(0, self.BG - 1)
        owner = self.block_owner.gather(1, c[..., 0] * self.BG + c[..., 1]).long()
        near_owner = (self.pos.gather(1, owner.clamp(min=0).unsqueeze(-1).expand(-1, -1, 2)) - self.pos).norm(dim=-1) < STOP_NEAR
        stop = (bumped & alive0 & ~is_b & (throttle != 0) & self._blocks_at(wanted) & ~self._walls_at(wanted)
                & (owner >= 0) & (owner // self.N != self.team) & (intruding | near_owner))
        block_stops = torch.zeros(B, self.A, device=d).scatter_add(1, owner.clamp(min=0), stop.float())
        self.pos = torch.where((alive0 & ~blocked).unsqueeze(-1), prop, self.pos)
        self.gun_cd = (self.gun_cd - 1).clamp(min=0)
        self.special_cd = (self.special_cd - 1).clamp(min=0)
        self.heart_timer = (self.heart_timer - 1).clamp(min=0)
        self.block_cd = (self.block_cd - 1).clamp(min=0)
        want = alive0 & ~is_b & (place > .5) & (self.block_cd == 0)
        counts = self._block_counts()
        placed, walled, counts = self._put_blocks(want, self.pos + u * BLOCK_REACH, counts)
        self.block_cd = torch.where(placed > 0, float(BLOCK_CD), self.block_cd)

        can_fire = alive0 & (fire > .5) & (self.gun_cd == 0)
        shoot = self._spawn(can_fire & ((role == SOLDIER) | (role == COMMANDER)), u, BULLET_SPEED, BULLET_LIFE, 1., False)
        big = self._spawn(can_fire & is_b, u, BASE_MISSILE_SPEED, BASE_MISSILE_LIFE, 2., True)
        self.gun_cd = torch.where(shoot, float(GUN_CD), torch.where(big, float(BASE_GUN_CD), self.gun_cd))
        launch = alive0 & (role == COMMANDER) & (special > .5) & (self.special_cd == 0)
        launch = self._spawn(launch, u, MISSILE_SPEED, MISSILE_LIFE, 2., True)
        self.special_cd = torch.where(launch, float(MISSILE_CD), self.special_cd)

        picked = self._pickup(alive0)
        deposited, healed, rescued = self._scout_special(alive0, special)
        built, spent, repaired, base_walls = self._base_orders(alive0, order, u, counts)
        dealt, friendly_dmg, kills, base_kills, friendly_kills, misses, saves, flank, defend = self._bullets(alive0, intruding)
        died = alive0 & (self.hp <= 0)
        # assists: everyone who landed a hit on an enemy in the last ASSIST_WINDOW turns
        # before it died -- the credit that makes focusing fire together pay
        recent = (self.t.view(B, 1, 1) - self.last_hit) <= ASSIST_WINDOW
        assists = (died.unsqueeze(-1) & recent).sum(1).float()

        respawn = ~self.heart_alive & (self.heart_timer == 0)
        fresh = self.spawn_pool.gather(1, (self._rand(B, self.H) * SPAWN_POOL).long().unsqueeze(-1).expand(-1, -1, 2))
        self.heart_pos = torch.where(respawn.unsqueeze(-1), fresh, self.heart_pos)
        self.heart_alive |= respawn

        bases_lost = (died & is_b).view(B, 2, self.N).sum(-1).float()
        # overlapping another tank (either side): only actual overlap, not merely being close
        tank = (self.hp > 0) & ~is_b
        dd = torch.cdist(self.pos, self.pos)
        gap = dd.masked_fill(~(tank[:, None] & tank[:, :, None]) | self.eye, FAR)     # between living tanks
        overlap = (1 - gap.amin(-1) / STACK_RADIUS).clamp(min=0) * tank
        fighter = tank & (role != SCOUT)
        second = gap.masked_fill(~self.same, FAR).topk(2, -1, largest=False).values[..., 1]
        away = ((second - FORM_MAX) / AWAY_SCALE).clamp(0, 1) * fighter
        fallen = torch.where(role == COMMANDER, COMMANDER_DEATH_W, COMRADE_DEATH_W) * (died & ~is_b)
        comrade_loss = torch.bmm(((dd < COMRADE_NEAR) & self.same & ~self.eye).float(), fallen.unsqueeze(-1)).squeeze(-1) * tank
        # carry shaping: hearts held x closeness to the nearest friendly base
        home = torch.where(self.same & (is_b & (self.hp > 0))[:, None], dd, torch.full_like(dd, FAR)).amin(-1)
        last_attack = torch.maximum(self.last_hit.amax(1), self.born)
        idle = fighter & (self.t.view(B, 1) - last_attack > IDLE_TURNS)
        phi = CARRY_PULL * self.supply * (1 - home / self.grid) * (tank & (role == SCOUT))
        pull, self.phi = phi - self.phi, phi
        reward = (KILL_W * kills + BASE_KILL_W * base_kills + DAMAGE_W * dealt + MISS_W * misses
                  + FRIENDLY_DAMAGE_W * friendly_dmg + FRIENDLY_KILL_W * friendly_kills
                  + torch.where(is_b, BASE_DEATH_W, DEATH_W) * died
                  + BASE_LOST_W * bases_lost[:, self.team] + BASE_WON_W * bases_lost[:, 1 - self.team]
                  + STACK_W * overlap + AWAY_W * away + comrade_loss + IDLE_W * idle + BLOCK_STOP_W * block_stops
                  + ASSIST_W * assists + FLANK_W * flank + DEFEND_W * defend
                  + BLOCK_PLACE_W * placed + BLOCK_WALL_W * walled + torch.where(is_b, BLOCK_SAVE_W, TANK_SAVE_W) * saves
                  + HEAL_W * healed + RESCUE_W * rescued + PICKUP_W * picked + DEPOSIT_W * deposited + pull
                  + BUILD_W * spent + WALL_W * base_walls)
        live = alive0.view(B, 2, self.N)
        team_mean = (reward.view(B, 2, self.N) * live).sum(-1) / live.sum(-1).clamp(min=1)
        reward = (1 - TEAM_SPIRIT) * reward + TEAM_SPIRIT * team_mean[:, self.team]
        self.stats['stacked'] += (overlap > 0).sum()
        self.stats['grouped'] += (fighter & (second <= FORM_MAX)).sum()
        self.stats['rescues'] += rescued.sum()
        self.stats['assists'] += assists.sum()
        self.stats['blocks_placed'] += placed.sum()
        self.stats['wall_blocks'] += walled.sum()
        self.stats['base_walls'] += base_walls.sum()
        self.stats['flank_hits'] += (flank > 0).sum()
        self.stats['defend_hits'] += (defend > 0).sum()
        self.stats['block_stops'] += block_stops.sum()
        self.stats['idle'] += idle.sum()
        self.stats['bases_lost'] += bases_lost.sum()

        self.t += 1
        by_team = (self.hp > 0).view(B, 2, self.N)
        wiped = ~by_team[..., self.slots:].any(-1)                # lost every base
        timeout = self.t >= self.limit
        done = wiped.any(-1) | timeout
        winner = torch.full_like(self.t, -1)
        winner = torch.where(wiped[:, 1] & ~wiped[:, 0], 0, winner)
        winner = torch.where(wiped[:, 0] & ~wiped[:, 1], 1, winner)
        by_count = timeout & ~wiped.any(-1)
        score = by_team[..., self.slots:].sum(-1) * 100 + by_team[..., :self.slots].sum(-1)   # bases first, then tanks
        winner = torch.where(by_count & (score[:, 0] > score[:, 1]), 0, winner)
        winner = torch.where(by_count & (score[:, 1] > score[:, 0]), 1, winner)
        size = torch.where(by_count, TIMEOUT_BONUS, WIN_BONUS)
        outcome = torch.where(winner.unsqueeze(1) == self.team, 1., -LOSS_SCALE) * size.unsqueeze(1)
        outcome = torch.where((winner == -1).unsqueeze(1), torch.zeros_like(outcome), outcome)
        reward = (reward + torch.where(done.unsqueeze(1), outcome, torch.zeros_like(outcome))) * alive0
        terminal = alive0 & (died | done.unsqueeze(1))
        self.winner = winner

        s = self.stats
        s['shots'] += shoot.sum() + big.sum() + launch.sum()
        s['kills'] += kills.sum()
        s['base_kills'] += base_kills.sum()
        s['misses'] += misses.sum()
        s['deaths'] += died.sum()
        s['heals'] += (healed > 0).sum()
        s['pickups'] += picked.sum()
        s['deposits'] += deposited.sum()
        s['builds'] += built.sum()
        s['repairs'] += repaired.sum()
        s['games'] += done.sum()
        s['decisive'] += (done & (winner >= 0) & ~by_count).sum()
        s['timeouts'] += (done & timeout).sum()
        s['turns'] += (self.t * done).sum()
        return reward, terminal, done, winner

    def _spawn(self, want, u, speed, life, dmg, pierce):
        free = ~self.b_alive
        want = want & free.any(-1)
        slot = free.float().argmax(-1)
        put = want.unsqueeze(-1) & (torch.arange(K, device=self.device) == slot.unsqueeze(-1))
        self.b_pos = torch.where(put.unsqueeze(-1), self.pos.unsqueeze(2), self.b_pos)
        self.b_vel = torch.where(put.unsqueeze(-1), (u * speed).unsqueeze(2), self.b_vel)
        self.b_life = torch.where(put, float(life), self.b_life)
        self.b_dmg = torch.where(put, float(dmg), self.b_dmg)
        self.b_pierce = torch.where(put, torch.full_like(self.b_pierce, pierce), self.b_pierce)
        self.b_alive |= put
        return want

    def _block_counts(self):
        """Standing blocks per team: (all of them, just the tanks'), each (B, 2). One pass
        over the block grid per turn -- recounting on every placement made the viewer crawl."""
        standing, own = self.blocks[:, :-1] > 0, self.block_owner[:, :-1]
        t1, by_tank = own >= self.N, own % self.N < self.slots
        split = lambda m: torch.stack(((m & ~t1).sum(-1), (m & t1).sum(-1)), -1)
        return split(standing), split(standing & by_tank)

    def _put_blocks(self, want, xy, counts):
        """Each agent that wants to drops a block into the cell at xy -- if that cell is
        empty ground, holds no block, nobody is standing in it, and its team is under
        its caps. Returns (placed, placed touching one of the team's own blocks, counts)."""
        cell = (xy / BLOCK).floor()
        inside = ((cell >= 0) & (cell < self.BG)).all(-1)
        cell = cell.clamp(0, self.BG - 1)
        corners = torch.tensor([[.5, .5], [1.5, .5], [.5, 1.5], [1.5, 1.5]], device=self.device)
        wall = self._walls_at(cell.unsqueeze(-2) * BLOCK + corners).any(-1)
        flat = (cell[..., 0] * self.BG + cell[..., 1]).long()
        taken = self.blocks.gather(1, flat) > 0
        centre = (cell + .5) * BLOCK
        crowded = (torch.cdist(centre, self.pos) < 1.6).logical_and((self.hp > 0)[:, None]).any(-1)
        total, tanks = counts
        team = self.team.view(1, -1).expand(self.B, -1)
        room = (total.gather(1, team) < MAX_BLOCKS) & (self.is_base | (tanks.gather(1, team) < TANK_BLOCKS))
        ok = want & inside & ~wall & ~taken & ~crowded & room
        # does it extend a wall, i.e. touch one of our own blocks (4-neighbours)?
        adjacent = torch.zeros_like(ok)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = cell + torch.tensor([dx, dy], device=self.device)
            inb = ((nb >= 0) & (nb < self.BG)).all(-1)
            nf = (nb.clamp(0, self.BG - 1)[..., 0] * self.BG + nb.clamp(0, self.BG - 1)[..., 1]).long()
            adjacent |= inb & (self.blocks.gather(1, nf) > 0) & (self.block_owner.gather(1, nf) // self.N == self.team)
        put = torch.where(ok, flat, self.BG * self.BG)            # everyone else writes to the spare cell
        self.blocks.scatter_(1, put, BLOCK_HP)
        self.block_owner.scatter_(1, put, torch.arange(self.A, device=self.device, dtype=torch.int16).expand(self.B, -1))
        per_team = lambda m: m.view(self.B, 2, self.N).sum(-1)
        counts = (total + per_team(ok), tanks + per_team(ok & ~self.is_base))
        return ok.float(), (ok & adjacent).float(), counts

    def _bullets(self, alive0, intruding):
        """Advance every bullet one step, test the whole swept segment against every
        agent but the shooter (friendly fire is on), and credit damage, kills and
        misses -- split into enemy and friendly -- to the agent that fired it."""
        B = self.B
        old, new = self.b_pos, self.b_pos + self.b_vel
        mid = (old + new) / 2
        out = ((new < 0) | (new >= self.grid)).any(-1)
        wm, wn, bm, bn = self._walls_at(mid), self._walls_at(new), self._blocks_at(mid), self._blocks_at(new)
        wall = (wm | (~bm & wn)) & ~self.b_pierce                   # whichever comes first along the path
        block = ((~wm & bm) | (~wm & ~bm & ~wn & bn)) & ~self.b_pierce
        seg = new - old
        rel = self.pos[:, None, None] - old.unsqueeze(3)                   # (B,A,K,A,2)
        tt = ((rel * seg.unsqueeze(3)).sum(-1) / (seg * seg).sum(-1, keepdim=True).clamp(min=1e-6)).clamp(0, 1)
        gap = (rel - tt.unsqueeze(-1) * seg.unsqueeze(3)).norm(dim=-1)
        radius = torch.where(self.role == BASE, BASE_HIT_RADIUS, HIT_RADIUS)[:, None, None]
        target_ok = alive0[:, None, None] & ~self.eye[None, :, None]
        gap = torch.where(target_ok, gap - radius, torch.full_like(gap, FAR))
        mind, tgt = gap.min(-1)
        hit = self.b_alive & (mind < 0)
        dmg = hit * self.b_dmg
        self.hp = (self.hp - torch.zeros(B, self.A, device=self.device).scatter_add(1, tgt.view(B, -1), dmg.view(B, -1))).clamp(min=0)
        died = alive0 & (self.hp <= 0)
        killed = hit & died.gather(1, tgt.view(B, -1)).view_as(hit)
        base_hit = (self.role == BASE).gather(1, tgt.view(B, -1)).view_as(hit)
        friendly = self.same[torch.arange(self.A, device=self.device).view(1, self.A, 1), tgt]
        # flank: the bullet travels the way the target faces, i.e. it hits the target from behind
        th = self.heading.gather(1, tgt.view(B, -1)).view_as(hit)
        rear = (self.b_vel[..., 0] * th.cos() + self.b_vel[..., 1] * th.sin()) > FLANK_COS * self.b_vel.norm(dim=-1)
        flank = (dmg * (rear & ~friendly & ~base_hit)).sum(-1)
        defend = (dmg * (intruding.gather(1, tgt.view(B, -1)).view_as(hit) & ~friendly)).sum(-1)
        # remember who last hit whom (enemy hits only), for assists
        pair = torch.nn.functional.one_hot(tgt, self.A).bool() & (hit & ~friendly).unsqueeze(-1)   # (B,shooter,K,target)
        self.last_hit = torch.where(pair.any(2).transpose(1, 2), self.t.view(B, 1, 1).float(), self.last_hit)
        # bullets that reach a placed block before anything else chip it; 0 hp and it's gone
        chip = self.b_alive & ~hit & block
        at = ((torch.where(bm.unsqueeze(-1), mid, new) / BLOCK).floor().clamp(0, self.BG - 1))
        at = (at[..., 0] * self.BG + at[..., 1]).long().view(B, -1)
        owner = self.block_owner.gather(1, at).long()                            # (B, A*K)
        shooter_team = self.team.view(1, self.A, 1).expand(B, -1, K).reshape(B, -1)
        mine = (owner >= 0) & (owner // self.N == shooter_team)
        stopped = chip.view(B, -1) & (owner >= 0) & ~mine                         # enemy fire soaked up
        saves = torch.zeros(B, self.A, device=self.device).scatter_add(1, owner.clamp(min=0), stopped.float())
        hp_before = self.blocks.gather(1, at)
        self.blocks.scatter_add_(1, at, -(chip * self.b_dmg).view(B, -1))
        self.stats['blocks_broken'] += (chip.view(B, -1) & (hp_before > 0) & (self.blocks.gather(1, at) <= 0)).sum()
        self.stats['block_saves'] += stopped.sum()
        own_block = chip & mine.view_as(chip)                                     # shooting your own wall is a wasted shot
        gone = self.b_alive & ~hit & (~chip | own_block) & (wall | out | (self.b_life <= 1) | own_block)
        self.stats['hits'] += (hit & ~friendly).sum()
        self.stats['friendly_hits'] += (hit & friendly).sum()
        self.b_pos, self.b_life = new, self.b_life - 1
        self.b_alive &= ~hit & ~chip & ~wall & ~out & (self.b_life > 0)
        foe_kill = killed & ~friendly
        return ((dmg * ~friendly).sum(-1), (dmg * friendly).sum(-1),
                (foe_kill & ~base_hit).sum(-1).float(), (foe_kill & base_hit).sum(-1).float(),
                (killed & friendly).sum(-1).float(), gone.sum(-1).float(), saves, flank, defend)

    def _pickup(self, alive0):
        scout = alive0 & (self.role == SCOUT) & (self.supply < self.carry)
        dh = torch.cdist(self.pos, self.heart_pos)
        dh = torch.where(self.heart_alive[:, None] & scout[..., None], dh, torch.full_like(dh, FAR))
        mind, h = dh.min(-1)
        can = mind < PICK_RADIUS
        best = torch.full((self.B, self.H), FAR, device=self.device).scatter_reduce(
            1, h, torch.where(can, mind, torch.full_like(mind, FAR)), 'amin')
        won = can & (mind <= best.gather(1, h))
        taken = torch.zeros(self.B, self.H, device=self.device).scatter_add(1, h, won.float()) > 0
        self.supply += won.float()
        self.heart_alive &= ~taken
        self.heart_timer = torch.where(taken, float(HEART_RESPAWN), self.heart_timer)
        return won.float()

    def _scout_special(self, alive0, special):
        """A scout's special: at a friendly base, unload every carried heart into it;
        anywhere else, spend one on the nearest hurt teammate."""
        B = self.B
        act = alive0 & (self.role == SCOUT) & (special > .5) & (self.supply > 0) & (self.special_cd == 0)
        dd = torch.cdist(self.pos, self.pos)
        base_ok = self.same & (self.role == BASE)[:, None] & (self.hp > 0)[:, None]
        bd, which = torch.where(base_ok, dd, torch.full_like(dd, FAR)).min(-1)
        unload = act & (bd < DEPOSIT_RADIUS)
        room = (self.carry - self.supply).gather(1, which)
        amount = torch.where(unload, self.supply.minimum(room), torch.zeros_like(self.supply))
        self.supply = (self.supply + torch.zeros_like(self.supply).scatter_add(1, which, amount) - amount).minimum(self.carry)

        heal = act & ~unload
        hurt = (self.hp > 0) & (self.hp < self.max_hp) & (self.role != BASE)
        md, who = torch.where(self.same & ~self.eye & hurt[:, None], dd, torch.full_like(dd, FAR)).min(-1)
        heal &= md < HEAL_RADIUS
        room = (self.max_hp - self.hp).gather(1, who)
        rescued = (heal & ((self.hp / self.max_hp).gather(1, who) < RESCUE_FRAC)).float()
        given = torch.where(heal, room.clamp(max=HEAL_AMOUNT), torch.zeros_like(room))
        self.hp = (self.hp + torch.zeros_like(self.hp).scatter_add(1, who, given)).minimum(self.max_hp)
        self.supply -= heal.float()
        self.special_cd = torch.where(heal | unload, float(HEAL_CD), self.special_cd)
        return amount, given, rescued

    def _base_orders(self, alive0, order, u, counts):
        """Bases spend stored hearts: repair (themselves and every friendly tank close
        by), build a new tank into one of the team's empty slots, next to the base (at
        most one build per team per turn), or raise a wall: 1 heart for a row of 3
        blocks, WALL_RADIUS out in the direction the base's turret points."""
        B, d = self.B, self.device
        ready = alive0 & (self.role == BASE) & (self.special_cd == 0)
        order = order.long()
        repair = ready & (order == 1) & (self.supply >= 1)
        near = torch.cdist(self.pos, self.pos) < DEPOSIT_RADIUS
        boost = (repair[:, :, None] & near & self.same).any(1) & (self.hp > 0) & (self.role != BASE)
        self.hp = torch.where(repair, self.hp + REPAIR_SELF, torch.where(boost, self.hp + REPAIR_NEAR, self.hp)).minimum(self.max_hp)
        self.supply -= repair.float()

        kind = (order - 2).clamp(0, 2)                          # 0 scout, 1 soldier, 2 heavy
        cost = torch.tensor([BUILD_COST[SCOUT], BUILD_COST[SOLDIER], BUILD_COST[COMMANDER]], device=d, dtype=torch.float32)[kind]
        want = ready & (order >= 2) & (self.supply >= cost)
        built, spent = torch.zeros(B, self.A, device=d), torch.zeros(B, self.A, device=d)
        idx = torch.arange(self.A, device=d)
        for team in range(2):                                    # masked writes: no data-dependent shapes
            lo = team * self.N
            wants = want[:, lo + self.slots:lo + self.N]
            empty = self.hp[:, lo:lo + self.slots] <= 0
            go = wants.any(-1) & empty.any(-1)
            base = (lo + self.slots + wants.float().argmax(-1)).unsqueeze(1)
            slot = go.unsqueeze(1) & (idx == lo + empty.float().argmax(-1).unsqueeze(1))
            at_base = go.unsqueeze(1) & (idx == base)
            k = kind.gather(1, base)
            here = self._near(self.pos.gather(1, base.unsqueeze(-1).expand(-1, -1, 2)), 8.)
            self.role = torch.where(slot, k, self.role)
            self.hp = torch.where(slot, self.tables['max_hp'][k], self.hp)
            self.pos = torch.where(slot.unsqueeze(-1), here, self.pos)
            self.heading = torch.where(slot, (self._rand(B, 1) * 2 - 1) * math.pi, self.heading)
            self.born = torch.where(slot, self.t.view(B, 1).float(), self.born)
            for x in ('supply', 'gun_cd', 'special_cd', 'block_cd'):
                setattr(self, x, torch.where(slot, 0., getattr(self, x)))
            self.supply = self.supply - at_base * cost
            built = built + at_base.float()
            spent = spent + at_base * cost

        wall = ready & (order == len(ORDERS) - 1) & (self.supply >= WALL_COST)
        perp = torch.stack((-u[..., 1], u[..., 0]), -1)
        raised = torch.zeros(B, self.A, device=d)
        for k in range(WALL_WIDTH):
            xy = self.pos + u * WALL_RADIUS + perp * (k - (WALL_WIDTH - 1) / 2) * BLOCK
            ok, _, counts = self._put_blocks(wall, xy, counts)
            raised = raised + ok
        wall = wall & (raised > 0)                               # nothing placed, nothing paid
        self.supply = self.supply - wall * WALL_COST
        self.special_cd = torch.where(repair | (built > 0) | wall, float(ORDER_CD), self.special_cd)
        return built, spent, repair.float(), raised

    # ---- scripted benchmark ---------------------------------------------------
    def raider(self):
        """Fixed opponent that sees everything. Soldiers and commanders drive at the
        nearest enemy and fire when lined up; scouts fetch hearts and take them to the
        nearest friendly base; bases shoot what's in range and build soldiers."""
        alive = self.hp > 0
        dd = torch.cdist(self.pos, self.pos)
        foe_d, foe = torch.where(alive[:, None] & ~self.same, dd, torch.full_like(dd, FAR)).min(-1)
        home = torch.where(alive[:, None] & self.same & (self.role == BASE)[:, None], dd, torch.full_like(dd, FAR)).argmin(-1)
        heart = torch.where(self.heart_alive[:, None], torch.cdist(self.pos, self.heart_pos), FAR).argmin(-1)
        scout = self.role == SCOUT
        target = self.pos.gather(1, foe[..., None].expand(-1, -1, 2))
        fetch = self.heart_pos.gather(1, heart[..., None].expand(-1, -1, 2))
        drop = self.pos.gather(1, home[..., None].expand(-1, -1, 2))
        target = torch.where((scout & (self.supply < 1)).unsqueeze(-1), fetch, target)
        target = torch.where((scout & (self.supply >= 1)).unsqueeze(-1), drop, target)
        delta = target - self.pos
        dist = delta.norm(dim=-1)
        bearing = wrap(torch.atan2(delta[..., 1], delta[..., 0]) - self.heading)
        base = self.role == BASE
        reach = torch.where(base, BASE_MISSILE_SPEED * BASE_MISSILE_LIFE, VISION_RANGE)
        throttle = ((dist > 4) | scout).float()
        fire = ((bearing.abs() < .12) & (dist < reach)).float()
        order = torch.where(base & (self.supply >= BUILD_COST[SOLDIER]), 3., 0.)
        return torch.stack((throttle, (bearing * 2).clamp(-1, 1), fire, torch.ones_like(fire), order, torch.zeros_like(fire)), -1)
