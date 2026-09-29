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
              first permanent wall and placed block on the sector's centre ray, the
              nearest friendly and enemy tank, how hurt the neediest friendly tank is,
              the nearest friendly and enemy base (at any distance), how many enemy
              tanks are closing on a friendly base there, and how damaged the enemy
              base there is -- where to defend and where to attack -- and the
              nearest heart, so scouts can find what they're for.
  * vision -- 9 sectors in a forward cone, blocked by walls. Per sector: the wall,
              placed block, nearest enemy (tank or base) and heart in front of it.
"""
import math
import os
import numpy as np
import torch
from scipy import ndimage

# torch.compile has Triton write compiled kernels to ~/.triton/cache by default; on this
# machine that folder is owned by root, which crashes the compile, so use one we own
os.environ.setdefault('TRITON_CACHE_DIR', os.path.expanduser('~/.cache/tank_arena/triton'))

GRID, LIMIT = 1125, 1500                  # full board (the viewer) and turn limit; training uses a smaller board
TANKS, START, BASES = 40, (6, 18, 4), 5    # defaults per team: tank slots, scouts/soldiers/heavies at start, bases
N_MAPS, N_RECTS, BASE_GAP, BASE_CLEAR = 16, 3440, 250, 15
STYLES = ('rubble', 'boulders', 'corridors', 'open')   # map k is drawn in style k % 4

ROLE_NAMES = ('scout', 'soldier', 'commander', 'base')
SCOUT, SOLDIER, COMMANDER, BASE = range(4)
SPEED = torch.tensor([3.0, 2.0, 1.5, 0.])    # scouts are quick: fetch hearts, reach the wounded
MAX_HP = torch.tensor([4., 6., 8., 40.])
SIGHT = torch.tensor([1., 1., 1., 5.])    # multiplier on vision and radar range
CARRY = torch.tensor([3., 1., 1., 30.])   # hearts a scout can carry / a base can store
TURN = .4

VISION_SECTORS, VISION_SPAN, VISION_RANGE = 9, .9, 48.
RADAR_SECTORS, RADAR_RANGE = 16, 160.
DIAG = GRID * math.sqrt(2)                # base distances are scaled by this on any board size

K = 5                                     # bullet slots per agent
BULLET_SPEED, BULLET_LIFE, GUN_CD = 3.5, 14, 5
MISSILE_SPEED, MISSILE_LIFE, MISSILE_CD = 3., 18, 15
BASE_MISSILE_SPEED, BASE_MISSILE_LIFE, BASE_GUN_CD = 5., 50, 8     # 5x a tank's reach
HIT_RADIUS, BASE_HIT_RADIUS = .9, 3.
BLOCK = 2                                 # a placed block covers BLOCK x BLOCK tiles
BLOCK_HP, BLOCK_CD, BLOCK_REACH, MAX_BLOCKS = 3., 4, 2.5, 60   # 3 hits; one per 4 turns; 60 standing per team
WALL_RADIUS, WALL_WIDTH, WALL_COST = 14., 5, 1   # a base's wall order: 5 blocks across where its turret points, 1 heart

HEARTS, HEART_RESPAWN, SPAWN_POOL = 400, 250, 4096   # only scouts can pick them up
PICK_RADIUS, HEAL_RADIUS, HEAL_AMOUNT, HEAL_CD = 1., 4., 2., 4
DEPOSIT_RADIUS = BASE_CLEAR + 3
ORDERS = ('none', 'repair', 'build scout', 'build soldier', 'build heavy', 'build wall')
BUILD_COST = torch.tensor([2., 5., 10.])  # scout, soldier, heavy
REPAIR_SELF, REPAIR_NEAR, ORDER_CD = 10., 2., 5

# ---- reward --------------------------------------------------------------------
# individual credit: the agent that lands the kill, wastes the shot or dies feels it
KILL_W, BASE_KILL_W, DAMAGE_W = 1., 3., .2
# a blind shot -- fired with no enemy anywhere within the shooter's radar range -- costs;
# a shot with an enemy about never does beyond the miss, even if it's badly aimed
# (charging every shot, or every shot without an enemy dead ahead, taught fresh
# policies to stop shooting before they could aim)
BLIND_SHOT_W, MISS_W = -.1, -.05
# bases: extra per damage to an enemy base, and more again for every allied tank (beyond
# the first, up to four) also at that base -- mass on one target instead of trickling in
BASE_DAMAGE_W, SIEGE_W = .1, .1
DEATH_W, BASE_DEATH_W = -1., -8.
FRIENDLY_DAMAGE_W, FRIENDLY_KILL_W = -.2, -2.   # friendly fire is on: a hit costs what an enemy hit earns, a kill a lot more
# bases: losing hurts more than winning pays (each living teammate -4 per base lost, +2
# per enemy base taken; the game loss is twice the win), and damage to an enemy within
# DEFEND_RADIUS of one of your bases is worth 3x
BASE_LOST_W, BASE_WON_W, WIN_BONUS, TIMEOUT_BONUS, LOSS_SCALE = -4., 2., 3., 1.5, 2.
DEFEND_RADIUS, DEFEND_W = 60., .4
# team spirit: every reward is blended half-and-half with the team's average, so helping
# the team pays as much as helping yourself -- the glue for board-wide plans
TEAM_SPIRIT = .5
ASSIST_W, ASSIST_WINDOW = 1., 20          # everyone who hit an enemy in its last 20 turns gets as much as the killer
# comrades: a soldier or heavy pays up to AWAY_W a turn for drifting from the group --
# nothing while its second-nearest allied tank is within FORM_MAX, all of it by FORM_MAX +
# AWAY_SCALE; a comrade dying close by stings (a heavy more); only actual overlap is penalised.
# Keep all of these mild: early in training, when nothing can aim yet, every one of them
# lands on *doing something*, and at twice these values fresh policies learned to freeze
FORM_MAX, AWAY_SCALE, AWAY_W, STACK_W = 12., 40., -.01, -.1
COMRADE_NEAR, COMRADE_DEATH_W, COMMANDER_DEATH_W = 25., -.05, -.15
# ...and one that hasn't hit an enemy for IDLE_TURNS pays as much again, wherever it is
# (as strong as AWAY_W, or groups just huddle; small next to death, so dying never pays).
# A scout is idle when it hasn't delivered or healed for that long (picking up doesn't
# count: a scout that carried its hearts around forever wasn't working)
IDLE_TURNS, IDLE_W = 60, -.01
# economy: a scout is paid for fetching hearts and for bringing them home, with a pull
# toward the nearest heart while it has room and toward the nearest base while carrying
# (potential-based: each nets to zero over the trip, so it guides without changing what's
# worth doing -- without it scouts picked up one heart a game and never found the chain
# from heart to base), and for healing a teammate within reach -- more for one nearly dead. A base is paid for the hp a repair
# restores, per heart spent on a tank, and per wall block raised. Kept short of the
# combat rewards: shared through team spirit, big economy rewards once taught whole
# armies to farm safely at home instead of fighting
PICKUP_W, DEPOSIT_W = .4, 1.
HEART_PULL, HEART_REACH, CARRY_PULL, HOME_REACH = .3, 50., .5, 100.
HEAL_W, RESCUE_W, RESCUE_FRAC = .5, .5, .35
REPAIR_W, BUILD_W, WALL_W = .05, .1, .1
# blocks: a block dropped with no enemy in radar range and no base of yours nearby is
# pointless and costs; anywhere else placing is free, one that extends a wall of your
# team's blocks pays, and a block pays its placer for every enemy bullet it stops and
# every enemy tank it stops at the placer's bases -- barricades go up where they matter
BLOCK_PLACE_W, BLOCK_WALL_W, BLOCK_SAVE_W, BLOCK_STOP_W = -.15, .15, .5, .2

OBS = 4 + 8 + 2 + 2 + 4 * VISION_SECTORS + 10 * RADAR_SECTORS
POS = slice(12, 14)                       # where an agent's position (as a fraction of the board) sits in its observation
RADAR = 16 + 4 * VISION_SECTORS           # where the radar channels start: wall, block, friend, enemy, need, own base, foe base, threat, foe damage, heart
FAR = 1e5
_maps = {}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _map(k, grid=GRID, bases=BASES):
    """Map k (deterministic) on a grid x grid board, in one of four styles: rubble
    (lots of small rocks), boulders (fewer, bigger), corridors (long thin walls) or
    open (sparse rocks). 2 x bases are spread apart (BASE_GAP on the full board,
    scaled down with it) with open ground round each, and every pocket not connected
    to the big open area is walled off. Returns (walls, base positions: first half
    team 0, second half team 1)."""
    style = STYLES[k % len(STYLES)]
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
        count, lo, hi = {'rubble': (N_RECTS, 3, 10), 'boulders': (N_RECTS // 6, 8, 24),
                         'corridors': (N_RECTS // 12, 2, 4), 'open': (N_RECTS // 4, 3, 10)}[style]
        wh = rng.uniform(lo, hi, (int(count * f * f), 2))
        if style == 'corridors':                                   # long one way, thin the other
            wh[np.arange(len(wh)), rng.integers(2, size=len(wh))] = rng.uniform(30, 90, len(wh)) * f
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
        self.N = self.slots + bases                                # agents per team: tank slots, then bases
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
        self.base_slots = self.is_base.nonzero().squeeze(1)       # team 0's, then team 1's
        self.vision_centers = (torch.arange(VISION_SECTORS, device=d) + .5) / VISION_SECTORS * 2 * VISION_SPAN - VISION_SPAN
        self.radar_centers = (torch.arange(RADAR_SECTORS, device=d) + .5) / RADAR_SECTORS * 2 * math.pi - math.pi
        self.vision_steps = torch.arange(1., VISION_RANGE + 1e-3, 1., device=d)
        self.radar_steps = torch.arange(2.5, RADAR_RANGE + 1e-3, 2.5, device=d)
        self.tables = {k: v.to(d) for k, v in dict(speed=SPEED, max_hp=MAX_HP, sight=SIGHT, carry=CARRY, cost=BUILD_COST).items()}

        B = batch
        z = lambda *s, dt=torch.float32: torch.zeros(B, *s, dtype=dt, device=d)
        self.pos, self.heading, self.hp, self.supply = z(self.A, 2), z(self.A), z(self.A), z(self.A)
        self.gun_cd, self.special_cd, self.block_cd = z(self.A), z(self.A), z(self.A)
        self.born = z(self.A)                                      # turn each agent (re)appeared, for idleness
        self.last_useful = z(self.A)                               # turn a scout last delivered or healed
        self.phi = z(self.A)                                       # carry-home shaping potential
        self.last_hit = z(self.A, self.A)                          # [target, shooter] turn of the last enemy hit
        self.role = torch.zeros(B, self.A, dtype=torch.long, device=d)
        self.b_pos, self.b_vel = z(self.A, K, 2), z(self.A, K, 2)
        self.b_life, self.b_dmg = z(self.A, K), z(self.A, K)
        self.b_pierce, self.b_alive = z(self.A, K, dt=torch.bool), z(self.A, K, dt=torch.bool)
        self.heart_pos, self.heart_timer, self.heart_alive = z(self.H, 2), z(self.H), z(self.H, dt=torch.bool)
        self.spawn_pool = z(SPAWN_POOL, 2)
        self.BG = -(-grid // BLOCK)                                # placed-block grid, one cell per block
        # hp of the placed block in each cell (a block stands while hp > 0) and the agent who
        # placed it (only meaningful while it stands). One extra cell at the end soaks up the
        # writes of agents that aren't placing, so every update is in place, never a full copy
        self.blocks = z(self.BG * self.BG + 1)
        self.block_owner = torch.full((B, self.BG * self.BG + 1), -1, dtype=torch.int16, device=d)
        self.t = torch.zeros(B, dtype=torch.long, device=d)
        self.winner = torch.full((B,), -1, dtype=torch.long, device=d)
        self.stats = {k: torch.zeros((), device=d) for k in (
            'shots', 'blind_shots', 'hits', 'friendly_hits', 'misses', 'kills', 'deaths', 'assists', 'base_kills', 'base_damage',
            'bases_lost', 'defend_hits', 'heals', 'rescues', 'pickups', 'deposits', 'builds', 'repairs', 'base_walls',
            'blocks_placed', 'wall_blocks', 'block_saves', 'block_stops', 'grouped', 'idle', 'stacked',
            'games', 'decisive', 'turns')}
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
        start = torch.tensor([SCOUT] * self.start[0] + [SOLDIER] * self.start[1] + [COMMANDER] * self.start[2], device=self.device)
        local = torch.arange(self.A, device=self.device) % self.N
        role = torch.where(local < len(start), start[local.clamp(max=len(start) - 1)], SOLDIER)
        role = torch.where(self.is_base, BASE, role)
        alive = (local < len(start)) | self.is_base
        self.role[rows] = role
        # every tank starts on a ring round one of its team's bases (dealt out in turn), evenly
        # spaced ~11 tiles apart: packed in a disc, any move meant an overlap penalty and any
        # shot hit a friend, and a fresh policy learned to freeze
        rank, per_base = local // self.nb, -(-self.slots // self.nb)
        ang = 2 * math.pi * rank / per_base + self._rand(len(rows), 1) * 2 * math.pi
        ring = torch.stack((ang.cos(), ang.sin()), -1) * (BASE_CLEAR - 4)
        pos = self.base_pos[rows][:, self.team * self.nb + local % self.nb] + ring
        pos[:, self.base_slots] = self.base_pos[rows]
        self.pos[rows] = pos
        self.heading[rows] = (self._rand(len(rows), self.A) * 2 - 1) * math.pi
        self.hp[rows] = self.tables['max_hp'][role] * alive
        for x in (self.supply, self.gun_cd, self.special_cd, self.block_cd, self.born, self.last_useful, self.phi,
                  self.heart_timer, self.blocks):
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

    def _cell(self, xy):
        """Index of the placed-block cell under each point."""
        c = (xy / BLOCK).floor().long().clamp(0, self.BG - 1)
        return c[..., 0] * self.BG + c[..., 1]

    def _blocks_at(self, xy):
        flat = self._cell(xy)
        return self.blocks.gather(1, flat.reshape(self.B, -1)).view(flat.shape) > 0

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
    def _sector(idx, value, valid, sectors, reduce='amin'):
        """Per sector, the min (distances) or max (levels) of value over the valid ones."""
        empty = FAR if reduce == 'amin' else 0.
        v = torch.where(valid, value, torch.full_like(value, empty))
        out = torch.full((*value.shape[:-1], sectors), empty, device=value.device)
        return out.scatter_reduce(-1, idx.clamp(0, sectors - 1), v, reduce)

    # ---- observation --------------------------------------------------------
    def _observe(self):
        B = self.B
        alive = self.hp > 0
        sight = self.sight
        radar_r, vision_r = RADAR_RANGE * sight, VISION_RANGE * sight
        delta = self.pos[:, None] - self.pos[:, :, None]           # (B,A,A,2): j seen from i
        dist = delta.norm(dim=-1)
        rel = wrap(torch.atan2(delta[..., 1], delta[..., 0]) - self.heading[..., None])
        seen = alive[:, None] & ~self.eye
        friend, enemy = seen & self.same, seen & ~self.same
        tank_j = ~self.is_base.view(1, 1, self.A)
        frac = self.hp / self.max_hp

        # radar: long range, through walls
        r_idx = ((rel + math.pi) / (2 * math.pi) * RADAR_SECTORS).long()
        near = dist < radar_r[..., None]
        R = lambda value, valid, reduce='amin': self._sector(r_idx, value, valid, RADAR_SECTORS, reduce)
        r_wall, r_block = self._rays(self.heading[..., None] + self.radar_centers, self.radar_steps, sight)
        r_friend, r_enemy = R(dist, friend & tank_j & near), R(dist, enemy & tank_j & near)
        r_need = R((1 - frac)[:, None].expand_as(dist), friend & tank_j & near, 'amax')
        r_own, r_foe = R(dist, friend & ~tank_j), R(dist, enemy & ~tank_j)
        # base strategy: enemy tanks closing on each base, and how damaged each base is
        threat = ((dist < DEFEND_RADIUS) & ~self.same & (alive & ~self.is_base)[:, :, None]).sum(1).float() / 5
        r_threat = R(threat.clamp(max=1)[:, None].expand_as(dist), friend & ~tank_j, 'amax')
        r_foe_dmg = R((1 - frac)[:, None].expand_as(dist), enemy & ~tank_j, 'amax')

        # vision: forward cone, stops at the first wall or placed block
        v_idx = ((rel + VISION_SPAN) / (2 * VISION_SPAN) * VISION_SECTORS).floor().long()
        v_wall, v_block = self._rays(self.heading[..., None] + self.vision_centers, self.vision_steps, sight)
        v_block = torch.where(v_block < v_wall, v_block, torch.full_like(v_block, FAR))
        v_stop = v_wall.minimum(v_block)
        cone = (rel.abs() < VISION_SPAN) & (dist < vision_r[..., None])
        v_enemy = self._sector(v_idx, dist, enemy & cone, VISION_SECTORS)
        h_delta = self.heart_pos[:, None] - self.pos[:, :, None]      # (B,A,H,2)
        h_dist = h_delta.norm(dim=-1)
        h_rel = wrap(torch.atan2(h_delta[..., 1], h_delta[..., 0]) - self.heading[..., None])
        h_ok = self.heart_alive[:, None] & (h_rel.abs() < VISION_SPAN) & (h_dist < vision_r[..., None])
        h_idx = ((h_rel + VISION_SPAN) / (2 * VISION_SPAN) * VISION_SECTORS).floor().long()
        v_heart = self._sector(h_idx, h_dist, h_ok, VISION_SECTORS)
        hr_idx = ((h_rel + math.pi) / (2 * math.pi) * RADAR_SECTORS).long()
        r_heart = self._sector(hr_idx, h_dist, self.heart_alive[:, None] & (h_dist < radar_r[..., None]), RADAR_SECTORS)
        v_enemy = torch.where(v_enemy < v_stop, v_enemy, torch.full_like(v_enemy, FAR))
        v_heart = torch.where(v_heart < v_stop, v_heart, torch.full_like(v_heart, FAR))

        by_team = alive.view(B, 2, self.N)
        tanks_alive = by_team[..., :self.slots].float().sum(-1) / sum(self.start)
        bases_alive = by_team[..., self.slots:].float().mean(-1)
        is_b = self.role == BASE
        clip = lambda x, r: (x / r.unsqueeze(-1)).clamp(max=1)
        me = torch.stack((frac, self.supply / self.carry,
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
            clip(r_own, diag), clip(r_foe, diag), r_threat, r_foe_dmg, clip(r_heart, radar_r),
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
        throttle, steer, fire, special, order, place = action.unbind(-1)
        role = self.role
        is_b = role == BASE
        dd0 = torch.cdist(self.pos, self.pos)
        intruding = torch.where(~self.same & (is_b & alive0)[:, None], dd0, FAR).amin(-1) < DEFEND_RADIUS
        attackers = ((dd0 < DEFEND_RADIUS) & ~self.same & (alive0 & ~is_b)[:, :, None]).sum(1).float()   # enemy tanks at each base

        # move, sliding along whatever is in the way
        self.heading = torch.where(alive0, wrap(self.heading + steer.clamp(-1, 1) * TURN), self.heading)
        u = torch.stack((self.heading.cos(), self.heading.sin()), -1)
        prop = (self.pos + u * (throttle.clamp(-1, 1) * self.speed).unsqueeze(-1)).clamp(.5, self.grid - .5)
        blocked = self._solid_at(prop)
        # an intruder whose move is stopped by an enemy's placed block earns the block's placer
        owner = self.block_owner.gather(1, self._cell(prop)).long()
        stop = (blocked & intruding & alive0 & ~is_b & (throttle != 0) & self._blocks_at(prop) & ~self._walls_at(prop)
                & (owner >= 0) & (owner // self.N != self.team))
        block_stops = torch.zeros(B, self.A, device=d).scatter_add(1, owner.clamp(min=0), stop.float())
        for axis in range(2):
            alt = self.pos.clone()
            alt[..., axis] = prop[..., axis]
            ok = blocked & ~self._solid_at(alt)
            prop = torch.where(ok.unsqueeze(-1), alt, prop)
            blocked = blocked & ~ok
        self.pos = torch.where((alive0 & ~blocked).unsqueeze(-1), prop, self.pos)
        for x in ('gun_cd', 'special_cd', 'block_cd', 'heart_timer'):
            setattr(self, x, (getattr(self, x) - 1).clamp(min=0))

        # place blocks, shoot, act
        counts = self._block_counts()
        placed, walled, counts = self._put_blocks(alive0 & ~is_b & (place > .5) & (self.block_cd == 0), self.pos + u * BLOCK_REACH, counts)
        self.block_cd = torch.where(placed > 0, float(BLOCK_CD), self.block_cd)
        dd = torch.cdist(self.pos, self.pos)
        blind = ~(alive0[:, None] & ~self.same & (dd < (RADAR_RANGE * self.sight)[..., None])).any(-1)   # nobody to shoot at
        can_fire = alive0 & (fire > .5) & (self.gun_cd == 0)
        shoot = self._spawn(can_fire & ((role == SOLDIER) | (role == COMMANDER)), u, BULLET_SPEED, BULLET_LIFE, 1., False)
        big = self._spawn(can_fire & is_b, u, BASE_MISSILE_SPEED, BASE_MISSILE_LIFE, 2., True)
        self.gun_cd = torch.where(shoot, float(GUN_CD), torch.where(big, float(BASE_GUN_CD), self.gun_cd))
        launch = self._spawn(alive0 & (role == COMMANDER) & (special > .5) & (self.special_cd == 0), u, MISSILE_SPEED, MISSILE_LIFE, 2., True)
        self.special_cd = torch.where(launch, float(MISSILE_CD), self.special_cd)
        picked, d_heart = self._pickup(alive0)
        deposited, healed, rescued = self._scout_special(alive0, special)
        built, spent, repaired, base_walls = self._base_orders(alive0, order, u, counts)
        dealt, base_dmg, friendly_dmg, kills, base_kills, friendly_kills, misses, saves, defend, siege = self._bullets(alive0, intruding, attackers)
        died = alive0 & (self.hp <= 0)
        blind_shots = ((shoot | launch | big) & blind).float()
        respawn = ~self.heart_alive & (self.heart_timer == 0)
        fresh = self.spawn_pool.gather(1, (self._rand(B, self.H) * SPAWN_POOL).long().unsqueeze(-1).expand(-1, -1, 2))
        self.heart_pos = torch.where(respawn.unsqueeze(-1), fresh, self.heart_pos)
        self.heart_alive |= respawn

        # teamwork terms
        assists = (died.unsqueeze(-1) & (self.t.view(B, 1, 1) - self.last_hit <= ASSIST_WINDOW)).sum(1).float()
        bases_lost = (died & is_b).view(B, 2, self.N).sum(-1).float()
        tank = (self.hp > 0) & ~is_b
        fighter = tank & (role != SCOUT)
        gap = dd.masked_fill(~(tank[:, None] & tank[:, :, None]) | self.eye, FAR)     # between living tanks
        overlap = (1 - gap.amin(-1)).clamp(min=0) * tank
        second = gap.masked_fill(~self.same, FAR).topk(2, -1, largest=False).values[..., 1]
        away = ((second - FORM_MAX) / AWAY_SCALE).clamp(0, 1) * fighter
        fallen = torch.where(role == COMMANDER, COMMANDER_DEATH_W, COMRADE_DEATH_W) * (died & ~is_b)
        comrade_loss = torch.bmm(((dd < COMRADE_NEAR) & self.same).float(), fallen.unsqueeze(-1)).squeeze(-1) * tank
        useful = torch.where(role == SCOUT, self.last_useful, self.last_hit.amax(1))
        idle = tank & (self.t.view(B, 1) - torch.maximum(useful, self.born) > IDLE_TURNS)
        home = torch.where(self.same & (is_b & (self.hp > 0))[:, None], dd, torch.full_like(dd, FAR)).amin(-1)
        enemy_near = ((dd < RADAR_RANGE) & ~self.same & tank[:, None]).any(-1)
        pointless = ~enemy_near & (home > DEFEND_RADIUS)
        scout = tank & (role == SCOUT)
        phi = (HEART_PULL * torch.exp(-d_heart / HEART_REACH) * (scout & (self.supply < self.carry))
               + CARRY_PULL * self.supply / self.carry * torch.exp(-home / HOME_REACH) * scout)
        pull, self.phi = phi - self.phi, phi

        reward = (KILL_W * kills + BASE_KILL_W * base_kills + DAMAGE_W * dealt + BASE_DAMAGE_W * base_dmg + SIEGE_W * siege
                  + BLIND_SHOT_W * blind_shots + MISS_W * misses
                  + FRIENDLY_DAMAGE_W * friendly_dmg + FRIENDLY_KILL_W * friendly_kills
                  + torch.where(is_b, BASE_DEATH_W, DEATH_W) * died
                  + BASE_LOST_W * bases_lost[:, self.team] + BASE_WON_W * bases_lost[:, 1 - self.team] + DEFEND_W * defend
                  + ASSIST_W * assists + AWAY_W * away + STACK_W * overlap + comrade_loss + IDLE_W * idle
                  + PICKUP_W * picked + DEPOSIT_W * deposited + pull + HEAL_W * healed + RESCUE_W * rescued
                  + REPAIR_W * repaired + BUILD_W * spent + WALL_W * base_walls
                  + BLOCK_PLACE_W * placed * pointless + BLOCK_WALL_W * walled + BLOCK_SAVE_W * saves + BLOCK_STOP_W * block_stops)
        live = alive0.view(B, 2, self.N)
        team_mean = (reward.view(B, 2, self.N) * live).sum(-1) / live.sum(-1).clamp(min=1)
        reward = (1 - TEAM_SPIRIT) * reward + TEAM_SPIRIT * team_mean[:, self.team]

        # outcome: a team with no bases left loses; at the limit, more bases (then tanks) wins
        self.t += 1
        by_team = (self.hp > 0).view(B, 2, self.N)
        wiped = ~by_team[..., self.slots:].any(-1)
        timeout = self.t >= self.limit
        done = wiped.any(-1) | timeout
        by_count = timeout & ~wiped.any(-1)
        score = by_team[..., self.slots:].sum(-1) * 100 + by_team[..., :self.slots].sum(-1)
        winner = torch.full_like(self.t, -1)
        winner = torch.where(wiped[:, 1] & ~wiped[:, 0], 0, winner)
        winner = torch.where(wiped[:, 0] & ~wiped[:, 1], 1, winner)
        winner = torch.where(by_count & (score[:, 0] > score[:, 1]), 0, winner)
        winner = torch.where(by_count & (score[:, 1] > score[:, 0]), 1, winner)
        size = torch.where(by_count, TIMEOUT_BONUS, WIN_BONUS).unsqueeze(1)
        outcome = torch.where(winner.unsqueeze(1) == self.team, 1., -LOSS_SCALE) * size * (winner >= 0).unsqueeze(1)
        reward = (reward + outcome * done.unsqueeze(1)) * alive0
        terminal = alive0 & (died | done.unsqueeze(1))
        self.winner = winner

        for k, v in dict(shots=shoot.sum() + big.sum() + launch.sum(), blind_shots=blind_shots, misses=misses, kills=kills, deaths=died,
                         assists=assists, base_kills=base_kills, base_damage=base_dmg, bases_lost=bases_lost,
                         defend_hits=defend > 0, heals=healed > 0, rescues=rescued, pickups=picked, deposits=deposited,
                         builds=built, repairs=repaired > 0, base_walls=base_walls, blocks_placed=placed, wall_blocks=walled,
                         block_saves=saves, block_stops=block_stops, grouped=fighter & (second <= FORM_MAX), idle=idle,
                         stacked=overlap > 0, games=done, decisive=done & ~by_count & (winner >= 0),
                         turns=self.t * done).items():
            self.stats[k] += v.sum()
        return reward, terminal, done, winner

    def _spawn(self, want, u, speed, life, dmg, pierce):
        free = ~self.b_alive
        want = want & free.any(-1)
        put = want.unsqueeze(-1) & (torch.arange(K, device=self.device) == free.float().argmax(-1, keepdim=True))
        self.b_pos = torch.where(put.unsqueeze(-1), self.pos.unsqueeze(2), self.b_pos)
        self.b_vel = torch.where(put.unsqueeze(-1), (u * speed).unsqueeze(2), self.b_vel)
        self.b_life = torch.where(put, float(life), self.b_life)
        self.b_dmg = torch.where(put, float(dmg), self.b_dmg)
        self.b_pierce = torch.where(put, torch.full_like(self.b_pierce, pierce), self.b_pierce)
        self.b_alive |= put
        return want

    def _block_counts(self):
        """Standing blocks per team (B, 2): one pass over the block grid per turn."""
        standing, t1 = self.blocks[:, :-1] > 0, self.block_owner[:, :-1] >= self.N
        return torch.stack(((standing & ~t1).sum(-1), (standing & t1).sum(-1)), -1)

    def _put_blocks(self, want, xy, counts):
        """Each agent that wants to drops a block into the cell at xy -- if that cell is
        empty ground, holds no block, nobody is standing in it, and its team has fewer
        than MAX_BLOCKS standing. Returns (placed, placed touching one of the team's own
        blocks, counts)."""
        cell = (xy / BLOCK).floor()
        inside = ((cell >= 0) & (cell < self.BG)).all(-1)
        cell = cell.clamp(0, self.BG - 1)
        corners = torch.tensor([[.5, .5], [1.5, .5], [.5, 1.5], [1.5, 1.5]], device=self.device)
        wall = self._walls_at(cell.unsqueeze(-2) * BLOCK + corners).any(-1)
        flat = (cell[..., 0] * self.BG + cell[..., 1]).long()
        taken = self.blocks.gather(1, flat) > 0
        crowded = (torch.cdist((cell + .5) * BLOCK, self.pos) < 1.6).logical_and((self.hp > 0)[:, None]).any(-1)
        room = counts.gather(1, self.team.view(1, -1).expand(self.B, -1)) < MAX_BLOCKS
        ok = want & inside & ~wall & ~taken & ~crowded & room
        adjacent = torch.zeros_like(ok)                       # does it extend one of our walls (4-neighbours)?
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = cell + torch.tensor([dx, dy], device=self.device)
            nf = self._cell(nb * BLOCK + .5)
            adjacent |= ((nb >= 0) & (nb < self.BG)).all(-1) & (self.blocks.gather(1, nf) > 0) & (self.block_owner.gather(1, nf) // self.N == self.team)
        put = torch.where(ok, flat, self.BG * self.BG)            # everyone else writes to the spare cell
        self.blocks.scatter_(1, put, BLOCK_HP)
        self.block_owner.scatter_(1, put, torch.arange(self.A, device=self.device, dtype=torch.int16).expand(self.B, -1))
        return ok.float(), (ok & adjacent).float(), counts + ok.view(self.B, 2, self.N).sum(-1)

    def _bullets(self, alive0, intruding, attackers):
        """Advance every bullet one step, test the whole swept segment against every
        agent but the shooter (friendly fire is on), and credit damage, kills and misses
        -- split into enemy and friendly -- to the agent that fired it."""
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
        gap = torch.where(alive0[:, None, None] & ~self.eye[None, :, None], gap - radius, torch.full_like(gap, FAR))
        mind, tgt = gap.min(-1)
        hit = self.b_alive & (mind < 0)
        dmg = hit * self.b_dmg
        per_target = lambda x: x.gather(1, tgt.view(B, -1)).view_as(hit)
        self.hp = (self.hp - torch.zeros(B, self.A, device=self.device).scatter_add(1, tgt.view(B, -1), dmg.view(B, -1))).clamp(min=0)
        killed = hit & per_target(alive0 & (self.hp <= 0))
        base_hit = per_target(self.role == BASE)
        friendly = self.same[torch.arange(self.A, device=self.device).view(1, self.A, 1), tgt]
        foe = hit & ~friendly
        pair = torch.nn.functional.one_hot(tgt, self.A).bool() & foe.unsqueeze(-1)   # (B,shooter,K,target): for assists
        self.last_hit = torch.where(pair.any(2).transpose(1, 2), self.t.view(B, 1, 1).float(), self.last_hit)
        # bullets that reach a placed block first chip it; enemy fire it stops pays its placer
        chip = self.b_alive & ~hit & block
        at = self._cell(torch.where(bm.unsqueeze(-1), mid, new)).view(B, -1)
        owner = self.block_owner.gather(1, at).long()
        mine = (owner >= 0) & (owner // self.N == self.team.view(1, self.A, 1).expand(B, -1, K).reshape(B, -1))
        stopped = chip.view(B, -1) & (owner >= 0) & ~mine
        saves = torch.zeros(B, self.A, device=self.device).scatter_add(1, owner.clamp(min=0), stopped.float())
        self.blocks.scatter_add_(1, at, -(chip * self.b_dmg).view(B, -1))
        own_block = chip & mine.view_as(chip)                                     # shooting your own wall is a wasted shot
        gone = self.b_alive & ~hit & (~chip | own_block) & (wall | out | (self.b_life <= 1) | own_block)
        self.stats['hits'] += foe.sum()
        self.stats['friendly_hits'] += (hit & friendly).sum()
        self.b_pos, self.b_life = new, self.b_life - 1
        self.b_alive &= ~hit & ~chip & ~wall & ~out & (self.b_life > 0)
        s = lambda x: x.sum(-1).float()
        on_base = dmg * (foe & base_hit)
        return (s(dmg * ~friendly), s(on_base), s(dmg * friendly), s(killed & foe & ~base_hit),
                s(killed & foe & base_hit), s(killed & friendly), s(gone), saves, s(dmg * (foe & per_target(intruding))),
                s(on_base * (per_target(attackers) - 1).clamp(0, 4)))

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
        nearest = torch.where(self.heart_alive[:, None] & ~taken[:, None], torch.cdist(self.pos, self.heart_pos), torch.full_like(dh, FAR)).amin(-1)
        self.heart_alive &= ~taken
        self.heart_timer = torch.where(taken, float(HEART_RESPAWN), self.heart_timer)
        return won.float(), nearest

    def _scout_special(self, alive0, special):
        """A scout's special: at a friendly base, unload every carried heart into it;
        anywhere else, spend one healing the nearest hurt teammate within reach."""
        act = alive0 & (self.role == SCOUT) & (special > .5) & (self.supply > 0) & (self.special_cd == 0)
        dd = torch.cdist(self.pos, self.pos)
        base_ok = self.same & (self.role == BASE)[:, None] & (self.hp > 0)[:, None]
        bd, which = torch.where(base_ok, dd, torch.full_like(dd, FAR)).min(-1)
        unload = act & (bd < DEPOSIT_RADIUS)
        amount = torch.where(unload, self.supply.minimum((self.carry - self.supply).gather(1, which)), torch.zeros_like(self.supply))
        self.supply = (self.supply + torch.zeros_like(self.supply).scatter_add(1, which, amount) - amount).minimum(self.carry)
        hurt = (self.hp > 0) & (self.hp < self.max_hp) & (self.role != BASE)
        md, who = torch.where(self.same & ~self.eye & hurt[:, None], dd, torch.full_like(dd, FAR)).min(-1)
        heal = act & ~unload & (md < HEAL_RADIUS)
        rescued = (heal & ((self.hp / self.max_hp).gather(1, who) < RESCUE_FRAC)).float()
        given = torch.where(heal, (self.max_hp - self.hp).gather(1, who).clamp(max=HEAL_AMOUNT), torch.zeros_like(self.hp))
        self.hp = (self.hp + torch.zeros_like(self.hp).scatter_add(1, who, given)).minimum(self.max_hp)
        self.supply -= heal.float()
        self.last_useful = torch.where(heal | unload, self.t.view(-1, 1).float(), self.last_useful)
        self.special_cd = torch.where(heal | unload, float(HEAL_CD), self.special_cd)
        return amount, given, rescued

    def _base_orders(self, alive0, order, u, counts):
        """Bases spend stored hearts: repair (themselves and every friendly tank close
        by), build a new tank into one of the team's empty slots next to the base (at
        most one build per team per turn), or raise a wall: WALL_WIDTH blocks side by
        side, WALL_RADIUS out where the base's turret points."""
        B, d = self.B, self.device
        ready = alive0 & (self.role == BASE) & (self.special_cd == 0)
        order = order.long()
        repair = ready & (order == 1) & (self.supply >= 1)
        near = torch.cdist(self.pos, self.pos) < DEPOSIT_RADIUS
        boost = (repair[:, :, None] & near & self.same).any(1) & (self.hp > 0) & (self.role != BASE)
        hp0 = self.hp
        self.hp = torch.where(repair, self.hp + REPAIR_SELF, torch.where(boost, self.hp + REPAIR_NEAR, self.hp)).minimum(self.max_hp)
        restored = (self.hp - hp0) * repair                      # what the repair actually put back on the base
        self.supply -= repair.float()

        kind = (order - 2).clamp(0, 2)                          # 0 scout, 1 soldier, 2 heavy
        cost = self.tables['cost'][kind]
        want = ready & (order >= 2) & (order <= 4) & (self.supply >= cost)
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
            built, spent = built + at_base.float(), spent + at_base * cost

        wall = ready & (order == 5) & (self.supply >= WALL_COST)
        perp = torch.stack((-u[..., 1], u[..., 0]), -1)
        raised = torch.zeros(B, self.A, device=d)
        for k in range(WALL_WIDTH):
            ok, _, counts = self._put_blocks(wall, self.pos + u * WALL_RADIUS + perp * (k - (WALL_WIDTH - 1) / 2) * BLOCK, counts)
            raised = raised + ok
        wall = wall & (raised > 0)                               # nothing placed, nothing paid
        self.supply = self.supply - wall * WALL_COST
        self.special_cd = torch.where(repair | (built > 0) | wall, float(ORDER_CD), self.special_cd)
        return built, spent, restored, raised

    # ---- scripted benchmark ---------------------------------------------------
    def raider(self):
        """Fixed opponent that sees everything. Soldiers and commanders drive at the
        nearest enemy and fire when lined up; scouts fetch hearts and take them to the
        nearest friendly base; bases shoot what's in range and build soldiers."""
        alive = self.hp > 0
        dd = torch.cdist(self.pos, self.pos)
        foe = torch.where(alive[:, None] & ~self.same, dd, torch.full_like(dd, FAR)).argmin(-1)
        home = torch.where(alive[:, None] & self.same & (self.role == BASE)[:, None], dd, torch.full_like(dd, FAR)).argmin(-1)
        heart = torch.where(self.heart_alive[:, None], torch.cdist(self.pos, self.heart_pos), FAR).argmin(-1)
        scout = self.role == SCOUT
        at = lambda src, i: src.gather(1, i[..., None].expand(-1, -1, 2))
        target = torch.where((scout & (self.supply < 1)).unsqueeze(-1), at(self.heart_pos, heart), at(self.pos, foe))
        target = torch.where((scout & (self.supply >= 1)).unsqueeze(-1), at(self.pos, home), target)
        delta = target - self.pos
        dist = delta.norm(dim=-1)
        bearing = wrap(torch.atan2(delta[..., 1], delta[..., 0]) - self.heading)
        base = self.role == BASE
        reach = torch.where(base, BASE_MISSILE_SPEED * BASE_MISSILE_LIFE, VISION_RANGE)
        fire = ((bearing.abs() < .12) & (dist < reach)).float()
        order = torch.where(base & (self.supply >= BUILD_COST[SOLDIER]), 3., 0.)
        return torch.stack((((dist > 4) | scout).float(), (bearing * 2).clamp(-1, 1), fire, torch.ones_like(fire),
                            order, torch.zeros_like(fire)), -1)
