"""Self-play PPO with a recurrent policy: the network drives, shoots, builds, and keeps
a shared map of the war.

Policy: observation -> 2-layer encoder -> + what it reads off the team's map -> GRU
memory -> actions. The GRU (512 units) lets an agent remember what it saw a few turns
ago; it is reset when an agent dies or its game ends.

Team map ("the radio"): each team keeps a 32 x 32 grid of 24-number vectors laid over
the board -- one vector per sector, whatever the board's size. Every turn each living
agent *writes* to the vector of the sector it stands in and *reads* the 5 x 5 sectors
round it plus the whole map pooled down to 8 x 8 -- its picture of the entire board --
digested by a small layer into 256 numbers before it meets the GRU (fed raw, the
2136-number read outweighed the agent's own senses four to one, and a team did better
with its map switched off). 8 of the 24 numbers are a *sighting report* taken
straight from the writer's sensors -- enemies on its radar, how close the nearest
enemy tank and base are, its health, threats to friendly bases -- so the map means
something from the first turn; the other 16 are a gated, learned write (the writer
decides how much to overwrite and with what), trained by the readers' policy
gradient. Old reports fade a little every turn. What gets written is learned: it is
all one differentiable pass, so a reader's policy gradient trains the writer. The
map is wiped when its game ends, and fades slowly so stale reports don't linger.

Thousands of games run side by side, on three board sizes at once, and reset
independently; training is a stream of 32-turn rollouts, and PPO replays each game's
rollout in order through the GRU and the map (backprop through time). A quarter of
the games put one side under someone else: half under the scripted raider, half under
a frozen past snapshot of the policy, so it keeps beating older styles instead of
chasing its own tail. The latest weights go to tank_policy.pt after every iteration
so viewer.py can watch while this runs.
"""
import argparse
import copy
import os
import random
import time
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import arena
from arena import Arena, OBS, POS, RADAR, RADAR_SECTORS as RS

FACTS = 8                                 # channels of each sector's vector that are a sensor report, not learned
HEADS = (9, 2, 2, len(arena.ORDERS), 2)   # move (3 throttle x 3 steer), fire, special, base order, place block
THROTTLE = torch.tensor([-.5, 0., 1.])
STEER = torch.tensor([-1., 0., 1.])
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# entropy bonus per head (move, fire, special, base order, place block). The block head
# once got extra (.05) to stop it collapsing to "never" under a placement cost; with
# placing free near fights that extra just held it at random, so it is back to normal
ENTROPY = torch.tensor([.01, .01, .01, .01, .01])
# the arenas trained on, round-robin: board side, tanks and bases per side, parallel games
ARENAS = {'small': dict(grid=400, tanks=14, bases=3, games=1536),
          'medium': dict(grid=560, tanks=28, bases=5, games=1024),
          'large': dict(grid=800, tanks=40, bases=7, games=512)}


class Policy(nn.Module):
    def __init__(self, hidden=512, cells=32, chan=24, window=5, coarse=8, digest=256, decay=.99):
        super().__init__()
        self.hidden, self.cells, self.chan, self.window, self.coarse, self.decay = hidden, cells, chan, window, coarse, decay
        self.enc = nn.Sequential(nn.Linear(OBS, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.digest = nn.Sequential(nn.Linear((window * window + coarse * coarse) * chan, digest), nn.ReLU())
        self.rnn = nn.GRUCell(hidden + digest, hidden)
        self.write = nn.Linear(hidden, 2 * (chan - FACTS))      # gate and value of the learned part of a write
        self.pi = nn.Linear(hidden, sum(HEADS))
        self.v = nn.Linear(hidden, 1)
        r = torch.arange(window) - window // 2
        self.register_buffer('offsets', torch.stack(torch.meshgrid(r, r, indexing='ij'), -1).view(-1, 2))
        with torch.no_grad():            # place-block head starts near "never", or a fresh policy carpets the board
            self.pi.bias[-2:] = torch.tensor([3., -3.])

    def memory(self, obs):
        """A blank memory for these games: GRU state per agent, and one map per team."""
        B, A = obs.shape[:2]
        return obs.new_zeros(B, A, self.hidden), obs.new_zeros(B, 2, self.cells, self.cells, self.chan)

    def forward(self, obs, alive, state, mute=None):
        """obs (B, A, OBS), alive (B, A), state (h, map) -> logits, value, new state.
        Dead agents' GRU state is wiped. mute cuts those agents off from the map (they
        neither read nor write) -- used to test whether the map matters."""
        h, M = state
        B, A = obs.shape[:2]
        S, C = self.cells, self.chan
        using = alive if mute is None else alive & ~mute
        team = torch.arange(A, device=obs.device) // (A // 2)                    # agents per team: team 0 first
        Mf = M.view(B, 2 * S * S, C)
        cell = (obs[..., POS] * S).long().clamp(0, S - 1)                        # (B, A, 2): the sector I stand in
        with torch.autocast('cuda', torch.bfloat16, enabled=obs.is_cuda):        # ~2x faster on the GPU
            x = self.enc(obs)
            read = self.digest(self.read(Mf, M, cell, team).to(x.dtype)) * using.unsqueeze(-1)
            h = self.rnn(torch.cat((x, read), -1).flatten(0, 1),
                         (h * alive.unsqueeze(-1)).flatten(0, 1)).view(B, A, -1).float()
            gate, value = self.write(h).float().chunk(2, -1)
        gate = torch.cat((torch.ones_like(obs[..., :FACTS]), torch.sigmoid(gate)), -1) * using.unsqueeze(-1)
        value = torch.cat((self.facts(obs), torch.tanh(value)), -1)
        M = self.update(Mf, cell, team, gate, value, using).view_as(M)
        return self.pi(h).float(), self.v(h).squeeze(-1).float(), (h, M)

    @staticmethod
    def facts(obs):
        """The sighting report an agent writes into its sector, straight from its senses:
        here I am; enemy tanks in how many radar directions; how close the nearest enemy
        tank, enemy base and friendly base are; my health; the worst threat I see to a
        friendly base; whether I am a base."""
        radar = lambda i: obs[..., RADAR + i * RS:RADAR + (i + 1) * RS]
        return torch.stack((torch.ones_like(obs[..., 0]), (radar(3) < 1).float().mean(-1), 1 - radar(3).amin(-1),
                            1 - radar(6).amin(-1), 1 - radar(5).amin(-1), obs[..., 4], radar(7).amax(-1), obs[..., 3]), -1)

    def read(self, Mf, M, cell, team):
        """What an agent sees on its team's map: the window x window sectors round it,
        and the whole map pooled down to coarse x coarse."""
        B, A = cell.shape[:2]
        S, C = self.cells, self.chan
        win = (cell.unsqueeze(2) + self.offsets).clamp(0, S - 1)                  # (B, A, K*K, 2)
        idx = (team * S * S).view(1, A, 1) + win[..., 0] * S + win[..., 1]
        local = Mf.gather(1, idx.flatten(1).unsqueeze(-1).expand(-1, -1, C)).view(B, A, -1)
        pooled = F.avg_pool2d(M.permute(0, 1, 4, 2, 3).reshape(B * 2, C, S, S), S // self.coarse)
        return torch.cat((local, pooled.reshape(B, 2, -1)[:, team]), -1)

    def update(self, Mf, cell, team, gate, value, using):
        """Gated write: every sector with agents in it moves toward their mean value by
        their mean gate; the rest fade a little."""
        B, A = cell.shape[:2]
        S, C = self.cells, self.chan
        idx = ((team * S * S).view(1, A) + cell[..., 0] * S + cell[..., 1]).unsqueeze(-1)
        add = lambda x: torch.zeros(B, 2 * S * S, x.shape[-1], device=Mf.device).scatter_add(1, idx.expand_as(x), x)
        n, g, gv = add(using.float().unsqueeze(-1)), add(gate), add(gate * value)
        M = Mf * self.decay
        return M + g / n.clamp(min=1) * (gv / g.clamp(min=1e-6) - M)


def dists(logits):
    return [Categorical(logits=l) for l in logits.split(HEADS, -1)]


def log_prob(ds, a, is_base):
    """Joint log-probability; the base-order head only counts for bases and the
    place-block head only for tanks (the arena ignores the other's samples)."""
    return (sum(d.log_prob(a[..., i]) for i, d in enumerate(ds[:3]))
            + ds[3].log_prob(a[..., 3]) * is_base + ds[4].log_prob(a[..., 4]) * (1 - is_base))


def act(policy, obs, alive, state, mute=None):
    logits, value, state = policy(obs, alive, state, mute)
    ds = dists(logits)
    a = torch.stack([d.sample() for d in ds], -1)
    return a, log_prob(ds, a, obs[..., 3]), value, state


def to_env(a):
    """Discrete choices -> the arena's (throttle, steer, fire, special, order, place)."""
    move = a[..., 0]
    return torch.stack((THROTTLE.to(a.device)[move // 3], STEER.to(a.device)[move % 3],
                        a[..., 1].float(), a[..., 2].float(), a[..., 3].float(), a[..., 4].float()), -1)


class Player:
    """A policy as a function (obs, alive) -> env action that keeps its own memory
    (GRU state and team maps) between calls; .state exposes them."""
    def __init__(self, policy, mute=None):
        self.policy, self.mute, self.state = policy, mute, None

    def __call__(self, obs, alive):
        if self.state is None:
            self.state = self.policy.memory(obs)
        a, _, _, self.state = act(self.policy, obs, alive, self.state, self.mute)
        return to_env(a)


def load(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=True)
    policy = Policy().to(device)
    policy.load_state_dict(ck['model'])
    return policy, ck.get('iteration', 0)


def save(policy, iteration, path):
    tmp = path + '.tmp'
    torch.save({'model': {k: v.detach().cpu() for k, v in policy.state_dict().items()}, 'iteration': iteration}, tmp)
    os.replace(tmp, path)            # atomic, so a viewer never loads half a file


@torch.no_grad()
def play(env, left, right):
    """Run every game of env to completion; team 0 is controlled by left(obs, alive),
    team 1 by right(obs, alive). Returns team 0's (win, loss) rate."""
    env.reset_rows(torch.ones(env.B, dtype=torch.bool, device=env.device))
    left_team = (env.team == 0).view(1, -1, 1)
    result = torch.full((env.B,), -2, device=env.device)
    for _ in range(env.limit):
        obs, alive = env.observe(), env.hp > 0
        _, _, done, winner = env.step(torch.where(left_team, left(obs, alive), right(obs, alive)))
        result = torch.where((result == -2) & done, winner, result)
        if (result > -2).all():
            break
    return (result == 0).float().mean().item(), (result == 1).float().mean().item()


def evaluate(policy, env):
    """Against the scripted raider, from both sides, and against a copy of itself cut
    off from the team map -- if the map carries anything useful, that copy should lose."""
    raider = lambda o, al: env.raider()
    w1, l1 = play(env, Player(policy), raider)
    l2, w2 = play(env, raider, Player(policy))
    cut = env.team == 1                                # team 1 can't read or write the map
    map_win, map_loss = play(env, Player(policy), Player(policy, mute=cut))
    return (w1 + w2) / 2, (l1 + l2) / 2, map_win, map_loss


# ---- training ------------------------------------------------------------------
GAMMA, LAM, CLIP, EPOCHS, MB_GAMES, LR = .99, .95, .2, 3, 32, 3e-4   # a minibatch is 32 games' whole rollouts


class Runner:
    """One arena size and its games' running state. The non-learning side of the first
    quarter of its games is half the scripted raider, half a frozen past snapshot of
    the policy; those agents are masked out of training."""
    def __init__(self, name, spec, policy, seed):
        dev = 'cuda'
        self.name = name
        self.env = Arena(batch=spec['games'], device=dev, seed=seed, grid=spec['grid'], tanks=spec['tanks'], bases=spec['bases'])
        self.env.t = torch.randint(self.env.limit, (self.env.B,), device=dev)   # stagger clocks so resets don't come in lockstep
        rows, self.q = torch.arange(self.env.B, device=dev), self.env.B // 4
        self.mask = (rows < self.q).unsqueeze(-1) & (self.env.team == rows.unsqueeze(-1) % 2)   # (B, A): not the learner
        self.raider = (self.mask & (rows < self.env.B // 8).unsqueeze(-1)).unsqueeze(-1)
        self.obs, self.alive = self.env.observe(), self.env.hp > 0
        self.state, self.state_past = policy.memory(self.obs), policy.memory(self.obs[:self.q])


def collect_rollout(run, net, past_net, steps):
    """Play `steps` turns of every game in one arena; memories carry over between
    rollouts (and are wiped when an agent dies or its game ends)."""
    env, q = run.env, run.q
    keys = ('obs', 'alive', 'act', 'logp', 'value', 'reward', 'term', 'done')
    data = {k: [] for k in keys}
    h0, m0 = run.state
    for _ in range(steps):
        a, logp, v, run.state = act(net, run.obs, run.alive, run.state)
        a_past, _, _, run.state_past = act(past_net, run.obs[:q], run.alive[:q], run.state_past)
        a[:q] = torch.where(run.mask[:q].unsqueeze(-1), a_past, a[:q])
        r, term, done, _ = env.step(torch.where(run.raider, env.raider(), to_env(a)))
        forget = lambda s, rows: (s[0] * (~term[rows]).unsqueeze(-1), s[1] * (~done[rows]).view(-1, 1, 1, 1, 1))
        run.state, run.state_past = forget(run.state, slice(None)), forget(run.state_past, slice(q))
        for k, x in zip(keys, (run.obs, run.alive, a, logp, v, r, term, done)):
            data[k].append(x)
        run.obs, run.alive = env.observe(), env.hp > 0
    data = {k: torch.stack(v) for k, v in data.items()}
    data['h0'], data['m0'], data['v_next'] = h0, m0, net(run.obs, run.alive, run.state)[1]
    return data


def advantages(d):
    """Generalised advantage estimation, cut wherever an agent died or its game ended."""
    term = d['term'].float()
    adv, gae = torch.zeros_like(d['reward']), torch.zeros_like(d['v_next'])
    for t in reversed(range(len(term))):
        nv = d['v_next'] if t == len(term) - 1 else d['value'][t + 1]
        gae = d['reward'][t] + GAMMA * nv * (1 - term[t]) - d['value'][t] + GAMMA * LAM * (1 - term[t]) * gae
        adv[t] = gae
    return adv, adv + d['value']


def ppo_update(net, policy, opt, d, learn):
    """PPO: replay each minibatch of games' rollouts in order through the GRU and the
    map (backprop through time from the memory the rollout began with). Returns
    losses and per-head entropies for logging."""
    adv, ret = advantages(d)
    term, done = d['term'].float(), d['done'].float()
    ent_sum, n = torch.zeros(len(HEADS), device=adv.device), 0
    for _ in range(EPOCHS):
        for g in torch.randperm(adv.shape[1], device=adv.device).split(MB_GAMES):
            h, M, logits, value = d['h0'][g], d['m0'][g], [], []
            for t in range(len(term)):
                lg, v, (h, M) = net(d['obs'][t, g], d['alive'][t, g], (h, M))
                h, M = h * (1 - term[t, g]).unsqueeze(-1), M * (1 - done[t, g]).view(-1, 1, 1, 1, 1)
                logits.append(lg); value.append(v)
            logits, value = torch.stack(logits), torch.stack(value)
            m, base = learn[:, g], d['obs'][:, g, :, 3] > .5
            ds = dists(logits)
            logp = log_prob(ds, d['act'][:, g], base.float())
            ent = torch.stack([x.entropy()[m].mean() for x in ds[:3]]
                              + [ds[3].entropy()[m & base].mean(), ds[4].entropy()[m & ~base].mean()])
            a = adv[:, g][m]
            a = (a - a.mean()) / (a.std() + 1e-8)
            ratio = (logp[m] - d['logp'][:, g][m]).exp()
            loss_pi = -torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a).mean()
            loss_v = (value[m] - ret[:, g][m]).pow(2).mean()
            loss = loss_pi + .5 * loss_v - (ENTROPY.to(a.device) * ent.nan_to_num()).sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), .5)
            opt.step()
            ent_sum += ent.detach(); n += 1
    return loss_pi.item(), loss_v.item(), (ent_sum / n).tolist()


def summarize(env, d, losses):
    """Everything worth watching in TensorBoard, from the games' own counters."""
    s = {k: float(v) for k, v in env.stats.items()}
    alive, tanks = d['alive'], d['alive'] & (d['obs'][..., 3] < .5)
    per_k = 1000 / max(alive.sum().item(), 1)              # counts are per 1000 alive agent-steps
    ratio = lambda a, b: s[a] / max(s[b], 1)
    log = {f'{k}_per_1k': s[k] * per_k for k in (
        'shots', 'blind_shots', 'kills', 'deaths', 'assists', 'base_kills', 'base_damage', 'bases_lost', 'defend_hits', 'heals',
        'rescues', 'pickups', 'deposits', 'builds', 'repairs', 'base_walls', 'blocks_placed', 'block_saves',
        'block_stops', 'stacked')}
    log.update({
        'accuracy': ratio('hits', 'shots'),
        'friendly_hit_fraction': s['friendly_hits'] / max(s['hits'] + s['friendly_hits'], 1),
        'block_saves_per_placed': ratio('block_saves', 'blocks_placed'),
        'wall_block_fraction': ratio('wall_blocks', 'blocks_placed'),
        'grouped_fraction': s['grouped'] * per_k / 1000,
        'idle_fraction': s['idle'] * per_k / 1000,
        'moving_fraction': ((d['act'][..., 0] // 3 != 1) & tanks).sum().item() / max(tanks.sum().item(), 1),
        'episode_turns': ratio('turns', 'games'),
        'decisive_fraction': ratio('decisive', 'games'),
        'reward_per_agent_step': d['reward'][alive].mean().item(),
        'loss_pi': losses[0], 'loss_v': losses[1],
    })
    for name, e in zip(('move', 'fire', 'special', 'base_order', 'place_block'), losses[2]):
        log[f'entropy_{name}'] = e
    return log


SHOWN = {'moving': 'moving_fraction', 'shots': 'shots_per_1k', 'blind': 'blind_shots_per_1k', 'acc': 'accuracy', 'kills': 'kills_per_1k', 'assists': 'assists_per_1k',
         'basedmg': 'base_damage_per_1k', 'defend': 'defend_hits_per_1k', 'grouped': 'grouped_fraction',
         'idle': 'idle_fraction', 'ff': 'friendly_hit_fraction', 'deposits': 'deposits_per_1k', 'builds': 'builds_per_1k',
         'rescues': 'rescues_per_1k', 'blocks': 'blocks_placed_per_1k', 'walls': 'wall_block_fraction',
         'stops': 'block_stops_per_1k', 'basewalls': 'base_walls_per_1k'}


def train(arenas, steps=32, minutes=None, output='tank_policy.pt', seed=1, logdir='runs', resume=False):
    dev = 'cuda'
    torch.manual_seed(seed)
    policy, it = load(output, dev) if resume and os.path.exists(output) else (Policy().to(dev), 0)
    net = torch.compile(policy)                                 # fused encoder/map/GRU kernels, same weights
    past = copy.deepcopy(policy)                                # the frozen snapshot the non-learning side plays
    past_net, pool = torch.compile(past), []
    opt = torch.optim.Adam(policy.parameters(), lr=LR, eps=1e-5)
    runs = [Runner(name, ARENAS[name], policy, seed + i) for i, name in enumerate(arenas)]
    eval_env = Arena(batch=128, device=dev, seed=seed + 99, auto_reset=False, grid=560)
    writer = SummaryWriter(f'{logdir}/{int(time.time())}')
    start, start_iter, agent_steps = time.monotonic(), it, 0
    while minutes is None or time.monotonic() - start < minutes * 60:
        it += 1
        run = runs[it % len(runs)]                              # the arena sizes take turns
        run.env.reset_stats()
        if it % 50 == 1:                                        # new past opponent: a random snapshot from the pool
            pool = (pool + [copy.deepcopy(policy.state_dict())])[-10:]
            past.load_state_dict(random.choice(pool))
        with torch.no_grad():
            d = collect_rollout(run, net, past_net, steps)
        losses = ppo_update(net, policy, opt, d, d['alive'] & ~run.mask)
        save(policy, it, output)
        if it % 100 == 0:                                       # snapshots for head-to-heads and the report
            os.makedirs('checkpoints', exist_ok=True)
            save(policy, it, f'checkpoints/{it:05d}.pt')

        elapsed, agent_steps = time.monotonic() - start, agent_steps + steps * run.env.B * run.env.A
        log = {f'{run.name}/{k}': v for k, v in summarize(run.env, d, losses).items()}
        log['time/agent_steps_per_s'], log['time/minutes'] = agent_steps / elapsed, elapsed / 60
        line = f"it {it} {elapsed / 60:.1f}min {run.name:6} " + ' '.join(f"{k}={log[f'{run.name}/{v}']:.2f}" for k, v in SHOWN.items())
        if it % 100 == 0:                                       # full games on the medium board, so this is slow-ish
            win, loss, mw, ml = evaluate(policy, eval_env)
            log.update({'eval/win_vs_raider': win, 'eval/loss_vs_raider': loss,
                        'eval/map_on_vs_off_win': mw, 'eval/map_on_vs_off_loss': ml})
            line += f" | vs raider win={win:.0%} loss={loss:.0%} | map on vs off win={mw:.0%} loss={ml:.0%}"
        for k, v in log.items():
            writer.add_scalar(k, v, it)
        print(line, flush=True)
    writer.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arenas', default=','.join(ARENAS), help='which board sizes to train on, comma-separated')
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--minutes', type=float, default=None, help='stop after this long (default: run until stopped)')
    p.add_argument('--output', default='tank_policy.pt')
    p.add_argument('--resume', action='store_true', help='continue from --output instead of starting fresh')
    a = p.parse_args()
    train(a.arenas.split(','), a.steps, a.minutes, a.output, resume=a.resume)
