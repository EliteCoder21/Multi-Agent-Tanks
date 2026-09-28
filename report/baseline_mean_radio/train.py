"""Self-play PPO with a recurrent policy: the network drives, shoots, builds and talks.

Policy: observation -> 2-layer encoder -> + teammates' radio -> GRU memory -> actions.
The GRU (512 units) lets an agent remember what it saw a few turns ago -- where an
enemy went behind a wall, which base is under attack, what the radio said. It is reset
when an agent dies or its game ends.

Team radio: every agent broadcasts a 32-number message computed from what it senses,
and hears the average of its living teammates' messages in the same turn. It is all
one differentiable pass, so a listener's policy gradient trains the speaker.

Thousands of games run side by side and reset independently; training is a stream of
32-turn rollouts, and PPO replays each game's rollout in order through the GRU
(backprop through time), starting from the memory the rollout began with. A quarter
of the games put one side under someone else: half of those under the scripted
raider, half under a frozen past snapshot of the policy, so it keeps beating older
styles instead of chasing its own tail. The latest weights go to tank_policy.pt
after every iteration so viewer.py can watch while this runs.
"""
import argparse
import copy
import os
import random
import time
import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import arena
from arena import Arena, OBS

HEADS = (9, 2, 2, len(arena.ORDERS), 2)   # move (3 throttle x 3 steer), fire, special, base order, place block
THROTTLE = torch.tensor([-.5, 0., 1.])
STEER = torch.tensor([-1., 0., 1.])
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# entropy bonus per head; placing a block has an immediate cost and a delayed, chancy
# payoff, so with the usual .01 that head collapsed to "never" and cover was never found
ENTROPY = torch.tensor([.01, .01, .01, .01, .05])


class Policy(nn.Module):
    def __init__(self, hidden=512, comm=32):
        super().__init__()
        self.hidden = hidden
        self.enc = nn.Sequential(nn.Linear(OBS, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.say = nn.Linear(hidden, comm)
        self.rnn = nn.GRUCell(hidden + comm, hidden)
        self.pi = nn.Linear(hidden, sum(HEADS))
        self.v = nn.Linear(hidden, 1)
        with torch.no_grad():            # place-block head starts near "never", or a fresh policy carpets the board
            self.pi.bias[-2:] = torch.tensor([3., -3.])

    def forward(self, obs, alive, h, mute=None):
        """obs (..., A, OBS), alive (..., A), memory h (..., A, hidden) -> logits, value,
        new memory. Dead agents' memory is wiped. mute silences those agents' messages
        -- used to test whether the radio matters."""
        with torch.autocast('cuda', torch.bfloat16, enabled=obs.is_cuda):   # ~2x faster on the GPU
            x = self.enc(obs)
            talking = alive if mute is None else alive & ~mute
            msg = torch.tanh(self.say(x)) * talking.unsqueeze(-1)
            n = obs.shape[-2] // 2                       # agents per team: team 0 first, then team 1
            by_team, talk = msg.unflatten(-2, (2, n)), talking.unflatten(-1, (2, n))
            heard = (by_team.sum(-2, keepdim=True) - by_team) / (talk.sum(-1, keepdim=True) - talk.float()).clamp(min=1).unsqueeze(-1)
            x = torch.cat((x, heard.flatten(-3, -2).to(x.dtype)), -1)
            h = self.rnn(x.flatten(0, -2), (h * alive.unsqueeze(-1)).flatten(0, -2)).view(h.shape).float()
            return self.pi(h).float(), self.v(h).squeeze(-1).float(), h

    def memory(self, obs):
        return obs.new_zeros(*obs.shape[:-1], self.hidden)


def dists(logits):
    return [Categorical(logits=l) for l in logits.split(HEADS, -1)]


def log_prob(ds, a, is_base):
    """Joint log-probability; the base-order head only counts for bases and the
    place-block head only for tanks (the arena ignores the other's samples)."""
    return (sum(d.log_prob(a[..., i]) for i, d in enumerate(ds[:3]))
            + ds[3].log_prob(a[..., 3]) * is_base + ds[4].log_prob(a[..., 4]) * (1 - is_base))


def act(policy, obs, alive, h, mute=None):
    logits, value, h = policy(obs, alive, h, mute)
    ds = dists(logits)
    a = torch.stack([d.sample() for d in ds], -1)
    return a, log_prob(ds, a, obs[..., 3]), value, h


def to_env(a):
    """Discrete choices -> the arena's (throttle, steer, fire, special, order, place)."""
    move = a[..., 0]
    return torch.stack((THROTTLE.to(a.device)[move // 3], STEER.to(a.device)[move % 3],
                        a[..., 1].float(), a[..., 2].float(), a[..., 3].float(), a[..., 4].float()), -1)


def player(policy, mute=None):
    """A policy as an opponent function (obs, alive) -> env action, keeping its own memory."""
    h = None
    def play(obs, alive):
        nonlocal h
        h = policy.memory(obs) if h is None else h
        a, _, _, h = act(policy, obs, alive, h, mute)
        return to_env(a)
    return play


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
    """Against the scripted raider, from both sides, and against a copy of itself
    whose radio is silenced -- if the messages mean anything, silence should lose."""
    raider = lambda o, al: env.raider()
    w1, l1 = play(env, player(policy), raider)
    l2, w2 = play(env, raider, player(policy))
    muted = env.team == 1                              # team 1 can't hear anyone
    talk_win, talk_loss = play(env, player(policy), player(policy, mute=muted))
    return (w1 + w2) / 2, (l1 + l2) / 2, talk_win, talk_loss


def train(games=1024, steps=32, minutes=None, output='tank_policy.pt', seed=1, logdir='runs', resume=False, grid=560):
    dev = 'cuda'
    torch.manual_seed(seed)
    # a smaller board than the viewer's (same armies, a quarter of the area) so tanks
    # meet far more often; everything the policy sees is scaled so it plays either size
    env = Arena(batch=games, device=dev, seed=seed, grid=grid)
    env.t = torch.randint(env.limit, (games,), device=dev)     # stagger game clocks so resets don't come in lockstep
    eval_env = Arena(batch=128, device=dev, seed=seed + 1, auto_reset=False, grid=grid)
    policy, it = load(output, dev) if resume and os.path.exists(output) else (Policy().to(dev), 0)
    start_iter = it
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    writer = SummaryWriter(f'{logdir}/{int(time.time())}')
    gamma, lam, clip, epochs, mb_games = .99, .95, .2, 3, 32      # a minibatch is 32 games' whole rollouts
    rows, q = torch.arange(games, device=dev), games // 4     # the first q games have a non-learner side
    opp = (rows < q).unsqueeze(-1) & (env.team == rows.unsqueeze(-1) % 2)   # (B, A): not the learner
    bot = (opp & (rows < games // 8).unsqueeze(-1)).unsqueeze(-1)                     # of those, half are the raider
    pool, past = [], copy.deepcopy(policy)
    net, past_net = torch.compile(policy), torch.compile(past)    # fused GRU/encoder kernels, same weights
    obs, alive = env.observe(), env.hp > 0
    h, h_past = policy.memory(obs), policy.memory(obs[:q])
    start = time.monotonic()
    while minutes is None or time.monotonic() - start < minutes * 60:
        it += 1
        env.reset_stats()
        if it % 50 == 1:                     # new past opponent: a random snapshot from the pool
            pool = (pool + [copy.deepcopy(policy.state_dict())])[-10:]
            past.load_state_dict(random.choice(pool))
        H0 = h
        O, AL, ACT, LOGP, V, R, TERM = [], [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(steps):
                a, logp, v, h = act(net, obs, alive, h)
                a_past, _, _, h_past = act(past_net, obs[:q], alive[:q], h_past)
                a[:q] = torch.where(opp[:q].unsqueeze(-1), a_past, a[:q])
                r, term, _, _ = env.step(torch.where(bot, env.raider(), to_env(a)))
                keep = (~term).unsqueeze(-1)             # died or game over: forget
                h, h_past = h * keep, h_past * keep[:q]
                for x, y in zip((O, AL, ACT, LOGP, V, R, TERM), (obs, alive, a, logp, v, r, term)):
                    x.append(y)
                obs, alive = env.observe(), env.hp > 0
            v_next = net(obs, alive, h)[1]
            O, AL, ACT, LOGP, V, R, TERM = map(torch.stack, (O, AL, ACT, LOGP, V, R, TERM))
            done = TERM.float()
            adv, gae = torch.zeros_like(R), torch.zeros_like(v_next)
            for t in reversed(range(steps)):
                nv = v_next if t == steps - 1 else V[t + 1]
                gae = R[t] + gamma * nv * (1 - done[t]) - V[t] + gamma * lam * (1 - done[t]) * gae
                adv[t] = gae
            ret = adv + V
            learn = AL & ~opp                            # the frozen opponent's and raider's agents don't train

        ent_sum, n_upd = torch.zeros(len(HEADS), device=dev), 0
        for _ in range(epochs):
            for g in torch.randperm(games, device=dev).split(mb_games):
                hg, logits, value = H0[g], [], []
                for t in range(steps):               # replay the rollout through the memory, in order
                    lg, v, hg = net(O[t, g], AL[t, g], hg)
                    hg = hg * (1 - done[t, g]).unsqueeze(-1)
                    logits.append(lg); value.append(v)
                logits, value = torch.stack(logits), torch.stack(value)
                m, base = learn[:, g], O[:, g, :, 3] > .5
                ds = dists(logits)
                logp = log_prob(ds, ACT[:, g], base.float())
                ent = torch.stack([d.entropy()[m].mean() for d in ds[:3]]
                                  + [ds[3].entropy()[m & base].mean(), ds[4].entropy()[m & ~base].mean()])
                a_ = adv[:, g][m]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                ratio = (logp[m] - LOGP[:, g][m]).exp()
                loss_pi = -torch.min(ratio * a_, ratio.clamp(1 - clip, 1 + clip) * a_).mean()
                loss_v = (value[m] - ret[:, g][m]).pow(2).mean()
                loss = loss_pi + .5 * loss_v - (ENTROPY.to(dev) * ent.nan_to_num()).sum()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), .5)
                opt.step()
                ent_sum += ent.detach(); n_upd += 1
        save(policy, it, output)

        s = {k: float(v) for k, v in env.stats.items()}
        elapsed = time.monotonic() - start
        per_k = 1000 / max(AL.sum().item(), 1)              # counts are per 1000 alive agent-steps
        ratio = lambda a, b: s[a] / max(s[b], 1)
        log = {f'play/{k}_per_1k': s[k] * per_k for k in (
            'shots', 'kills', 'base_kills', 'heals', 'rescues', 'pickups', 'deposits', 'builds', 'repairs', 'base_walls',
            'blocks_placed', 'blocks_broken', 'block_stops', 'assists', 'flank_hits', 'defend_hits', 'stacked', 'bases_lost')}
        log.update({
            'play/accuracy': ratio('hits', 'shots'),
            'play/friendly_hit_fraction': s['friendly_hits'] / max(s['hits'] + s['friendly_hits'], 1),
            'play/block_saves_per_placed': ratio('block_saves', 'blocks_placed'),
            'play/wall_block_fraction': ratio('wall_blocks', 'blocks_placed'),
            'play/grouped_fraction': s['grouped'] * per_k / 1000,
            'play/idle_fraction': s['idle'] * per_k / 1000,
            'play/episode_turns': ratio('turns', 'games'),
            'play/decisive_fraction': ratio('decisive', 'games'),
            'train/reward_per_agent_step': R[AL].mean().item(),
            'train/loss_pi': loss_pi.item(), 'train/loss_v': loss_v.item(),
            'time/agent_steps_per_s': steps * games * env.A * (it - start_iter) / elapsed, 'time/minutes': elapsed / 60,
        })
        for name, e in zip(('move', 'fire', 'special', 'base_order', 'place_block'), ent_sum / n_upd):
            log[f'entropy/{name}'] = e.item()
        short = {'acc': 'accuracy', 'kills': 'kills_per_1k', 'deposits': 'deposits_per_1k', 'builds': 'builds_per_1k',
                 'basewalls': 'base_walls_per_1k', 'blocks': 'blocks_placed_per_1k', 'saves/block': 'block_saves_per_placed', 'stops': 'block_stops_per_1k',
                 'assists': 'assists_per_1k', 'flank': 'flank_hits_per_1k', 'defend': 'defend_hits_per_1k',
                 'ff': 'friendly_hit_fraction', 'stacked': 'stacked_per_1k',
                 'grouped': 'grouped_fraction', 'idle': 'idle_fraction', 'rescues': 'rescues_per_1k'}
        line = f"it {it} {elapsed / 60:.1f}min " + ' '.join(f"{k}={log['play/' + v]:.2f}" for k, v in short.items())
        if it % 100 == 0:                      # full games, so this is slow-ish
            win, loss, tw, tl = evaluate(policy, eval_env)
            log.update({'eval/win_vs_raider': win, 'eval/loss_vs_raider': loss,
                        'eval/radio_on_vs_muted_win': tw, 'eval/radio_on_vs_muted_loss': tl})
            line += f" | vs raider win={win:.0%} loss={loss:.0%} | radio on vs muted win={tw:.0%} loss={tl:.0%}"
        for k, v in log.items():
            writer.add_scalar(k, v, it)
        print(line, flush=True)
    writer.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=1024)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--minutes', type=float, default=None, help='stop after this long (default: run until stopped)')
    p.add_argument('--output', default='tank_policy.pt')
    p.add_argument('--resume', action='store_true', help='continue from --output instead of starting fresh')
    p.add_argument('--grid', type=int, default=560, help='training board size (the viewer uses the full 1125)')
    a = p.parse_args()
    train(a.games, a.steps, a.minutes, a.output, resume=a.resume, grid=a.grid)
