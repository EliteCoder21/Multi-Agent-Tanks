"""Collect everything the report needs from the current training run.

    python report/collect.py            # writes report/data/

  scalars.csv     every TensorBoard scalar of the latest run: step, tag, value
  behaviour.csv   behaviour probes per checkpoint (full self-play games on the training board)
  ladder.csv      the final policy against earlier checkpoints, from both sides
  radio.json      radio ablation (talking vs muted copy) and who agents listen to
  raider.json     the final policy against the scripted raider
  speed.json      throughput of the simulator, the policy and a training iteration
"""
import csv, glob, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arena
from arena import Arena, BASE, SCOUT, VISION_SECTORS
from train import load, player, play, act

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'report', 'data')
DEV, GRID, GAMES = 'cuda', 560, 128


def scalars():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run = sorted(glob.glob(os.path.join(ROOT, 'runs', '*')))[-1]
    ea = EventAccumulator(run, size_guidance={'scalars': 0})
    ea.Reload()
    with open(os.path.join(OUT, 'scalars.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('step', 'tag', 'value'))
        for tag in ea.Tags()['scalars']:
            for e in ea.Scalars(tag):
                w.writerow((e.step, tag, e.value))


def attention(policy, obs, alive):
    """The radio's attention weights (B, 2, heads, n, n): listener i -> speaker j."""
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
        x = policy.enc(obs)
        n = x.shape[-2] // 2
        q, k, _ = policy.radio(x).unflatten(-2, (2, n)).unflatten(-1, (3, policy.heads, -1)).unbind(-3)
        score = torch.einsum('...ihd,...jhd->...hij', q, k).float() / q.shape[-1] ** .5
        hears = alive.unflatten(-1, (2, n)).unsqueeze(-2).unsqueeze(-2) & ~torch.eye(n, dtype=torch.bool, device=obs.device)
        return torch.softmax(score.masked_fill(~hears, -1e9), -1) * hears, hears


@torch.no_grad()
def probe(env, policy, radio=False):
    """Play every game of env to the end with policy on both sides; return behaviour
    measures (and, with radio=True, who listens to whom)."""
    env.reset_rows(torch.ones(env.B, dtype=torch.bool, device=DEV))
    env.reset_stats()
    brain = player(policy)
    c = {k: torch.zeros((), device=DEV) for k in (
        'fire_see', 'n_see', 'fire_blind', 'n_blind', 'd_hurt', 'n_hurt', 'd_ok', 'n_ok', 'def_threat', 'n_threat',
        'def_calm', 'n_calm', 'siege', 'n_siege', 'grouped', 'n_fighter', 'first_base_kill', 'n_base_kill',
        'att_near', 'uni_near', 'att_seeing', 'uni_seeing', 'att_base', 'uni_base', 'att_guard', 'uni_guard')}
    v0 = 16 + 2 * VISION_SECTORS                                   # vision enemy channel
    first_kill = torch.full((env.B,), -1., device=DEV)
    for t in range(env.limit):
        obs, alive = env.observe(), env.hp > 0
        action = brain(obs, alive)
        role, pos = env.role, env.pos
        gun = alive & (role != SCOUT) & (role != BASE)
        sees = (obs[..., v0:v0 + VISION_SECTORS] < 1).any(-1)
        fire = action[..., 2] > .5
        c['fire_see'] += (gun & sees & fire).sum(); c['n_see'] += (gun & sees).sum()
        c['fire_blind'] += (gun & ~sees & fire).sum(); c['n_blind'] += (gun & ~sees).sum()
        dd = torch.cdist(pos, pos)
        tank = alive & (role != BASE)
        enemy_tank = (tank[:, None] & ~env.same)
        d_enemy = torch.where(enemy_tank, dd, torch.full_like(dd, 1e5)).amin(-1)
        frac = env.hp / env.max_hp
        near_fight = gun & (d_enemy < 60)
        hurt, ok = near_fight & (frac < .5), near_fight & (frac > .99)
        c['d_hurt'] += d_enemy[hurt].sum(); c['n_hurt'] += hurt.sum()
        c['d_ok'] += d_enemy[ok].sum(); c['n_ok'] += ok.sum()
        ally = torch.where(tank[:, None] & env.same & ~env.eye, dd, torch.full_like(dd, 1e5))
        second = ally.topk(2, -1, largest=False).values[..., 1]
        c['grouped'] += (gun & (second <= arena.FORM_MAX)).sum(); c['n_fighter'] += gun.sum()
        base = alive & (role == BASE)
        threatened = base & (enemy_tank & (dd < 100)).any(-1)      # enemy tanks within 100 of the base
        own_near = (tank[:, None] & env.same & (dd < 60)).sum(-1).float()
        foe_near = (tank[:, None] & ~env.same & (dd < 60)).sum(-1).float()
        c['def_threat'] += own_near[threatened].sum(); c['n_threat'] += threatened.sum()
        c['def_calm'] += own_near[base & ~threatened].sum(); c['n_calm'] += (base & ~threatened).sum()
        besieged = base & (foe_near > 0)                           # attackers around an enemy base
        c['siege'] += foe_near[besieged].sum(); c['n_siege'] += besieged.sum()
        if radio and t % 10 == 0:
            w, hears = attention(policy, obs, alive)               # (B,2,h,n,n)
            n = env.N
            w, hears = w.mean(2), hears[:, :, 0].float()           # average the heads
            uniform = hears / hears.sum(-1, keepdim=True).clamp(min=1)
            dteam = dd.unflatten(1, (2, n)).unflatten(-1, (2, n)).diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2)   # (B,2,n,n)
            listener = (alive & (role != BASE)).unflatten(-1, (2, n)).unsqueeze(-1).float()
            for key, speaker in (('near', dteam < 30),
                                 ('seeing', sees.unflatten(-1, (2, n)).unsqueeze(-2).expand_as(dteam)),
                                 ('base', (role == BASE).unflatten(-1, (2, n)).unsqueeze(-2).expand_as(dteam)),
                                 ('guard', threatened.unflatten(-1, (2, n)).unsqueeze(-2).expand_as(dteam))):   # bases under attack
                c['att_' + key] += (w * speaker * listener).sum(); c['uni_' + key] += (uniform * speaker * listener).sum()
        _, _, done, _ = env.step(action)
        lost = (env.hp[:, env.base_slots] <= 0).any(-1) & (first_kill < 0)
        first_kill = torch.where(lost, float(t), first_kill)
        if done.all():
            break
    s = {k: float(v) for k, v in env.stats.items()}
    f = {k: float(v) for k, v in c.items()}
    g = env.B
    r = lambda a, b: f[a] / max(f[b], 1)
    out = {
        'fire_rate_enemy_in_sight': r('fire_see', 'n_see'), 'fire_rate_nothing_in_sight': r('fire_blind', 'n_blind'),
        'dist_to_enemy_hurt': r('d_hurt', 'n_hurt'), 'dist_to_enemy_healthy': r('d_ok', 'n_ok'),
        'grouped_fraction': r('grouped', 'n_fighter'),
        'defenders_threatened_base': r('def_threat', 'n_threat'), 'defenders_calm_base': r('def_calm', 'n_calm'),
        'siege_group_size': r('siege', 'n_siege'),
        'first_base_loss_turn': float(first_kill[first_kill >= 0].mean()) if (first_kill >= 0).any() else None,
        'games_with_base_lost': float((first_kill >= 0).float().mean()),
        'accuracy': s['hits'] / max(s['shots'], 1), 'friendly_hit_fraction': s['friendly_hits'] / max(s['hits'] + s['friendly_hits'], 1),
        'wall_block_fraction': s['wall_blocks'] / max(s['blocks_placed'], 1), 'saves_per_block': s['block_saves'] / max(s['blocks_placed'], 1),
        'decisive_fraction': s['decisive'] / max(s['games'], 1), 'turns_per_game': s['turns'] / max(s['games'], 1),
    }
    for k in ('kills', 'deaths', 'assists', 'base_kills', 'base_damage', 'defend_hits', 'shots', 'heals', 'rescues',
              'pickups', 'deposits', 'builds', 'repairs', 'base_walls', 'blocks_placed', 'block_stops'):
        out[k + '_per_game'] = s[k] / g
    if radio:
        for k in ('near', 'seeing', 'base', 'guard'):
            out['attention_lift_' + k] = r('att_' + k, 'uni_' + k)   # >1: listens to them more than to a random teammate
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    scalars()
    cks = sorted(glob.glob(os.path.join(ROOT, 'checkpoints', '*.pt')))
    final_path = os.path.join(ROOT, 'tank_policy.pt')
    final, final_it = load(final_path, DEV)
    env = Arena(batch=GAMES, device=DEV, seed=7, auto_reset=False, grid=GRID)
    picks = cks[::max(1, len(cks) // 8)] + ([final_path] if final_path not in cks else [])
    rows = []
    for path in picks:
        policy, it = load(path, DEV)
        row = {'iteration': it, **probe(env, policy, radio=True)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, 'behaviour.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1]))
        w.writeheader(); w.writerows(rows)

    ladder = []
    for path in cks[::max(1, len(cks) // 6)]:
        other, it = load(path, DEV)
        w1, l1 = play(env, player(final), player(other))
        l2, w2 = play(env, player(other), player(final))
        ladder.append({'final_iteration': final_it, 'opponent_iteration': it, 'win': (w1 + w2) / 2, 'loss': (l1 + l2) / 2})
        print(ladder[-1], flush=True)
    with open(os.path.join(OUT, 'ladder.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ladder[0]) if ladder else ['final_iteration'])
        w.writeheader(); w.writerows(ladder)

    muted = env.team == 1
    tw, tl = play(env, player(final), player(final, mute=muted))
    mw, ml = play(env, player(final, mute=env.team == 0), player(final))       # swap sides
    json.dump({'iteration': final_it, 'talking_vs_muted_win': (tw + ml) / 2, 'talking_vs_muted_loss': (tl + mw) / 2,
               'final_probe': rows[-1]}, open(os.path.join(OUT, 'radio.json'), 'w'), indent=1)
    raider = lambda o, al: env.raider()
    w1, l1 = play(env, player(final), raider)
    l2, w2 = play(env, raider, player(final))
    json.dump({'iteration': final_it, 'win': (w1 + w2) / 2, 'loss': (l1 + l2) / 2}, open(os.path.join(OUT, 'raider.json'), 'w'), indent=1)

    big = Arena(batch=1024, device=DEV, grid=GRID)
    obs, alive = big.observe(), big.hp > 0
    h = final.memory(obs)
    def timed(fn, n=20):
        fn(); torch.cuda.synchronize(); t = time.time()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.time() - t) / n
    step = timed(lambda: big.step(torch.zeros(1024, big.A, 6, device=DEV)))
    see = timed(lambda: big.observe())
    think = timed(lambda: act(final, obs, alive, h))
    json.dump({'games': 1024, 'agents_per_game': big.A, 'step_s': step, 'observe_s': see, 'policy_s': think,
               'agent_steps_per_s_sim_only': 1024 * big.A / (step + see),
               'agent_steps_per_s_with_policy': 1024 * big.A / (step + see + think)},
              open(os.path.join(OUT, 'speed.json'), 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
