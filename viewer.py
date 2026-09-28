"""Watch the current policy play, live, while train.py keeps running -- or take over
one of the tanks (C) and fight it: top-down with the tank facing the mouse, or in a
first-person view on the ground (V switches).

The viewer re-loads tank_policy.pt whenever training writes a new one, and rolls
straight into a fresh game when one ends. Runs on the CPU so it doesn't compete with
training for the GPU.
"""
import argparse
import math
import os
import numpy as np
import pygame
import torch
import arena
from arena import Arena, ROLE_NAMES, BASE, SCOUT, COMMANDER, wrap
from train import Policy, player, load

# one game's tensors are tiny: spreading each op over all 20 cores costs more in
# thread hand-offs than it saves (a turn took ~40 ms on 20 threads, ~14 ms on 8)
torch.set_num_threads(min(8, os.cpu_count()))

BLUE, RED = (74, 205, 255), (255, 109, 78)
HUD_H = 64
OVERVIEW = 4                  # the zoomed-out layer is the map at 1/4 resolution
# first-person renderer: a raycaster at low resolution, scaled up
FPS_W, FPS_H, FOV, FPS_RANGE, STEP = 320, 200, 1.2, 80., .15
FOCAL = FPS_W / 2 / math.tan(FOV / 2)
EYE, WALL_H, BLOCK_H = .6, 2.5, 1.       # eye height; walls tower, placed blocks are chest-high barricades
MOUSE_SENS = .004


def label(surface, font, s, xy, color=(230, 239, 226)):
    surface.blit(font.render(str(s), True, color), xy)


class Viewer:
    def __init__(self, path, seed, opponent, grid=arena.GRID, tanks=sum(arena.START), bases=arena.BASES):
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 960), pygame.RESIZABLE)
        pygame.display.set_caption('Tank Arena')
        self.font = pygame.font.SysFont('dejavusansmono', 15)
        self.small = pygame.font.SysFont('dejavusansmono', 12)
        self.big = pygame.font.SysFont('dejavusans', 24, bold=True)
        self.clock = pygame.time.Clock()
        self.path, self.seed, self.opponent = path, seed, opponent
        self.grid, self.tanks, self.bases = grid, tanks, bases
        self.mtime, self.iteration, self.policy = 0, None, Policy()
        self.brain = player(self.policy)
        self.reload()
        self.new_game()
        self.zoom = self.fit_zoom()
        self.selected, self.follow = None, False
        self.playing, self.speed, self.accum = True, 16, 0.
        self.note, self.note_at = None, 0
        rows = np.arange(FPS_H)[None, :, None]                    # sky and ground, darker toward the horizon
        sky = np.array([70, 110, 150]) * (.55 + .45 * (1 - rows / (FPS_H / 2)))
        ground = np.array([58, 82, 48]) * (.45 + .55 * (rows - FPS_H / 2) / (FPS_H / 2))
        self.backdrop = np.where(rows < FPS_H / 2, sky, ground).repeat(FPS_W, 0).astype(np.uint8)

    def reload(self):
        if os.path.exists(self.path) and os.path.getmtime(self.path) != self.mtime:
            try:
                self.policy, self.iteration = load(self.path)
                self.brain = player(self.policy)
                self.mtime = os.path.getmtime(self.path)
            except Exception:
                pass                     # caught mid-write; try again next time

    def new_game(self):
        self.seed += 1
        self.env = Arena(batch=1, seed=self.seed, auto_reset=False, grid=self.grid, tanks=self.tanks, bases=self.bases)
        self.walls = arena._map(self.env.maps[0], self.grid, self.bases)[0]
        rgb = np.where(self.walls[..., None], np.uint8([38, 34, 28]), np.uint8([58, 82, 48]))
        self.world = pygame.surfarray.make_surface(rgb)
        self.overview = pygame.transform.smoothscale(self.world, (self.grid // OVERVIEW, self.grid // OVERVIEW))
        self.brain = player(self.policy)        # fresh memory every game
        self.winner, self.ended_at = None, None
        self.camera = np.array([self.grid / 2, self.grid / 2])
        self.release()

    # ---- camera ------------------------------------------------------------
    def fit_zoom(self):
        w, h = self.screen.get_size()
        return min(w, h - HUD_H) / self.grid

    def to_screen(self, p):
        w, h = self.screen.get_size()
        return (int(w / 2 + (p[0] - self.camera[0]) * self.zoom),
                int(HUD_H + (h - HUD_H) / 2 + (p[1] - self.camera[1]) * self.zoom))

    def to_world(self, xy):
        w, h = self.screen.get_size()
        return np.array([self.camera[0] + (xy[0] - w / 2) / self.zoom,
                         self.camera[1] + (xy[1] - HUD_H - (h - HUD_H) / 2) / self.zoom])

    def on_screen(self, pts, margin=20):
        w, h = self.screen.get_size()
        s = (pts - self.camera) * self.zoom + np.array([w / 2, HUD_H + (h - HUD_H) / 2])
        return (s[..., 0] > -margin) & (s[..., 0] < w + margin) & (s[..., 1] > HUD_H - margin) & (s[..., 1] < h + margin)

    # ---- top-down drawing ----------------------------------------------------
    def draw_world(self):
        w, h = self.screen.get_size()
        self.screen.fill((14, 20, 18))
        x0, y0 = self.to_world((0, HUD_H))
        x1, y1 = self.to_world((w, h))
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.grid, int(x1) + 1), min(self.grid, int(y1) + 1)
        if x1 > x0 and y1 > y0:
            src, f = (self.world, 1) if self.zoom >= 1 / OVERVIEW * 2 else (self.overview, OVERVIEW)
            rect = pygame.Rect(x0 // f, y0 // f, max(1, (x1 - x0) // f), max(1, (y1 - y0) // f)).clip(src.get_rect())
            size = (int(rect.w * f * self.zoom), int(rect.h * f * self.zoom))
            self.screen.blit(pygame.transform.scale(src.subsurface(rect), size), self.to_screen((rect.x * f, rect.y * f)))
        e = self.env
        # placed blocks: bluish or reddish sand by owner, darkening as they take hits
        cells = torch.nonzero(e.blocks[0, :-1] > 0).flatten()
        if len(cells):
            hp = e.blocks[0, cells].numpy()
            red = (e.block_owner[0, cells] >= e.N).numpy()
            xy = np.stack(((cells // e.BG).numpy(), (cells % e.BG).numpy()), -1) * arena.BLOCK
            size = max(2, int(arena.BLOCK * self.zoom))
            vis = self.on_screen(xy + arena.BLOCK / 2)
            for (bx, by), hh, r in zip(xy[vis], hp[vis], red[vis]):
                shade = .55 + .45 * hh / arena.BLOCK_HP
                col = [int(c * shade) for c in ((230, 160, 110) if r else (150, 180, 200))]
                x, y = self.to_screen((bx, by))
                pygame.draw.rect(self.screen, col, (x, y, size, size))
                if size >= 6:
                    pygame.draw.rect(self.screen, (90, 70, 40), (x, y, size, size), 1)
        hearts = e.heart_pos[0][e.heart_alive[0]].numpy()
        for p in hearts[self.on_screen(hearts)]:
            if self.zoom < 1:
                self.screen.set_at(self.to_screen(p), (190, 60, 90))      # a single dim pixel when zoomed out
            else:
                pygame.draw.circle(self.screen, (255, 70, 110), self.to_screen(p), max(2, int(.9 * self.zoom)))
        bp, bv, alive = e.b_pos[0].numpy(), e.b_vel[0].numpy(), e.b_alive[0].numpy()
        for i, k in zip(*np.nonzero(alive & self.on_screen(bp))):
            col = (255, 217, 110) if e.b_pierce[0, i, k] else (255, 240, 170)
            pygame.draw.line(self.screen, col, self.to_screen(bp[i, k] - bv[i, k]), self.to_screen(bp[i, k]), 2)
        for i in range(e.A):
            self.draw_agent(i)

    def draw_agent(self, i):
        e = self.env
        if e.hp[0, i] <= 0:
            return
        p = e.pos[0, i].numpy()
        if not self.on_screen(p, 60):
            return
        c = np.array(self.to_screen(p))
        role = e.role[0, i].item()
        color = BLUE if i < e.N else RED
        frac = (e.hp[0, i] / e.max_hp[0, i]).item()
        if role == BASE:
            size = max(14, arena.BASE_CLEAR * self.zoom)
            pygame.draw.rect(self.screen, color, (c[0] - size / 2, c[1] - size / 2, size, size))
            pygame.draw.rect(self.screen, (10, 20, 20), (c[0] - size / 2, c[1] - size / 2, size, size), 2)
            bar = max(20, size)
            pygame.draw.rect(self.screen, (20, 25, 25), (c[0] - bar / 2, c[1] - size / 2 - 7, bar, 4))
            pygame.draw.rect(self.screen, (102, 230, 101), (c[0] - bar / 2, c[1] - size / 2 - 7, int(bar * frac), 4))
            label(self.screen, self.small, f'♥{int(e.supply[0, i])}', (c[0] + size / 2 + 3, c[1] - 8), (255, 150, 170))
            return
        size = max(4, (1.3 if role == COMMANDER else 1.) * self.zoom)
        a = e.heading[0, i].item()
        fw = np.array([math.cos(a), math.sin(a)])
        sd = np.array([-fw[1], fw[0]])
        pts = [c + fw * size * 1.2, c - fw * size + sd * size * .8, c - fw * size - sd * size * .8]
        pygame.draw.polygon(self.screen, color, pts)
        pygame.draw.polygon(self.screen, (10, 20, 20), pts, 1)
        if role == SCOUT:
            pygame.draw.circle(self.screen, (255, 70, 110) if e.supply[0, i] > 0 else (230, 230, 230), c, max(1, size * .35))
        elif role == COMMANDER:
            pygame.draw.circle(self.screen, (255, 217, 107), c, max(1, size * .35))
        if self.zoom >= 3 or self.selected == i:
            pygame.draw.rect(self.screen, (20, 25, 25), (c[0] - 10, c[1] + size * 1.5, 20, 3))
            pygame.draw.rect(self.screen, (102, 230, 101), (c[0] - 10, c[1] + size * 1.5, int(20 * frac), 3))
        if self.selected == i:
            pygame.draw.circle(self.screen, (255, 242, 125), c, size * 1.4 + 4, 2)
        if self.control == i:                    # your tank: its vision cone
            for side in (-arena.VISION_SPAN, arena.VISION_SPAN):
                d = np.array([math.cos(a + side), math.sin(a + side)])
                pygame.draw.line(self.screen, (255, 242, 125), c, c + d * arena.VISION_RANGE * self.zoom, 1)

    # ---- first-person drawing -------------------------------------------------
    def draw_fps(self):
        """Raycast the walls and placed blocks at FPS_W x FPS_H, draw billboards for
        everything else (hidden behind nearer walls), then scale it to the window."""
        e, i = self.env, self.control
        pos, a = e.pos[0, i].numpy().astype(float), e.heading[0, i].item()
        rel = np.linspace(-FOV / 2, FOV / 2, FPS_W)
        dirs = np.stack((np.cos(a + rel), np.sin(a + rel)), -1)
        d = np.arange(STEP, FPS_RANGE, STEP)
        pts = pos + dirs[:, None] * d[None, :, None]                       # (W, S, 2)
        c = np.floor(pts).astype(int)
        out = ((c < 0) | (c >= self.grid)).any(-1)
        c = c.clip(0, self.grid - 1)
        wall = self.walls[c[..., 0], c[..., 1]] | out
        blocks = (e.blocks[0, :-1] > 0).numpy().reshape(e.BG, e.BG)
        bc = (c // arena.BLOCK).clip(0, e.BG - 1)
        block = blocks[bc[..., 0], bc[..., 1]]
        first = lambda hit: np.where(hit.any(1), hit.argmax(1), len(d) - 1)
        wi, bi = first(wall), first(block)
        cos = np.cos(rel)
        wd, bd = d[wi] * cos, d[bi] * cos                                  # perpendicular distances (no fisheye)
        img = self.backdrop.copy()
        rows = np.arange(FPS_H)[None, :]
        fog = lambda dist: np.clip(1 - dist / FPS_RANGE, .12, 1)[:, None]
        def column(dist, height, color, keep):
            top = FPS_H / 2 - FOCAL * (height - EYE) / dist
            bottom = FPS_H / 2 + FOCAL * EYE / dist
            mask = (rows >= top[:, None]) & (rows < bottom[:, None]) & keep[:, None]
            img[mask] = np.broadcast_to(color[:, None, :], img.shape)[mask]
        side = np.floor(pts[np.arange(FPS_W), wi, 0]) != np.floor(pts[np.arange(FPS_W), np.maximum(wi - 1, 0), 0])
        rock = np.array([96, 86, 72]) * np.where(side, 1., .75)[:, None] * fog(wd)
        column(wd, WALL_H, rock.astype(np.uint8), wi < len(d) - 1)
        has_block = block.any(1) & (bd < wd)
        owner = e.block_owner[0, :-1].numpy().reshape(e.BG, e.BG)[bc[np.arange(FPS_W), bi, 0], bc[np.arange(FPS_W), bi, 1]]
        sand = np.where((owner >= e.N)[:, None], [230, 160, 110], [150, 180, 200]) * fog(bd)
        column(bd, BLOCK_H, sand.astype(np.uint8), has_block)
        depth = np.where(has_block, bd, np.where(wi < len(d) - 1, wd, 1e9))
        # billboards: tanks, bases, hearts, bullets -- far to near, each clipped by the depth buffer
        things = []
        for j in range(e.A):
            if e.hp[0, j] > 0 and j != i:
                role = e.role[0, j].item()
                col = np.array(BLUE if j < e.N else RED)
                size = (6., 3.) if role == BASE else ((1.6, 1.1) if role == COMMANDER else (1.3, .9))
                things.append((e.pos[0, j].numpy(), size, col))
        for p in e.heart_pos[0][e.heart_alive[0]].numpy():
            things.append((p, (.5, .5), np.array([255, 70, 110])))
        for p in e.b_pos[0][e.b_alive[0]].numpy():
            things.append((p, (.3, .3), np.array([255, 240, 170])))
        drawn = []
        for p, (width, height), col in things:
            v = p - pos
            dist = math.hypot(*v)
            ang = wrap(math.atan2(v[1], v[0]) - a)
            if dist < .4 or dist > FPS_RANGE or abs(ang) > FOV / 2 + .2:
                continue
            drawn.append((dist * math.cos(ang), ang, width, height, col))
        for z, ang, width, height, col in sorted(drawn, key=lambda t: -t[0]):
            x = FPS_W / 2 + FOCAL * math.tan(ang)
            half = FOCAL * width / 2 / z
            x0, x1 = int(max(0, x - half)), int(min(FPS_W, x + half + 1))
            if x1 <= x0:
                continue
            top, bottom = int(FPS_H / 2 - FOCAL * (height - EYE) / z), int(FPS_H / 2 + FOCAL * EYE / z)
            top, bottom = max(0, top), min(FPS_H, max(bottom, top + 1))
            cols = np.arange(x0, x1)[depth[x0:x1] > z]
            img[cols, top:bottom] = (col * fog(np.array([z]))[0]).astype(np.uint8)
            if bottom - top > 6:                                          # a dark turret band on tanks
                img[cols, top:top + (bottom - top) // 4] = (col * .5).astype(np.uint8)
        w, h = self.screen.get_size()
        view = pygame.transform.scale(pygame.surfarray.make_surface(img), (w, h - HUD_H))
        self.screen.blit(view, (0, HUD_H))
        cx, cy = w // 2, HUD_H + (h - HUD_H) // 2
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):                  # crosshair
            pygame.draw.line(self.screen, (255, 242, 125), (cx + dx * 6, cy + dy * 6), (cx + dx * 16, cy + dy * 16), 2)
        self.minimap(pos, a)

    def minimap(self, pos, a):
        w, h = self.screen.get_size()
        m = 200
        self.screen.blit(pygame.transform.scale(self.overview, (m, m)), (w - m - 10, HUD_H + 10))
        s = m / self.grid
        e = self.env
        for j in range(e.A):
            if e.hp[0, j] > 0:
                p = e.pos[0, j].tolist()
                big = 3 if e.role[0, j] == BASE else 1
                pygame.draw.circle(self.screen, BLUE if j < e.N else RED, (w - m - 10 + p[0] * s, HUD_H + 10 + p[1] * s), big + 1)
        me = (w - m - 10 + float(pos[0]) * s, HUD_H + 10 + float(pos[1]) * s)
        pygame.draw.circle(self.screen, (255, 242, 125), me, 4, 1)
        pygame.draw.line(self.screen, (255, 242, 125), me, (me[0] + 12 * math.cos(a), me[1] + 12 * math.sin(a)), 2)

    # ---- HUD -----------------------------------------------------------------
    def hud(self):
        w, h = self.screen.get_size()
        pygame.draw.rect(self.screen, (8, 16, 18), (0, 0, w, HUD_H))
        e = self.env
        alive = (e.hp[0] > 0).view(2, e.N)
        it = self.iteration if self.iteration is not None else '- (waiting for tank_policy.pt from train.py)'
        label(self.screen, self.big, 'TANK ARENA', (14, 6))
        label(self.screen, self.font, f'policy iteration {it}   turn {e.t.item()}/{e.limit}   {self.speed}x   vs {self.opponent}', (14, 38))
        for team, x, col in ((0, w - 460, BLUE), (1, w - 230, RED)):
            label(self.screen, self.font, f'{"BLUE" if team == 0 else "RED"} tanks {alive[team, :e.slots].sum().item():2d}  '
                  f'bases {alive[team, e.slots:].sum().item()}', (x, 10), col)
        i = self.control if self.control is not None else self.selected
        if i is not None:
            what = 'hearts stored' if e.role[0, i] == BASE else 'hearts'
            who = 'YOU: ' if i == self.control else ''
            label(self.screen, self.font, f'{who}{"BLUE" if i < e.N else "RED"} {ROLE_NAMES[e.role[0, i]]}  hp {e.hp[0, i]:.1f}/'
                  f'{e.max_hp[0, i]:.0f}  {what} {int(e.supply[0, i])}', (w - 520, 38), BLUE if i < e.N else RED)
        if self.control is not None:
            ready = lambda cd: 'ready' if cd <= 0 else f'{int(cd)}'
            label(self.screen, self.font, f'gun {ready(e.gun_cd[0, i])}   special {ready(e.special_cd[0, i])}   '
                  f'block {ready(e.block_cd[0, i])}', (14, HUD_H + 6), (255, 242, 125))
            aim = 'mouse look' if self.fps else 'mouse aims'
            keys = f'W/S drive   {aim}   click/SPACE fire   right-click/E special   Q block   V view   C/ESC let go   P pause'
        else:
            keys = 'C drive a tank   click select   F follow   wheel zoom   drag pan   SPACE pause   N next game   +/- speed'
        if self.note and pygame.time.get_ticks() - self.note_at < 3000:
            r = self.big.render(self.note, True, (255, 150, 130))
            self.screen.blit(r, (w // 2 - r.get_width() // 2, HUD_H + 44))
        help_text = self.small.render(keys, True, (200, 210, 200))
        pygame.draw.rect(self.screen, (8, 16, 18), (0, h - 20, help_text.get_width() + 16, 20))
        self.screen.blit(help_text, (8, h - 17))
        if self.winner is not None:
            text = {0: 'BLUE WINS', 1: 'RED WINS'}.get(self.winner, 'DRAW')
            r = self.big.render(text, True, (255, 237, 158))
            self.screen.blit(r, (w // 2 - r.get_width() // 2, HUD_H + 10))

    # ---- playing a tank --------------------------------------------------------
    def take_control(self):
        """Drive the selected tank, or a random blue one."""
        e = self.env
        tank = (e.hp[0] > 0) & (e.role[0] != BASE)
        if self.selected is not None and tank[self.selected]:
            i = self.selected
        else:
            mine = tank.nonzero().flatten()
            mine = mine[mine < e.N]
            if not len(mine):
                return
            i = mine[torch.randint(len(mine), ())].item()
        self.control, self.selected, self.follow = i, i, False
        self.look = e.heading[0, i].item()
        self.zoom, self.speed = 6., 8                # close up and slow enough to play

    def release(self):
        self.control, self.fps = None, False
        self.grab(False)

    def grab(self, on):
        pygame.event.set_grab(on)
        pygame.mouse.set_visible(not on)
        pygame.mouse.get_rel()                       # drop motion accumulated before the switch

    def keys(self):
        """Your tank's action: throttle, steer, fire, special, order, place. The tank
        turns toward the mouse (top-down) or toward where you look (first person)."""
        e, i = self.env, self.control
        k, mouse = pygame.key.get_pressed(), pygame.mouse.get_pressed()
        heading = e.heading[0, i].item()
        if self.fps:
            target = self.look
        else:
            v = self.to_world(pygame.mouse.get_pos()) - e.pos[0, i].numpy()
            target = math.atan2(v[1], v[0])
        steer = float(np.clip(wrap(target - heading) / arena.TURN, -1, 1))
        fwd, back = k[pygame.K_w] or k[pygame.K_UP], k[pygame.K_s] or k[pygame.K_DOWN]
        return torch.tensor([1. if fwd else -.5 if back else 0., steer, float(mouse[0] or k[pygame.K_SPACE]),
                             float(mouse[2] or k[pygame.K_e]), 0., float(k[pygame.K_q])])

    # ---- loop --------------------------------------------------------------
    def step(self):
        with torch.no_grad():
            obs = self.env.observe()
            action = self.brain(obs, self.env.hp > 0)
            if self.opponent == 'raider':
                action[:, self.env.N:] = self.env.raider()[:, self.env.N:]
            if self.control is not None:
                action[0, self.control] = self.keys()
            _, _, done, winner = self.env.step(action)
        if done[0]:
            self.winner, self.ended_at = winner.item(), pygame.time.get_ticks()

    def pick(self, xy):
        e = self.env
        world = torch.tensor(self.to_world(xy), dtype=torch.float32)
        d = (e.pos[0] - world).norm(dim=-1).masked_fill(e.hp[0] <= 0, 1e9)
        i = d.argmin().item()
        self.selected = i if d[i] * self.zoom < 20 else None

    def run(self):
        drag, moved, reload_timer = None, False, 0
        while True:
            dt = self.clock.tick(60) / 1000
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(ev.size, pygame.RESIZABLE)
                if ev.type == pygame.MOUSEWHEEL and not self.fps:
                    self.zoom = float(np.clip(self.zoom * (1.15 if ev.y > 0 else 1 / 1.15), self.fit_zoom() * .8, 20))
                if self.control is None:
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        drag, moved = np.array(ev.pos), False
                    if ev.type == pygame.MOUSEMOTION and drag is not None:
                        delta = np.array(ev.pos) - drag
                        if np.abs(delta).sum() > 3:
                            self.camera -= delta / self.zoom
                            drag, moved, self.follow = np.array(ev.pos), True, False
                    if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                        if drag is not None and not moved:
                            self.pick(ev.pos)
                        drag = None
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_SPACE and self.control is None: self.playing = not self.playing
                    if ev.key == pygame.K_p: self.playing = not self.playing
                    if ev.key == pygame.K_c or (ev.key == pygame.K_ESCAPE and self.control is not None):
                        if self.control is None:
                            self.take_control()
                        else:
                            self.release(); self.zoom, self.speed = self.fit_zoom(), 16
                    if ev.key == pygame.K_v and self.control is not None:
                        self.fps = not self.fps
                        self.look = self.env.heading[0, self.control].item()
                        self.grab(self.fps)
                    if ev.key == pygame.K_f: self.follow = not self.follow
                    if ev.key == pygame.K_n: self.reload(); self.new_game()
                    if ev.key in (pygame.K_PLUS, pygame.K_EQUALS): self.speed = min(128, self.speed * 2)
                    if ev.key == pygame.K_MINUS: self.speed = max(1, self.speed // 2)
            if self.fps:
                self.look += pygame.mouse.get_rel()[0] * MOUSE_SENS
            reload_timer += dt
            if reload_timer > 5:
                reload_timer = 0
                self.reload()
            if self.winner is not None and pygame.time.get_ticks() - self.ended_at > 3000:
                self.reload(); self.new_game()
            if self.playing and self.winner is None:
                self.accum = min(self.accum + dt * self.speed, 8)       # don't try to catch up forever on a slow CPU
                while self.accum >= 1 and self.winner is None:
                    self.step(); self.accum -= 1
            if self.control is not None and self.env.hp[0, self.control] <= 0:
                self.note, self.note_at = 'YOUR TANK WAS DESTROYED  (C for another)', pygame.time.get_ticks()
                self.release(); self.zoom, self.speed = self.fit_zoom(), 16
            if self.control is not None:
                self.camera = self.env.pos[0, self.control].numpy().astype(float)
            elif self.follow and self.selected is not None and self.env.hp[0, self.selected] > 0:
                self.camera = self.env.pos[0, self.selected].numpy().astype(float)
            if self.fps:
                self.draw_fps()
            else:
                self.draw_world()
            self.hud()
            pygame.display.flip()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy', default='tank_policy.pt')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--opponent', choices=('neural', 'raider'), default='neural')
    p.add_argument('--grid', type=int, default=arena.GRID, help='board side length in tiles')
    p.add_argument('--tanks', type=int, default=sum(arena.START), help='tanks per side at the start')
    p.add_argument('--bases', type=int, default=arena.BASES, help='bases per side')
    a = p.parse_args()
    Viewer(a.policy, a.seed, a.opponent, a.grid, a.tanks, a.bases).run()
