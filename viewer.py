"""Watch the current policy play, live, while train.py keeps running -- or take over
one of the tanks (C) and fight it: top-down with the tank facing the mouse, or in a
first-person view on the ground (V switches). M overlays a team's shared map: what
its agents have written about each sector of the board.

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
from train import Policy, Player, load

# one game's tensors are tiny: spreading each op over all 20 cores costs more in
# thread hand-offs than it saves (a turn took ~40 ms on 20 threads, ~14 ms on 8)
torch.set_num_threads(min(8, os.cpu_count()))

BLUE, RED = (74, 205, 255), (255, 109, 78)
TEAM = (BLUE, RED)
HUD_H = 84
OVERVIEW = 4                  # the zoomed-out layer is the map at 1/4 resolution
# first-person renderer: a raycaster at low resolution, scaled up
FPS_W, FPS_H, FOV, FPS_RANGE, STEP = 320, 200, 1.2, 80., .15
FOCAL = FPS_W / 2 / math.tan(FOV / 2)
EYE, WALL_H, BLOCK_H = .6, 2.5, 1.       # eye height; walls tower, placed blocks are chest-high barricades
MOUSE_SENS = .004


def label(surface, font, s, xy, color=(230, 239, 226)):
    surface.blit(font.render(str(s), True, color), xy)


def rotate(pts, a):
    c, s = math.cos(a), math.sin(a)
    return [(x * c - y * s, x * s + y * c) for x, y in pts]


class Viewer:
    def __init__(self, path, seed, opponent, grid=arena.GRID, tanks=sum(arena.START), bases=arena.BASES):
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 960), pygame.RESIZABLE)
        pygame.display.set_caption('Multi-Agent Tanks')
        self.font = pygame.font.SysFont('dejavusansmono', 15)
        self.small = pygame.font.SysFont('dejavusansmono', 12)
        self.big = pygame.font.SysFont('dejavusans', 24, bold=True)
        self.clock = pygame.time.Clock()
        self.path, self.seed, self.opponent = path, seed, opponent
        self.grid, self.tanks, self.bases = grid, tanks, bases
        self.mtime, self.iteration, self.policy = 0, None, Policy()
        self.reload()
        self.new_game()
        self.zoom = self.fit_zoom()
        self.selected, self.follow = None, False
        self.playing, self.speed, self.accum = True, 16, 0.
        self.note, self.note_at = None, 0
        self.show_map = 0                                          # 0 off, 1 blue's map, 2 red's map
        rows = np.arange(FPS_H)[None, :, None]                     # sky and ground, darker toward the horizon
        sky = np.array([70, 110, 150]) * (.55 + .45 * (1 - rows / (FPS_H / 2)))
        ground = np.array([58, 82, 48]) * (.45 + .55 * (rows - FPS_H / 2) / (FPS_H / 2))
        self.backdrop = np.where(rows < FPS_H / 2, sky, ground).repeat(FPS_W, 0).astype(np.uint8)

    def reload(self):
        if os.path.exists(self.path) and os.path.getmtime(self.path) != self.mtime:
            try:
                self.policy, self.iteration = load(self.path)
                self.brain = Player(self.policy)
                self.mtime = os.path.getmtime(self.path)
            except Exception:
                pass                     # caught mid-write; try again next time

    def new_game(self):
        self.seed += 1
        self.env = Arena(batch=1, seed=self.seed, auto_reset=False, grid=self.grid, tanks=self.tanks, bases=self.bases)
        self.style = arena.STYLES[self.env.maps[0] % len(arena.STYLES)]
        self.walls = arena._map(self.env.maps[0], self.grid, self.bases)[0]
        rng = np.random.default_rng(self.seed)                       # a little grain so the ground isn't flat
        grain = rng.integers(-7, 8, (self.grid, self.grid, 1))
        rgb = np.where(self.walls[..., None], np.uint8([40, 36, 30]) + grain // 2, np.uint8([60, 84, 50]) + grain).astype(np.uint8)
        self.world = pygame.surfarray.make_surface(rgb)
        self.overview = pygame.transform.smoothscale(self.world, (self.grid // OVERVIEW, self.grid // OVERVIEW))
        self.brain = Player(self.policy)        # fresh memory every game
        self.winner, self.ended_at = None, None
        self.camera = np.array([self.grid / 2, self.grid / 2])
        self.effects = []                        # (world pos, started at ms, size) explosions
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

    def visible_world(self):
        w, h = self.screen.get_size()
        x0, y0 = self.to_world((0, HUD_H))
        x1, y1 = self.to_world((w, h))
        return max(0, int(x0)), max(0, int(y0)), min(self.grid, int(x1) + 1), min(self.grid, int(y1) + 1)

    # ---- the team map -------------------------------------------------------
    def map_image(self, team):
        """The team's shared map as an RGBA image, one pixel per sector: colour from
        the first three channels, opacity from how much has been written there."""
        M = self.brain.state[1][0, team].numpy() if self.brain.state is not None else None
        if M is None:
            return None
        S, C = M.shape[0], M.shape[-1]
        rgba = np.zeros((S, S, 4), np.uint8)
        rgba[..., :3] = ((M[..., :3] + 1) / 2 * 255).clip(0, 255)
        rgba[..., 3] = (np.linalg.norm(M, axis=-1) / math.sqrt(C) * 900).clip(0, 170)
        return rgba

    def draw_map_overlay(self):
        rgba = self.map_image(self.show_map - 1)
        if rgba is None:
            return
        S = rgba.shape[0]
        x0, y0, x1, y1 = self.visible_world()
        if x1 <= x0 or y1 <= y0:
            return
        cell = self.grid / S
        c0, r0 = int(x0 / cell), int(y0 / cell)
        c1, r1 = min(S, int(x1 / cell) + 1), min(S, int(y1 / cell) + 1)
        patch = pygame.image.frombuffer(np.ascontiguousarray(rgba[c0:c1, r0:r1].transpose(1, 0, 2)).tobytes(), (c1 - c0, r1 - r0), 'RGBA')
        size = (max(1, int((c1 - c0) * cell * self.zoom)), max(1, int((r1 - r0) * cell * self.zoom)))
        self.screen.blit(pygame.transform.scale(patch, size), self.to_screen((c0 * cell, r0 * cell)))
        if cell * self.zoom >= 12:                                  # sector grid lines when zoomed in
            for c in range(c0, c1 + 1):
                pygame.draw.line(self.screen, (0, 0, 0), self.to_screen((c * cell, r0 * cell)), self.to_screen((c * cell, r1 * cell)), 1)
            for r in range(r0, r1 + 1):
                pygame.draw.line(self.screen, (0, 0, 0), self.to_screen((c0 * cell, r * cell)), self.to_screen((c1 * cell, r * cell)), 1)

    # ---- top-down drawing ----------------------------------------------------
    def draw_world(self):
        self.screen.fill((14, 20, 18))
        x0, y0, x1, y1 = self.visible_world()
        if x1 > x0 and y1 > y0:
            src, f = (self.world, 1) if self.zoom >= 1 / OVERVIEW * 2 else (self.overview, OVERVIEW)
            rect = pygame.Rect(x0 // f, y0 // f, max(1, (x1 - x0) // f), max(1, (y1 - y0) // f)).clip(src.get_rect())
            size = (int(rect.w * f * self.zoom), int(rect.h * f * self.zoom))
            self.screen.blit(pygame.transform.scale(src.subsurface(rect), size), self.to_screen((rect.x * f, rect.y * f)))
        if self.show_map:
            self.draw_map_overlay()
        e = self.env
        # placed blocks: bevelled sand blocks in their team's tint, darkening as they take hits
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
                    pygame.draw.line(self.screen, [min(255, c + 40) for c in col], (x, y), (x + size - 1, y), 1)
                    pygame.draw.line(self.screen, [max(0, c - 50) for c in col], (x, y + size - 1), (x + size - 1, y + size - 1), 1)
        hearts = e.heart_pos[0][e.heart_alive[0]].numpy()
        for p in hearts[self.on_screen(hearts)]:
            self.draw_heart(self.to_screen(p), self.zoom)
        bp, bv, alive = e.b_pos[0].numpy(), e.b_vel[0].numpy(), e.b_alive[0].numpy()
        for i, k in zip(*np.nonzero(alive & self.on_screen(bp))):
            col = (255, 217, 110) if e.b_pierce[0, i, k] else (255, 240, 170)
            pygame.draw.line(self.screen, tuple(c // 2 for c in col), self.to_screen(bp[i, k] - 3 * bv[i, k]), self.to_screen(bp[i, k] - bv[i, k]), 1)
            pygame.draw.line(self.screen, col, self.to_screen(bp[i, k] - bv[i, k]), self.to_screen(bp[i, k]), 2)
        for i in range(e.A):
            self.draw_agent(i)
        now = pygame.time.get_ticks()
        self.effects = [f for f in self.effects if now - f[1] < 600]
        for p, t0, size in self.effects:
            k = (now - t0) / 600
            r = max(1, int((0.6 + 2.5 * k) * size * self.zoom))
            pygame.draw.circle(self.screen, (255, int(200 * (1 - k)), 60), self.to_screen(p), r, max(1, int(3 * (1 - k))))

    def draw_heart(self, c, zoom):
        if zoom < 1.5:
            self.screen.set_at(c, (230, 70, 110)) if zoom < .7 else pygame.draw.circle(self.screen, (255, 70, 110), c, 2)
            return
        r = max(2, int(.45 * zoom))
        pygame.draw.circle(self.screen, (255, 70, 110), (c[0] - r // 2 - 1, c[1] - r // 2), r)
        pygame.draw.circle(self.screen, (255, 70, 110), (c[0] + r // 2 + 1, c[1] - r // 2), r)
        pygame.draw.polygon(self.screen, (255, 70, 110), [(c[0] - 2 * r, c[1] - r // 4), (c[0] + 2 * r, c[1] - r // 4), (c[0], c[1] + int(1.8 * r))])

    def draw_agent(self, i):
        e = self.env
        if e.hp[0, i] <= 0:
            return
        p = e.pos[0, i].numpy()
        if not self.on_screen(p, 60):
            return
        c = np.array(self.to_screen(p))
        role = e.role[0, i].item()
        color = TEAM[i >= e.N]
        dark = tuple(int(v * .45) for v in color)
        frac = (e.hp[0, i] / e.max_hp[0, i]).item()
        if role == BASE:
            size = max(14, arena.BASE_CLEAR * self.zoom)
            pygame.draw.rect(self.screen, dark, (c[0] - size / 2, c[1] - size / 2, size, size))
            pygame.draw.rect(self.screen, color, (c[0] - size / 2 + 3, c[1] - size / 2 + 3, size - 6, size - 6))
            pygame.draw.circle(self.screen, dark, c, max(3, size * .22))
            a = e.heading[0, i].item()                             # the turret: shows where a wall order would go
            pygame.draw.line(self.screen, dark, c, c + np.array([math.cos(a), math.sin(a)]) * size * .5, max(2, int(size * .08)))
            bar = max(20, size)
            pygame.draw.rect(self.screen, (20, 25, 25), (c[0] - bar / 2, c[1] - size / 2 - 7, bar, 4))
            pygame.draw.rect(self.screen, (102, 230, 101), (c[0] - bar / 2, c[1] - size / 2 - 7, int(bar * frac), 4))
            label(self.screen, self.small, f'♥{int(e.supply[0, i])}', (c[0] + size / 2 + 3, c[1] - 8), (255, 150, 170))
            return
        size = max(3.5, (1.3 if role == COMMANDER else .85 if role == SCOUT else 1.) * self.zoom)
        a = e.heading[0, i].item()
        if size >= 6:                                              # a proper tank: treads, hull, turret, barrel
            treads = [(-1.1, -.9), (1.1, -.9), (1.1, .9), (-1.1, .9)]
            hull = [(-.9, -.55), (.9, -.55), (.9, .55), (-.9, .55)]
            pygame.draw.polygon(self.screen, dark, [c + np.array(q) * size for q in rotate(treads, a)])
            pygame.draw.polygon(self.screen, color, [c + np.array(q) * size for q in rotate(hull, a)])
            fw = np.array([math.cos(a), math.sin(a)])
            pygame.draw.line(self.screen, dark, c, c + fw * size * 1.7, max(2, int(size * .3)))
            pygame.draw.circle(self.screen, (255, 217, 107) if role == COMMANDER else dark, c, size * .45)
            if role == SCOUT and e.supply[0, i] > 0:
                pygame.draw.circle(self.screen, (255, 70, 110), c, size * .3)
        else:                                                      # zoomed out: an arrowhead
            fw = np.array([math.cos(a), math.sin(a)]); sd = np.array([-fw[1], fw[0]])
            pygame.draw.polygon(self.screen, color, [c + fw * size * 1.3, c - fw * size + sd * size * .8, c - fw * size - sd * size * .8])
        if self.zoom >= 3 or self.selected == i:
            pygame.draw.rect(self.screen, (20, 25, 25), (c[0] - 10, c[1] + size * 1.5, 20, 3))
            pygame.draw.rect(self.screen, (102, 230, 101), (c[0] - 10, c[1] + size * 1.5, int(20 * frac), 3))
        if self.selected == i:
            pygame.draw.circle(self.screen, (255, 242, 125), c, size * 1.6 + 4, 2)
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
        cols = np.arange(FPS_W)
        hit = pts[cols, wi]
        side = np.floor(hit[:, 0]) != np.floor(pts[cols, np.maximum(wi - 1, 0), 0])
        stripe = .85 + .15 * ((np.floor(hit[:, 0] * 2) + np.floor(hit[:, 1] * 2)) % 2)   # a little texture on the rock
        rock = np.array([96, 86, 72]) * (np.where(side, 1., .75) * stripe)[:, None] * fog(wd)
        column(wd, WALL_H, rock.astype(np.uint8), wi < len(d) - 1)
        has_block = block.any(1) & (bd < wd)
        owner = e.block_owner[0, :-1].numpy().reshape(e.BG, e.BG)[bc[cols, bi, 0], bc[cols, bi, 1]]
        sand = np.where((owner >= e.N)[:, None], [230, 160, 110], [150, 180, 200]) * fog(bd)
        column(bd, BLOCK_H, sand.astype(np.uint8), has_block)
        depth = np.where(has_block, bd, np.where(wi < len(d) - 1, wd, 1e9))
        # billboards: tanks, bases, hearts, bullets -- far to near, each clipped by the depth buffer
        things = []
        for j in range(e.A):
            if e.hp[0, j] > 0 and j != i:
                role = e.role[0, j].item()
                size = (6., 3.) if role == BASE else ((1.6, 1.1) if role == COMMANDER else (1.3, .9))
                things.append((e.pos[0, j].numpy(), size, np.array(TEAM[j >= e.N])))
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
            keep = np.arange(x0, x1)[depth[x0:x1] > z]
            img[keep, top:bottom] = (col * fog(np.array([z]))[0]).astype(np.uint8)
            if bottom - top > 6:                                          # a dark turret band on tanks
                img[keep, top:top + (bottom - top) // 4] = (col * .5).astype(np.uint8)
        w, h = self.screen.get_size()
        view = pygame.transform.scale(pygame.surfarray.make_surface(img), (w, h - HUD_H))
        self.screen.blit(view, (0, HUD_H))
        cx, cy = w // 2, HUD_H + (h - HUD_H) // 2
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):                  # crosshair
            pygame.draw.line(self.screen, (255, 242, 125), (cx + dx * 6, cy + dy * 6), (cx + dx * 16, cy + dy * 16), 2)
        self.minimap(pos, a)

    def minimap(self, pos, a):
        w, h = self.screen.get_size()
        m, x, y = 200, w - 210, HUD_H + 10
        self.screen.blit(pygame.transform.scale(self.overview, (m, m)), (x, y))
        if self.show_map:
            rgba = self.map_image(self.show_map - 1)
            if rgba is not None:
                patch = pygame.image.frombuffer(np.ascontiguousarray(rgba.transpose(1, 0, 2)).tobytes(), rgba.shape[:2], 'RGBA')
                self.screen.blit(pygame.transform.scale(patch, (m, m)), (x, y))
        s = m / self.grid
        e = self.env
        for j in range(e.A):
            if e.hp[0, j] > 0:
                p = e.pos[0, j].tolist()
                pygame.draw.circle(self.screen, TEAM[j >= e.N], (x + p[0] * s, y + p[1] * s), 4 if e.role[0, j] == BASE else 2)
        me = (x + float(pos[0]) * s, y + float(pos[1]) * s)
        pygame.draw.circle(self.screen, (255, 242, 125), me, 4, 1)
        pygame.draw.line(self.screen, (255, 242, 125), me, (me[0] + 12 * math.cos(a), me[1] + 12 * math.sin(a)), 2)

    # ---- HUD -----------------------------------------------------------------
    def hud(self):
        w, h = self.screen.get_size()
        pygame.draw.rect(self.screen, (8, 16, 18), (0, 0, w, HUD_H))
        e = self.env
        alive = (e.hp[0] > 0).view(2, e.N)
        it = self.iteration if self.iteration is not None else '- (waiting for tank_policy.pt from train.py)'
        label(self.screen, self.big, 'MULTI-AGENT TANKS', (14, 6))
        label(self.screen, self.font, f'policy iteration {it}   turn {e.t.item()}/{e.limit}   {self.speed}x   '
              f'{self.grid}x{self.grid} {self.style}   vs {self.opponent}', (14, 38))
        for team, x, col in ((0, w - 460, BLUE), (1, w - 230, RED)):
            label(self.screen, self.font, f'{"BLUE" if team == 0 else "RED"} tanks {alive[team, :e.slots].sum().item():2d}  '
                  f'bases {alive[team, e.slots:].sum().item()}', (x, 10), col)
        i = self.control if self.control is not None else self.selected
        if i is not None:
            what = 'hearts stored' if e.role[0, i] == BASE else 'hearts'
            who = 'YOU: ' if i == self.control else ''
            label(self.screen, self.font, f'{who}{"BLUE" if i < e.N else "RED"} {ROLE_NAMES[e.role[0, i]]}  hp {e.hp[0, i]:.1f}/'
                  f'{e.max_hp[0, i]:.0f}  {what} {int(e.supply[0, i])}', (14, 60), TEAM[i >= e.N])
        if self.show_map:
            label(self.screen, self.font, f'{"BLUE" if self.show_map == 1 else "RED"} TEAM MAP', (w // 2 - 60, HUD_H + 6), TEAM[self.show_map - 1])
        if self.control is not None:
            ready = lambda cd: 'ready' if cd <= 0 else f'{int(cd)}'
            label(self.screen, self.font, f'gun {ready(e.gun_cd[0, i])}   special {ready(e.special_cd[0, i])}   '
                  f'block {ready(e.block_cd[0, i])}', (14, HUD_H + 6), (255, 242, 125))
            aim = 'mouse look' if self.fps else 'mouse aims'
            keys = f'W/S drive   {aim}   click/SPACE fire   right-click/E special   Q block   V view   M team map   C/ESC let go   P pause'
        else:
            keys = 'C drive a tank   M team map   click select   F follow   wheel zoom   drag pan   SPACE pause   N next game   +/- speed'
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
        e = self.env
        with torch.no_grad():
            obs = e.observe()
            action = self.brain(obs, e.hp > 0)
            if self.opponent == 'raider':
                action[:, e.N:] = e.raider()[:, e.N:]
            if self.control is not None:
                action[0, self.control] = self.keys()
            before = e.hp[0] > 0
            _, _, done, winner = e.step(action)
        now = pygame.time.get_ticks()
        for j in torch.nonzero(before & (e.hp[0] <= 0)).flatten().tolist():
            self.effects.append((e.pos[0, j].numpy().copy(), now, 5. if e.role[0, j] == BASE else 1.5))
        if done[0]:
            self.winner, self.ended_at = winner.item(), now

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
                    if ev.key == pygame.K_m: self.show_map = (self.show_map + 1) % 3
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
