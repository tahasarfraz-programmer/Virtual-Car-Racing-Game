#!/usr/bin/env python3
"""
Virtual Steering Racer
----------------------
First-person cockpit racing game you steer with BOTH HANDS in front of your
webcam (hold them like you're gripping a wheel and tilt left / right).

Layout copies the demo video: a "Virtual Steering Wheel" webcam panel on the
left (HANDS DETECTED, FPS, LEFT/RIGHT bar, angle in degrees, wheel dial) and
the racing game on the right.

Controls
  Hands  : tilt both hands like a wheel  -> steer
  A / D or Left / Right : keyboard steering (also the fallback with no camera)
  W / Up : extra gas        S / Down / Space : brake
  R      : restart after a crash-out      ESC : quit

Run:  python racer.py            (camera 0)
      python racer.py --camera 1
      python racer.py --no-camera
"""
import argparse
import math
import os
import random
import sys
import threading
import time

import numpy as np
import pygame

try:
    import cv2
except ImportError:  # keyboard mode still works
    cv2 = None
try:
    import mediapipe as mp
except ImportError:
    mp = None

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
PANEL_W = 320                 # webcam panel (left)
GW, GH = 960, 640             # game viewport (right)
WIN_W, WIN_H = PANEL_W + GW, GH
HORIZON = 250
ROWS = GH - HORIZON
ANGLE_MAX = 45.0              # hand tilt (deg) that equals full lock
DEADZONE = 3.0
Z0 = 12.0                     # camera depth constant


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


def shade(c, k):
    return tuple(int(clamp(v * k, 0, 255)) for v in c)


# --------------------------------------------------------------------------
# Hand tracking thread
# --------------------------------------------------------------------------
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
              (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def make_detector():
    """Return fn(rgb) -> list of hands, each a list of 21 (x, y) in 0..1.
    Works with old mediapipe (mp.solutions) and new (Tasks API)."""
    if hasattr(mp, "solutions"):
        det = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0,
                                       min_detection_confidence=0.6,
                                       min_tracking_confidence=0.5)

        def run(rgb):
            res = det.process(rgb)
            return [[(p.x, p.y) for p in h.landmark]
                    for h in (res.multi_hand_landmarks or [])]
        return run

    # new Tasks API: needs the small model file, downloaded once
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        print("Downloading hand model (one time, ~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.6, min_tracking_confidence=0.5)
    lmk = vision.HandLandmarker.create_from_options(opts)
    t0 = time.time()

    def run(rgb):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = lmk.detect_for_video(img, int((time.time() - t0) * 1000))
        return [[(p.x, p.y) for p in h] for h in res.hand_landmarks]
    return run


class HandSteer(threading.Thread):
    """Reads the webcam, finds two hands and turns their tilt into an angle."""

    def __init__(self, cam_index=0):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.running = True
        self.frame = None          # RGB uint8 (240x320)
        self.angle = 0.0           # degrees, + = clockwise = steer right
        self.hands = 0
        self.fps = 0.0
        self.ok = False
        self.error = None

    def run(self):
        if cv2 is None or mp is None:
            self.error = "opencv / mediapipe not installed"
            return
        try:
            detect = make_detector()
        except Exception as e:
            self.error = f"hand model failed: {e}"[:60]
            print(self.error)
            return
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            self.error = f"cannot open camera {self.cam_index}"
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ok = True
        last, smooth = time.time(), 0.0
        while self.running:
            good, img = cap.read()
            if not good:
                time.sleep(0.01)
                continue
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            hands = detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pts = []
            for lm in hands:
                for a, b in HAND_LINKS:
                    cv2.line(img, (int(lm[a][0] * w), int(lm[a][1] * h)),
                             (int(lm[b][0] * w), int(lm[b][1] * h)), (255, 255, 255), 2)
                for x, y in lm:
                    cv2.circle(img, (int(x * w), int(y * h)), 4, (80, 80, 255), -1)
                ids = (0, 5, 9, 13, 17)          # palm centre
                pts.append((sum(lm[i][0] for i in ids) / 5 * w,
                            sum(lm[i][1] for i in ids) / 5 * h))
            if len(pts) >= 2:
                pts.sort(key=lambda p: p[0])
                (lx, ly), (rx, ry) = pts[0], pts[-1]
                ang = math.degrees(math.atan2(ry - ly, rx - lx))
                smooth = lerp(smooth, ang, 0.45)
                self.hands = 2
                cv2.line(img, (int(lx), int(ly)), (int(rx), int(ry)), (80, 255, 120), 3)
            else:
                self.hands = len(pts)
                smooth = lerp(smooth, 0.0, 0.2)
            self.angle = smooth
            now = time.time()
            self.fps = lerp(self.fps, 1.0 / max(now - last, 1e-3), 0.1)
            last = now
            self.frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (PANEL_W, 240))
        cap.release()


# --------------------------------------------------------------------------
# Track + world
# --------------------------------------------------------------------------
def track_curve(w):
    """Road curvature at world position w (about -1.2 .. 1.2)."""
    return (0.8 * math.sin(w * 0.0021) + 0.55 * math.sin(w * 0.0053 + 1.3)
            + 0.25 * math.sin(w * 0.011 + 0.4))


CAR_COLORS = [(210, 40, 40), (40, 90, 210), (235, 190, 30), (240, 240, 240),
              (50, 170, 90), (150, 60, 190), (240, 120, 30)]


class Traffic:
    def __init__(self, d, lane, speed, color):
        self.d, self.lane, self.speed, self.color = d, lane, speed, color
        self.passed = False


class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.dist = 0.0
        self.speed = 18.0
        self.player_x = 0.0
        self.steer = 0.0
        self.heading = 0.0
        self.score = 0
        self.time = 0.0
        self.cars = []
        self.crash_t = 0.0
        self.shake = 0.0
        self.crashes = 0
        self.next_spawn = 0.0
        self.msg = ""
        self.msg_t = 0.0

    # ---- simulation ------------------------------------------------------
    def update(self, dt, steer_in, gas, brake):
        self.steer += (steer_in - self.steer) * min(1.0, dt * 9)
        top = 62.0
        if brake:
            self.speed -= 45 * dt
        else:
            self.speed += (14 + (10 if gas else 0)) * dt * (1 - self.speed / (top + 8))
        off_road = abs(self.player_x) > 1.05
        if off_road:
            self.speed -= 28 * dt
            self.shake = max(self.shake, 3.0)
        self.speed = clamp(self.speed, 6.0 if not brake else 0.0, top)

        curve = track_curve(self.dist + Z0 + 20)
        # steering moves the car sideways, curves push it outwards
        self.player_x += self.steer * dt * (0.5 + self.speed * 0.045)
        self.player_x -= curve * dt * self.speed * 0.035
        self.player_x = clamp(self.player_x, -1.9, 1.9)
        self.heading += curve * self.speed * dt * 0.0026

        self.dist += self.speed * dt
        self.time += dt
        self.score += int(self.speed * dt * 1.2)
        self.shake = max(0.0, self.shake - dt * 14)
        self.crash_t = max(0.0, self.crash_t - dt)
        self.msg_t = max(0.0, self.msg_t - dt)

        # traffic
        self.next_spawn -= dt
        if self.next_spawn <= 0:
            self.next_spawn = random.uniform(0.7, 1.6)
            lane = random.choice([-0.62, 0.0, 0.62]) + random.uniform(-0.08, 0.08)
            if all(abs(c.d - 380) > 40 or abs(c.lane - lane) > 0.3 for c in self.cars):
                self.cars.append(Traffic(380, lane, random.uniform(14, 34),
                                         random.choice(CAR_COLORS)))
        for c in self.cars:
            c.d -= (self.speed - c.speed) * dt
            if not c.passed and c.d < 0:
                c.passed = True
                self.score += 50
                self.say("+50 overtake")
            if 0 <= c.d < 5 and abs(c.lane - self.player_x) < 0.30 and not c.passed:
                c.passed = True
                self.crash(c)
        self.cars = [c for c in self.cars if -10 < c.d < 420]

    def crash(self, car):
        self.crash_t = 0.5
        self.shake = 14.0
        self.crashes += 1
        self.speed = max(6.0, self.speed * 0.35)
        self.score = max(0, self.score - 100)
        self.say("CRASH!  -100")

    def say(self, text):
        self.msg, self.msg_t = text, 1.2


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.view = pygame.Surface((GW, GH))
        self.f_big = pygame.font.Font(None, 56)
        self.f_mid = pygame.font.Font(None, 32)
        self.f_sm = pygame.font.Font(None, 22)
        self.f_xs = pygame.font.Font(None, 18)
        rnd = random.Random(7)
        self.stars = [(rnd.randrange(GW), rnd.randrange(HORIZON - 10), rnd.random())
                      for _ in range(90)]
        self.clouds = [(rnd.randrange(GW * 2), rnd.randrange(40, 150), rnd.randrange(60, 130))
                       for _ in range(6)]
        self.peaks = [(i * 34, rnd.randrange(14, 70)) for i in range(0, 60)]
        self.offs = [0.0] * ROWS
        self.zs = [0.0] * ROWS

    # ---- world -----------------------------------------------------------
    def world(self, g):
        s = self.view
        night = 0.5 - 0.5 * math.cos(g.dist * 0.0007)         # 0 day .. 1 night
        k = 1 - 0.62 * night
        top = mix((60, 130, 225), (6, 8, 28), night)
        bot = mix((195, 228, 250), (60, 45, 85), night)
        for y in range(0, HORIZON, 5):
            pygame.draw.rect(s, mix(top, bot, y / HORIZON), (0, y, GW, 6))
        if night > 0.45:
            for x, y, b in self.stars:
                v = int(255 * clamp((night - 0.45) * 2, 0, 1) * (0.4 + 0.6 * b))
                s.set_at((x, y), (v, v, v))
        sx = int(GW * 0.72 - g.heading * 380) % (GW * 2) - GW // 2
        if night < 0.8:
            pygame.draw.circle(s, mix((255, 245, 180), (255, 120, 60), night), (sx, int(150 + night * 90)), 34)
        else:
            pygame.draw.circle(s, (225, 225, 240), (sx, 90), 22)
        cc = mix((255, 255, 255), (90, 80, 120), night)
        for x, y, w in self.clouds:
            cx = (x - g.heading * 220) % (GW * 2) - GW // 2
            pygame.draw.ellipse(s, cc, (cx, y, w, w // 3))
            pygame.draw.ellipse(s, cc, (cx + w // 4, y - w // 8, w // 2, w // 3))
        # distant mountains (parallax with heading)
        shift = g.heading * 520
        pts = [(0, HORIZON)]
        for i, h in self.peaks:
            pts.append(((i - shift) % (len(self.peaks) * 34) - 40, HORIZON - h))
        pts = sorted(pts, key=lambda p: p[0])
        pts.append((GW + 60, HORIZON))
        pygame.draw.polygon(s, mix((190, 120, 70), (30, 25, 45), night), pts)

        # road offsets (accumulated curvature, near -> far)
        dx = x = 0.0
        prev = None
        for i in range(ROWS - 1, -1, -1):
            p = (i + 1) / ROWS
            z = Z0 / (p + 0.03)
            self.zs[i] = z
            if prev is not None:
                dz = z - prev
                dx += track_curve(g.dist + z) * dz * 0.0022
                x += dx * dz
            self.offs[i] = x
            prev = z

        grass_a = shade((222, 168, 92), k)
        grass_b = shade((208, 152, 78), k)
        road_a = shade((82, 82, 88), k)
        road_b = shade((76, 76, 82), k)
        rum_a = shade((230, 40, 40), k)
        rum_b = shade((240, 240, 240), k)
        line = shade((235, 235, 235), k)
        for i in range(0, ROWS, 2):
            y = HORIZON + i
            p = (i + 1) / ROWS
            z = self.zs[i]
            w = g.dist + z
            half = GW * (0.015 + 0.62 * p)
            mid = GW / 2 + self.offs[i] - g.player_x * half
            band = int(w // 14) % 2
            pygame.draw.rect(s, grass_a if band else grass_b, (0, y, GW, 2))
            rw = half * 0.13
            pygame.draw.rect(s, rum_a if int(w // 7) % 2 else rum_b,
                             (mid - half - rw, y, 2 * (half + rw), 2))
            pygame.draw.rect(s, road_a if band else road_b, (mid - half, y, 2 * half, 2))
            if int(w // 10) % 2:
                lw = max(1, half * 0.022)
                for lx in (-0.34, 0.34):
                    pygame.draw.rect(s, line, (mid + lx * half - lw, y, 2 * lw, 2))

        # sprites: roadside posts + traffic, far -> near
        sprites = []
        first = int(g.dist // 30 + 1) * 30
        for w in range(first, int(g.dist) + 400, 30):
            sprites.append((w - g.dist, "post", 0, None))
        for c in g.cars:
            if c.d >= 0:
                sprites.append((c.d, "car", c.lane, c.color))
        sprites.sort(key=lambda t: -t[0])
        for d, kind, lane, col in sprites:
            z = Z0 + d
            p = Z0 / z - 0.03
            if not 0.01 < p <= 1.0:
                continue
            i = min(ROWS - 1, int(p * ROWS))
            y = HORIZON + i
            half = GW * (0.015 + 0.62 * p)
            mid = GW / 2 + self.offs[i] - g.player_x * half
            if kind == "post":
                for side in (-1, 1):
                    px = mid + side * half * 1.28
                    h = max(2, half * 0.55)
                    pygame.draw.rect(s, shade((70, 70, 75), k), (px - max(1, half * 0.02), y - h, max(2, half * 0.04), h))
                    pygame.draw.rect(s, (255, 200, 60) if night > 0.4 else shade((200, 60, 50), k),
                                     (px - half * 0.05, y - h, half * 0.1, max(2, half * 0.06)))
            else:
                self.draw_car(s, mid + lane * half, y, half * 0.34, col, k, night)

    def draw_car(self, s, x, y, w, col, k, night):
        w = max(3, w)
        h = w * 0.62
        pygame.draw.ellipse(s, (30, 30, 30), (x - w * 0.55, y - h * 0.12, w * 1.1, h * 0.28))
        pygame.draw.rect(s, shade(col, k), (x - w / 2, y - h, w, h * 0.78), border_radius=int(w * 0.1))
        pygame.draw.rect(s, shade((25, 30, 40), k), (x - w * 0.38, y - h * 0.98, w * 0.76, h * 0.34))
        lc = (255, 40, 40) if night < 0.4 else (255, 90, 90)
        for sx in (-1, 1):
            pygame.draw.rect(s, lc, (x + sx * w * 0.36 - w * 0.09, y - h * 0.42, w * 0.18, h * 0.16))
            pygame.draw.rect(s, (15, 15, 15), (x + sx * w * 0.5 - w * 0.05, y - h * 0.2, w * 0.1, h * 0.2))

    # ---- cockpit ---------------------------------------------------------
    def cockpit(self, g, wheel_deg, night):
        s = self.view
        W, H = GW, GH
        k = 1 - 0.45 * night
        # roof + pillars
        pygame.draw.polygon(s, shade((22, 22, 26), 1), [(0, 0), (W, 0), (W, 36), (0, 36)])
        pygame.draw.polygon(s, (18, 18, 22), [(0, 0), (150, 0), (70, 330), (0, 420)])
        pygame.draw.polygon(s, (18, 18, 22), [(W, 0), (W - 150, 0), (W - 70, 330), (W, 420)])
        pygame.draw.polygon(s, (30, 30, 36), [(0, 420), (70, 330), (150, 360), (0, 520)])
        pygame.draw.polygon(s, (30, 30, 36), [(W, 420), (W - 70, 330), (W - 150, 360), (W, 520)])
        # rear-view mirror
        mx, my, mw, mh = W // 2 - 90, 40, 180, 46
        pygame.draw.rect(s, (14, 14, 16), (mx - 6, my - 6, mw + 12, mh + 12), border_radius=10)
        mir = pygame.transform.smoothscale(
            self.view.subsurface((W // 2 - 160, HORIZON - 60, 320, 120)).copy(), (mw, mh))
        s.blit(mir, (mx, my))
        # dashboard
        pygame.draw.polygon(s, shade((38, 38, 44), k), [(0, 520), (W * 0.28, 470), (W * 0.72, 470), (W, 520), (W, H), (0, H)])
        pygame.draw.polygon(s, shade((26, 26, 30), k), [(0, 560), (W, 560), (W, H), (0, H)])
        pygame.draw.line(s, shade((90, 90, 100), k), (W * 0.28, 470), (W * 0.72, 470), 3)
        # speed readout + gauge
        kmh = int(g.speed * 3.6)
        gx, gy = W // 2 - 190, 540
        pygame.draw.circle(s, (12, 14, 16), (gx, gy), 46)
        pygame.draw.circle(s, (70, 200, 255), (gx, gy), 46, 2)
        a = math.radians(-210 + 240 * clamp(g.speed / 70, 0, 1))
        pygame.draw.line(s, (255, 90, 60), (gx, gy), (gx + 36 * math.cos(a), gy + 36 * math.sin(a)), 3)
        t = self.f_mid.render(f"{kmh}", True, (230, 240, 255))
        s.blit(t, t.get_rect(center=(gx, gy + 22)))
        # steering wheel
        cx, cy, r = W // 2, 600, 118
        ang = math.radians(wheel_deg)
        pygame.draw.circle(s, (12, 12, 14), (cx, cy), r + 8, 20)
        pygame.draw.circle(s, (46, 46, 52), (cx, cy), r + 8, 12)
        for off in (90, 210, 330):
            a2 = ang + math.radians(off)
            pygame.draw.line(s, (20, 20, 24), (cx, cy), (cx + r * math.cos(a2), cy + r * math.sin(a2)), 14)
        pygame.draw.circle(s, (28, 28, 34), (cx, cy), 34)
        pygame.draw.circle(s, (120, 120, 135), (cx, cy), 34, 2)
        # hands glued to the rim at 9 and 3 o'clock
        for side in (-1, 1):
            a3 = ang + (math.pi if side < 0 else 0) + side * math.radians(-12)
            hx, hy = cx + (r + 6) * math.cos(a3), cy + (r + 6) * math.sin(a3)
            pygame.draw.line(s, (190, 140, 110), (hx + side * 90, H + 10), (hx, hy), 30)
            pygame.draw.circle(s, (205, 155, 120), (int(hx), int(hy)), 21)
            pygame.draw.circle(s, (170, 120, 92), (int(hx), int(hy)), 21, 3)

    # ---- HUD -------------------------------------------------------------
    def hud(self, g):
        s = self.view
        box = pygame.Surface((190, 96), pygame.SRCALPHA)
        pygame.draw.rect(box, (20, 20, 40, 190), (0, 0, 190, 96), border_radius=12)
        s.blit(box, (16, 14))
        rows = [((255, 210, 60), f"{g.score}"), ((120, 220, 255), f"{g.dist / 1000:05.2f} KM"),
                ((150, 255, 150), f"{g.time:4.1f} s")]
        for n, (col, txt) in enumerate(rows):
            pygame.draw.circle(s, col, (38, 38 + n * 28), 8)
            s.blit(self.f_mid.render(txt, True, (240, 240, 250)), (58, 28 + n * 28))
        if g.msg_t > 0:
            t = self.f_big.render(g.msg, True, (255, 90, 80) if "CRASH" in g.msg else (120, 255, 150))
            s.blit(t, t.get_rect(center=(GW // 2, 150)))

    # ---- left panel ------------------------------------------------------
    def panel(self, tracker, steer_val, wheel_deg, mode, g):
        scr = self.screen
        pygame.draw.rect(scr, (24, 24, 30), (0, 0, PANEL_W, WIN_H))
        pygame.draw.rect(scr, (52, 52, 60), (0, 0, PANEL_W, 28))
        t = self.f_sm.render("Virtual Steering Wheel", True, (200, 200, 210))
        scr.blit(t, t.get_rect(center=(PANEL_W // 2, 14)))
        cam = pygame.Rect(0, 28, PANEL_W, 240)
        if tracker and tracker.frame is not None:
            scr.blit(pygame.image.frombuffer(tracker.frame.tobytes(), (PANEL_W, 240), "RGB"), cam)
        else:
            pygame.draw.rect(scr, (14, 14, 18), cam)
            msg = (tracker.error if tracker and tracker.error else "no camera - keyboard mode")
            for n, line in enumerate(("NO CAMERA", msg[:34])):
                t = self.f_sm.render(line, True, (230, 120, 100))
                scr.blit(t, t.get_rect(center=(PANEL_W // 2, 130 + n * 24)))
        # overlays
        det = tracker is not None and tracker.hands >= 2
        label = "HANDS DETECTED" if det else "SHOW BOTH HANDS"
        scr.blit(self.f_xs.render(label, True, (90, 255, 120) if det else (255, 90, 80)), (8, 34))
        if tracker and tracker.ok:
            scr.blit(self.f_xs.render(f"FPS: {int(tracker.fps)}", True, (255, 200, 80)), (PANEL_W - 70, 34))
        by = 28 + 240 - 58
        scr.blit(self.f_xs.render("<- LEFT", True, (255, 150, 90)), (10, by - 18))
        t = self.f_xs.render("RIGHT ->", True, (130, 255, 130))
        scr.blit(t, (PANEL_W - 16 - t.get_width() - 70, by - 18))
        bar = pygame.Rect(10, by, PANEL_W - 100, 12)
        pygame.draw.rect(scr, (50, 50, 56), bar)
        mid = bar.centerx
        fill = int(steer_val * bar.w / 2)
        pygame.draw.rect(scr, (120, 255, 70) if fill >= 0 else (255, 150, 70),
                         (min(mid, mid + fill), bar.y, abs(fill), bar.h))
        pygame.draw.line(scr, (255, 255, 255), (mid, bar.y - 3), (mid, bar.bottom + 3), 2)
        word = "STRAIGHT" if abs(steer_val) < 0.08 else ("RIGHT" if steer_val > 0 else "LEFT")
        scr.blit(self.f_sm.render(word, True, (235, 235, 150)), (10, by + 18))
        scr.blit(self.f_xs.render(f"{wheel_deg:+.1f} deg", True, (235, 235, 150)), (10, by + 40 - 4))
        # dial
        dc, dr = (PANEL_W - 40, by + 22), 30
        pygame.draw.circle(scr, (10, 12, 12), dc, dr)
        pygame.draw.circle(scr, (150, 255, 90), dc, dr, 3)
        for off in (90, 210, 330):
            a = math.radians(wheel_deg + off)
            pygame.draw.line(scr, (150, 255, 90), dc, (dc[0] + (dr - 4) * math.cos(a), dc[1] + (dr - 4) * math.sin(a)), 2)
        # info
        y = 28 + 240 + 14
        info = [("MODE", mode, (255, 210, 90)),
                ("SPEED", f"{int(g.speed * 3.6)} km/h", (200, 230, 255)),
                ("CRASHES", str(g.crashes), (255, 120, 110))]
        for n, (a, b, col) in enumerate(info):
            scr.blit(self.f_xs.render(a, True, (130, 130, 145)), (14, y + n * 24))
            scr.blit(self.f_sm.render(b, True, col), (110, y + n * 24 - 2))
        y += 90
        help_lines = ["Hold both hands like a wheel,", "tilt left / right to steer.", "",
                      "A / D  or  < >   keyboard steer", "W gas   S / Space brake",
                      "R restart      ESC quit"]
        for n, line in enumerate(help_lines):
            scr.blit(self.f_xs.render(line, True, (150, 150, 165)), (14, y + n * 20))

    # ---- frame -----------------------------------------------------------
    def draw(self, g, tracker, steer_val, wheel_deg, mode):
        night = 0.5 - 0.5 * math.cos(g.dist * 0.0007)
        self.world(g)
        self.cockpit(g, steer_val * 90.0, night)
        self.hud(g)
        if g.crash_t > 0:
            f = pygame.Surface((GW, GH), pygame.SRCALPHA)
            f.fill((255, 40, 30, int(120 * g.crash_t / 0.5)))
            self.view.blit(f, (0, 0))
        ox = oy = 0
        if g.shake > 0:
            ox, oy = random.randint(-int(g.shake), int(g.shake)), random.randint(-int(g.shake), int(g.shake))
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.view, (PANEL_W + ox, oy))
        self.panel(tracker, steer_val, wheel_deg, mode, g)
        pygame.display.flip()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--selftest", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.selftest:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Virtual Steering Racer")
    clock = pygame.time.Clock()
    rend, game = Renderer(screen), Game()

    tracker = None
    if not args.no_camera and not args.selftest:
        tracker = HandSteer(args.camera)
        tracker.start()
    elif not args.selftest:
        tracker = HandSteer(args.camera)
        tracker.error = "camera disabled (--no-camera)"

    frame = 0
    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        if args.selftest:
            dt = 1 / 60
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                game.reset()
        keys = pygame.key.get_pressed()
        kb = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        gas = keys[pygame.K_w] or keys[pygame.K_UP]
        brake = keys[pygame.K_s] or keys[pygame.K_DOWN] or keys[pygame.K_SPACE]

        if args.selftest:
            steer_in = clamp(-game.player_x * 1.6 + track_curve(game.dist + 32) * 0.9
                             + math.sin(frame * 0.03) * 0.25, -1, 1)
            wheel_deg, mode = steer_in * ANGLE_MAX, "SELFTEST"
        elif tracker and tracker.hands >= 2:
            a = tracker.angle
            a = 0.0 if abs(a) < DEADZONE else a - math.copysign(DEADZONE, a)
            steer_in = clamp(a / (ANGLE_MAX - DEADZONE), -1, 1)
            wheel_deg, mode = tracker.angle, "HAND TRACKING"
        else:
            steer_in = float(kb)
            wheel_deg, mode = steer_in * ANGLE_MAX, "KEYBOARD"

        game.update(dt, steer_in, gas, brake)
        rend.draw(game, tracker, game.steer, wheel_deg if mode == "HAND TRACKING" else game.steer * ANGLE_MAX, mode)

        frame += 1
        if args.selftest:
            if frame in (args.selftest // 2, args.selftest):
                pygame.image.save(screen, f"selftest_{frame}.png")
            if frame >= args.selftest:
                break
    if tracker:
        tracker.running = False
    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
