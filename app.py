import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
import base64
import json
import math
import random
import time
import urllib.request
import urllib.parse
import tempfile
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Wildlife Camera-Trap Pipeline", layout="wide")

DEMO_LABEL = "Built-in live trail-cam demo"
FRAME_W, FRAME_H = 720, 480
FPS = 15
CHUNK_FRAMES = 24
CLIP_FRAMES = 60
MAX_TRACK_MISSED = 14
MIN_TRACK_FRAMES = 5
MAX_TRACK_FRAMES = 900

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "camera_trap_assets"

try:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    test_file = ASSET_DIR / ".write_test"
    test_file.write_bytes(b"ok")
    test_file.unlink()
except Exception:
    ASSET_DIR = Path(tempfile.gettempdir()) / "camera_trap_assets"
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_DIR = ASSET_DIR / "clips"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = ASSET_DIR / "sighting_log.db"

WIKI_SPECIES = [
    {"key": "white_tailed_deer", "wiki": "White-tailed_deer", "label": "White-tailed deer", "base_size": (220, 170), "speed": (1.4, 2.6)},
    {"key": "red_fox", "wiki": "Red_fox", "label": "Red fox", "base_size": (150, 110), "speed": (1.8, 3.2)},
    {"key": "raccoon", "wiki": "Raccoon", "label": "Raccoon", "base_size": (130, 105), "speed": (1.1, 2.1)},
    {"key": "coyote", "wiki": "Coyote", "label": "Coyote", "base_size": (160, 120), "speed": (2.0, 3.5)},
    {"key": "wild_boar", "wiki": "Wild_boar", "label": "Wild boar", "base_size": (170, 125), "speed": (1.2, 2.2)},
    {"key": "gray_squirrel", "wiki": "Eastern_gray_squirrel", "label": "Eastern gray squirrel", "base_size": (90, 75), "speed": (2.2, 4.0)},
    {"key": "black_bear", "wiki": "American_black_bear", "label": "American black bear", "base_size": (240, 190), "speed": (1.0, 1.8)},
]

COMMONS_VIDEO_QUERIES = [
    ("roe deer", "Roe deer"),
    ("deer forest", "Deer"),
    ("red fox", "Red fox"),
    ("wild boar", "Wild boar"),
    ("squirrel", "Squirrel"),
    ("bird feeder", "Bird"),
    ("bear", "Bear"),
]

COCO_ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

ANIMAL_KEYWORDS = {
    "deer", "fox", "bear", "wolf", "coyote", "raccoon", "boar", "wild boar", "hog", "pig", "squirrel", "marmot", "beaver",
    "otter", "badger", "weasel", "mink", "polecat", "ferret", "skunk", "porcupine", "opossum", "possum", "rabbit", "hare",
    "mouse", "rat", "vole", "hamster", "gerbil", "moose", "elk", "antelope", "gazelle", "impala", "zebra",
    "giraffe", "elephant", "rhinoceros", "hippopotamus", "bison", "buffalo", "ox", "cow", "sheep", "goat", "horse",
    "lion", "tiger", "leopard", "cheetah", "jaguar", "cougar", "lynx", "bobcat", "cat", "dog", "hyena", "mongoose",
    "meerkat", "monkey", "ape", "gorilla", "chimpanzee", "orangutan", "gibbon", "baboon", "macaque", "marmoset",
    "bird", "eagle", "hawk", "owl", "falcon", "kite", "vulture", "heron", "crane", "stork", "ibis", "flamingo",
    "penguin", "pelican", "cormorant", "gull", "tern", "duck", "goose", "swan", "quail", "grouse", "turkey", "peacock",
    "parrot", "macaw", "cockatoo", "toucan", "hummingbird", "woodpecker", "crow", "raven", "jay", "magpie", "robin",
    "sparrow", "finch", "bunting", "warbler", "thrush", "wren", "lizard", "snake", "iguana", "chameleon", "gecko",
    "turtle", "tortoise", "crocodile", "alligator", "frog", "toad", "salamander", "newt", "fish", "shark", "ray",
    "seal", "sea lion", "walrus", "whale", "dolphin", "otter"
}

MODE_TEXT = {
    "real_video": "REAL FOOTAGE mode: looping real wildlife video (Wikimedia Commons) with live motion detection",
    "real_images": "REAL IMAGES mode: real wildlife photos animated over a real forest photograph",
    "synthetic": "OFFLINE fallback: synthetic trail-cam scene (no internet detected)",
}


def ensure_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                species TEXT,
                confidence REAL,
                image BLOB,
                track_id INTEGER
            )
            """
        )


ensure_db()


def do_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def http_get_bytes(url, timeout=6):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (WildlifeCameraTrapDemo)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def make_placeholder_image(label, size=(240, 180), color=(70, 120, 70)):
    img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    cv2.ellipse(img, (size[0] // 2, size[1] // 2), (size[0] // 2 - 6, size[1] // 2 - 6), 0, 0, 360, (230, 230, 230), -1)
    cv2.putText(img, label, (8, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def try_fetch_wikipedia_image(title, dest_path, timeout=5):
    dest_path = Path(dest_path)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path, True

    try:
        api = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_"))
        payload = json.loads(http_get_bytes(api, timeout=timeout).decode("utf-8"))

        url = payload.get("thumbnail", {}).get("source")
        if url:
            for token in ("/320px-", "/220px-", "/250px-", "/500px-"):
                if token in url:
                    url = url.replace(token, "/640px-")
                    break
        else:
            url = payload.get("originalimage", {}).get("source")

        if not url:
            return None, False

        img_bytes = http_get_bytes(url, timeout=timeout + 5)
        dest_path.write_bytes(img_bytes)

        if dest_path.exists() and dest_path.stat().st_size > 0:
            return dest_path, True
    except Exception:
        pass

    return None, False


def search_commons_video(query, timeout=12):
    api = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search"
        "&gsrsearch=" + urllib.parse.quote("filetype:video " + query) +
        "&gsrnamespace=6&gsrlimit=10&prop=imageinfo&iiprop=url|size|mime"
    )
    payload = json.loads(http_get_bytes(api, timeout=timeout).decode("utf-8"))
    pages = payload.get("query", {}).get("pages", {})

    candidates = []
    for page in pages.values():
        for ii in page.get("imageinfo", []):
            url = ii.get("url", "")
            size = ii.get("size", 0)
            if url.lower().endswith((".webm", ".ogv", ".mp4")) and size > 0:
                candidates.append((size, url))

    candidates.sort(key=lambda t: t[0])
    for size, url in candidates:
        if 150000 <= size <= 40000000:
            return url
    return None


@st.cache_resource(show_spinner=False)
def fetch_real_video():
    for query, species in COMMONS_VIDEO_QUERIES:
        try:
            url = search_commons_video(query, timeout=12)
            if not url:
                continue

            ext = Path(urllib.parse.urlparse(url).path).suffix or ".webm"
            dest = VIDEO_DIR / ("real_" + query.replace(" ", "_") + ext)

            if not dest.exists():
                data = http_get_bytes(url, timeout=120)
                if len(data) < 100000:
                    continue
                dest.write_bytes(data)

            cap_test = cv2.VideoCapture(str(dest))
            opened = cap_test.isOpened()
            ok_read, _ = cap_test.read() if opened else (False, None)
            cap_test.release()

            if opened and ok_read:
                return str(dest), species
        except Exception:
            continue

    return None, None


@st.cache_resource(show_spinner=False)
def load_demo_assets():
    network_ok = True
    assets = []

    for i, meta in enumerate(WIKI_SPECIES):
        img_path = None

        if network_ok:
            img_path, ok = try_fetch_wikipedia_image(meta["wiki"], ASSET_DIR / f"{meta['key']}.jpg", timeout=5)
            if not ok:
                network_ok = False

        if img_path is None:
            img_path = ASSET_DIR / f"{meta['key']}_placeholder.jpg"
            if not img_path.exists():
                color = (40 + i * 18, 90, 60 + i * 12)
                cv2.imwrite(str(img_path), make_placeholder_image(meta["label"], color=color))

        img = cv2.imread(str(img_path))
        if img is None:
            img = make_placeholder_image(meta["label"])

        assets.append({**meta, "img": img})

    human_path = None
    if network_ok:
        human_path, ok = try_fetch_wikipedia_image("Hiking", ASSET_DIR / "human.jpg", timeout=5)
        if not ok:
            human_path = None

    if human_path is None:
        human_path = ASSET_DIR / "human_placeholder.jpg"
        if not human_path.exists():
            cv2.imwrite(str(human_path), make_placeholder_image("Human", color=(90, 80, 160)))

    human_img = cv2.imread(str(human_path))
    if human_img is None:
        human_img = make_placeholder_image("Human")

    return {"species": assets, "human": human_img}


def resize_keep_aspect(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(16, int(w * scale))
    new_h = max(16, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def prepare_sprite(img, target_w, target_h):
    sprite = resize_keep_aspect(img, target_w, target_h)
    h, w = sprite.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (max(4, w // 2 - 2), max(4, h // 2 - 2)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    return sprite, mask


def overlay_sprite(frame, sprite, mask, x, y):
    h, w = sprite.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(FRAME_W, x + w)
    y2 = min(FRAME_H, y + h)

    sx1 = max(0, -x)
    sy1 = max(0, -y)
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    sub = sprite[sy1:sy2, sx1:sx2]
    m = mask[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0

    if roi.shape[:2] != sub.shape[:2]:
        return

    m3 = np.stack([m, m, m], axis=-1)
    frame[y1:y2, x1:x2] = (sub.astype(np.float32) * m3 + roi.astype(np.float32) * (1.0 - m3)).astype(np.uint8)


def make_background():
    bg = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    horizon = int(FRAME_H * 0.52)

    for y in range(FRAME_H):
        if y < horizon:
            t = y / max(1, horizon)
            color = (int(120 + 50 * t), int(90 + 60 * t), int(70 + 50 * t))
        else:
            t = (y - horizon) / max(1, FRAME_H - horizon)
            color = (int(35 + 45 * t), int(60 + 45 * t), int(30 + 30 * t))
        bg[y, :] = color

    rng = np.random.default_rng(7)

    for _ in range(28):
        x = int(rng.integers(0, FRAME_W))
        h = int(rng.integers(40, 150))
        w = int(rng.integers(8, 26))
        y0 = horizon + int(rng.integers(-10, 25))
        cv2.rectangle(bg, (x, y0 - h), (x + w, y0), (25, 45, 20), -1)
        cv2.ellipse(bg, (x + w // 2, y0 - h), (w * 2, h // 2), 0, 0, 360, (35, 70, 25), -1)

    pts = np.array(
        [
            [FRAME_W // 2 - 40, FRAME_H],
            [FRAME_W // 2 - 10, horizon],
            [FRAME_W // 2 + 10, horizon],
            [FRAME_W // 2 + 40, FRAME_H],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(bg, [pts], (45, 65, 60))

    noise = np.random.randint(0, 25, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
    bg = cv2.add(bg, noise)
    return bg


def clip_box(box, width, height):
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
    except Exception:
        return None

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_to_bytes(frame, box, size=(160, 160), quality=80):
    clipped = clip_box(box, frame.shape[1], frame.shape[0])
    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


def text_to_jpeg_bytes(text):
    img = make_placeholder_image(text)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else None


class MotionDetector:
    def __init__(self):
        self.sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=28, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, frame, min_area=800):
        mask = self.sub.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w * h < 1200:
                continue

            dets.append(
                {
                    "box": [x, y, x + w, y + h],
                    "category": "animal",
                    "label": "Unidentified wildlife",
                    "species": "Unidentified wildlife",
                    "conf": float(np.clip(0.45 + area / 200000.0, 0.35, 0.80)),
                    "event_id": None,
                }
            )

        return dets


class RealTrailCamSource:
    """Loops a REAL downloaded wildlife video and runs motion detection on it."""

    def __init__(self, cap, species):
        self.cap = cap
        self.species = species
        self.motion = MotionDetector()
        self.frame_idx = 0
        self.warm = 0

    def read(self):
        ret, frame = self.cap.read()

        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return False, None, []
            self.motion = MotionDetector()
            self.warm = 0

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        dets = []
        self.warm += 1
        if self.warm > 25:
            dets = self.motion.detect(frame, min_area=2500)
            for d in dets:
                d["species"] = self.species
                d["label"] = self.species

        self.frame_idx += 1
        return True, frame, dets


class Actor:
    def __init__(self, kind, label, sprite, mask, speed_range=(1.4, 2.8), conf_base=0.85, event_id=0):
        self.kind = kind
        self.label = label
        self.sprite = sprite
        self.mask = mask
        self.h, self.w = sprite.shape[:2]
        self.event_id = event_id

        direction = random.choice([-1, 1])
        self.x = -self.w - random.randint(5, 40) if direction > 0 else FRAME_W + random.randint(5, 40)

        min_y = int(FRAME_H * 0.48)
        max_y = max(min_y + 1, FRAME_H - self.h - 15)
        self.y = random.randint(min_y, max_y)

        self.vx = direction * random.uniform(*speed_range)
        self.age = 0
        self.done = False
        self.phase = random.uniform(0, 2 * math.pi)
        self.conf_base = conf_base

    def update(self):
        self.age += 1
        self.x += self.vx
        self.y += math.sin(self.age / 7.0 + self.phase) * 0.5

        if (self.vx > 0 and self.x > FRAME_W + 20) or (self.vx < 0 and self.x + self.w < -20):
            self.done = True

    def box(self):
        return [int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h)]

    def confidence(self):
        return float(np.clip(self.conf_base + random.uniform(-0.07, 0.06), 0.35, 0.99))


class DemoCameraTrap:
    def __init__(self):
        self.assets = load_demo_assets()
        self.mode = "synthetic"
        self.real_source = None
        self.frame_idx = 0
        self.actors = []
        self.next_spawn = 15
        self.night = False
        self.last_night_switch = 0
        self.actor_id = 1

        vid_path, species = fetch_real_video()
        if vid_path:
            cap = cv2.VideoCapture(vid_path)
            if cap.isOpened():
                self.real_source = RealTrailCamSource(cap, species)
                self.mode = "real_video"

        if self.real_source is None:
            bg_path, ok = try_fetch_wikipedia_image("Forest", ASSET_DIR / "forest_bg.jpg", timeout=6)
            bg = cv2.imread(str(bg_path)) if ok else None
            if bg is not None:
                self.bg = cv2.resize(bg, (FRAME_W, FRAME_H))
                self.mode = "real_images"
            else:
                self.bg = make_background()
        else:
            self.bg = make_background()

    def _overlay_stamp(self, frame, night):
        color = (0, 0, 255) if night else (255, 255, 255)
        stamp = "CAM-TRAP-01 | " + now_iso()
        cv2.putText(frame, stamp, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, stamp, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
        mode = "NIGHT IR" if night else "DAY COLOR"
        footer = f"{mode} | FRAME {self.frame_idx:06d}"
        cv2.putText(frame, footer, (10, FRAME_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    def _make_noise_sprite(self):
        w = random.randint(45, 95)
        h = random.randint(35, 75)
        sprite = np.random.randint(30, 220, (h, w, 3), dtype=np.uint8)
        sprite = cv2.GaussianBlur(sprite, (5, 5), 0)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return sprite, mask

    def _spawn(self):
        roll = random.random()

        if roll < 0.72:
            meta = random.choice(self.assets["species"])
            scale = random.uniform(0.75, 1.2)
            tw = int(meta["base_size"][0] * scale)
            th = int(meta["base_size"][1] * scale)
            sprite, mask = prepare_sprite(meta["img"], tw, th)
            actor = Actor("animal", meta["label"], sprite, mask, meta["speed"], random.uniform(0.78, 0.96), self.actor_id)
            self.actor_id += 1
            self.actors.append(actor)

        elif roll < 0.88:
            sprite, mask = prepare_sprite(self.assets["human"], 130, 190)
            actor = Actor("human", "Human", sprite, mask, (0.9, 1.6), random.uniform(0.82, 0.95), self.actor_id)
            self.actor_id += 1
            self.actors.append(actor)

        else:
            sprite, mask = self._make_noise_sprite()
            actor = Actor("empty", "Foliage motion", sprite, mask, (2.5, 5.0), random.uniform(0.35, 0.65), self.actor_id)
            self.actor_id += 1
            self.actors.append(actor)

    def read(self):
        if self.real_source is not None:
            ret, frame, dets = self.real_source.read()
            if not ret:
                return False, None, []
            self._overlay_stamp(frame, night=False)
            self.frame_idx += 1
            return True, frame, dets

        if self.frame_idx >= self.next_spawn:
            self._spawn()
            self.next_spawn = self.frame_idx + random.randint(35, 95)

        frame = self.bg.copy()

        self.actors = [a for a in self.actors if not a.done]
        self.actors.sort(key=lambda a: a.y)

        dets = []

        for actor in self.actors:
            actor.update()
            overlay_sprite(frame, actor.sprite, actor.mask, int(actor.x), int(actor.y))

            if not actor.done:
                box = actor.box()
                if box[2] > 0 and box[0] < FRAME_W and box[3] > 0 and box[1] < FRAME_H:
                    dets.append(
                        {
                            "box": box,
                            "category": actor.kind,
                            "label": actor.label,
                            "species": actor.label if actor.kind == "animal" else actor.label,
                            "conf": actor.confidence(),
                            "event_id": actor.event_id,
                        }
                    )

        if self.frame_idx - self.last_night_switch > 900:
            self.night = not self.night
            self.last_night_switch = self.frame_idx

        if self.night:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            frame = cv2.merge([gray // 4, gray // 4, gray])

        noise = np.random.normal(0, 2.5 if not self.night else 6.0, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        self._overlay_stamp(frame, self.night)

        self.frame_idx += 1
        return True, frame, dets


def get_demo_source():
    if "demo_source" not in st.session_state or st.session_state.demo_source is None:
        st.session_state.demo_source = DemoCameraTrap()
    return st.session_state.demo_source


def refresh_assets():
    for fn in (load_demo_assets, fetch_real_video):
        try:
            fn.clear()
        except Exception:
            pass

    st.session_state.demo_source = None
    st.session_state.running = True
    st.session_state.clip_path = None
    st.session_state.clip_format = None
    st.session_state.clip_buffer = []
    st.session_state.last_frame = None


def imagenet_name_is_animal(name):
    lower = f" {name.lower()} "
    for keyword in ANIMAL_KEYWORDS:
        if " " in keyword:
            if keyword in lower:
                return True
        else:
            if f" {keyword} " in lower:
                return True
    return False


class AIModels:
    def __init__(self):
        self.yolo = None
        self.torch = None
        self.clf = None
        self.device = "cpu"
        self.class_names = []
        self.messages = []
        self.ready = False

    def load(self):
        try:
            from ultralytics import YOLO

            self.yolo = YOLO("yolov8n.pt")
            self.ready = True
            self.messages.append("YOLOv8 loaded")
        except Exception as e:
            self.messages.append(f"YOLO unavailable: {e.__class__.__name__}")

        try:
            import torch
            import torchvision

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            try:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                self.clf = torchvision.models.resnet18(weights=weights).to(self.device).eval()
                self.class_names = list(weights.meta.get("categories", []))
            except Exception:
                self.clf = torchvision.models.resnet18(weights="IMAGENET1K_V1").to(self.device).eval()
                self.class_names = []

            self.ready = True
            self.messages.append("ResNet18 classifier loaded")
        except Exception as e:
            self.messages.append(f"Classifier unavailable: {e.__class__.__name__}")

    def predict_species(self, crop):
        if self.clf is None or self.torch is None or crop is None or crop.size == 0:
            return None, 0.0

        torch = self.torch

        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        arr = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.clf(tensor)
            probs = torch.softmax(out, dim=1)[0]
            top = torch.topk(probs, 8)

        idxs = top.indices.cpu().numpy().tolist()
        vals = top.values.cpu().numpy().tolist()

        for idx, p in zip(idxs, vals):
            name = self.class_names[idx] if self.class_names and idx < len(self.class_names) else None
            if name and imagenet_name_is_animal(name):
                return name.split(",")[0].strip().title(), float(p)

        return None, float(vals[0]) if vals else 0.0

    def detect(self, frame):
        dets = []

        if self.yolo is None:
            return dets

        try:
            results = self.yolo(frame, verbose=False)
            result = results[0]
            names = result.names

            for b in result.boxes:
                cls = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                name = names.get(cls, str(cls))

                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]

                if name == "person":
                    category = "human"
                    species = "Human"
                elif name in COCO_ANIMALS:
                    category = "animal"
                    species = name.title()
                else:
                    continue

                if conf < 0.30:
                    continue

                crop = frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]

                if category == "animal":
                    sp, sp_conf = self.predict_species(crop)
                    if sp:
                        species = sp
                        conf = float((conf + sp_conf) / 2.0)

                dets.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "category": category,
                        "label": species,
                        "species": species,
                        "conf": float(np.clip(conf, 0.0, 1.0)),
                        "event_id": None,
                    }
                )
        except Exception:
            pass

        return dets


@st.cache_resource(show_spinner=False)
def get_ai_models():
    model = AIModels()
    model.load()
    return model


class ExternalVideoSource:
    def __init__(self, cap):
        self.cap = cap
        self.motion = MotionDetector()
        self.frame_idx = 0

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None, []

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        dets = self.motion.detect(frame)
        self.frame_idx += 1
        return True, frame, dets


def open_external_source(choice, uploaded, url):
    cap = None

    try:
        if choice == "Upload video" and uploaded is not None:
            safe_name = "".join(c for c in uploaded.name if c.isalnum() or c in "._-") or "upload.mp4"
            tmp_path = ASSET_DIR / safe_name
            tmp_path.write_bytes(uploaded.getbuffer())
            cap = cv2.VideoCapture(str(tmp_path))

        elif choice == "Webcam":
            cap = cv2.VideoCapture(0)

        elif choice == "Video URL/RTSP" and url:
            cap = cv2.VideoCapture(url)

        if cap is not None and cap.isOpened():
            return ExternalVideoSource(cap)

    except Exception:
        pass

    return None


def release_external():
    src = st.session_state.get("external_source")
    if src is not None:
        try:
            src.cap.release()
        except Exception:
            pass

    st.session_state.external_source = None


def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def update_tracks(dets, frame_idx, frame):
    trackable = [d for d in dets if d.get("category") in ("animal", "human")]
    unmatched = list(range(len(trackable)))

    for track in st.session_state.tracks:
        best_iou = 0.0
        best_idx = None

        for idx in unmatched:
            det = trackable[idx]
            if det["category"] != track["category"]:
                continue

            val = iou(track["box"], det["box"])
            if val > best_iou:
                best_iou = val
                best_idx = idx

        if best_idx is not None and best_iou >= 0.12:
            det = trackable[best_idx]
            unmatched.remove(best_idx)

            track["box"] = det["box"]
            track["missed"] = 0
            track["frames"] += 1
            track["last_seen"] = frame_idx

            if det.get("event_id") is not None:
                track["event_id"] = det.get("event_id")

            if float(det.get("conf", 0.0)) > track["best_conf"]:
                track["best_conf"] = float(det.get("conf", 0.0))
                track["species"] = det.get("species") or det.get("label") or track["species"]

                crop = crop_to_bytes(frame, det["box"])
                if crop:
                    track["best_crop"] = crop
        else:
            track["missed"] += 1

    for idx in unmatched:
        det = trackable[idx]
        crop = crop_to_bytes(frame, det["box"])

        track = {
            "id": st.session_state.track_id,
            "category": det["category"],
            "species": det.get("species") or det.get("label") or ("Human" if det["category"] == "human" else "Unidentified wildlife"),
            "box": det["box"],
            "frames": 1,
            "missed": 0,
            "best_conf": float(det.get("conf", 0.5)),
            "best_crop": crop,
            "last_seen": frame_idx,
            "created": frame_idx,
            "event_id": det.get("event_id"),
        }

        st.session_state.track_id += 1
        st.session_state.tracks.append(track)

    finalized = []
    remaining = []

    for track in st.session_state.tracks:
        lost = track["missed"] > MAX_TRACK_MISSED or (frame_idx - track["last_seen"] > MAX_TRACK_MISSED)
        too_long = track["frames"] > MAX_TRACK_FRAMES

        if lost or too_long:
            finalized.append(track)
        else:
            remaining.append(track)

    st.session_state.tracks = remaining
    return finalized


def insert_sighting(species, confidence, image_bytes, track_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO sightings (timestamp, species, confidence, image, track_id) VALUES (?, ?, ?, ?, ?)",
            (now_iso(), species, float(confidence), sqlite3.Binary(image_bytes or b""), int(track_id)),
        )


def log_sighting(track):
    species = track.get("species") or "Unidentified wildlife"
    now = time.time()
    last = st.session_state.species_last.get(species, 0.0)

    if now - last < 2.0:
        return False

    if not track.get("best_crop"):
        track["best_crop"] = text_to_jpeg_bytes(species)

    st.session_state.species_last[species] = now
    st.session_state.species_counts[species] = st.session_state.species_counts.get(species, 0) + 1

    insert_sighting(species, track["best_conf"], track["best_crop"], track["id"])
    return True


def update_and_annotate(frame, dets, frame_idx):
    annotated = frame.copy()
    finalized = update_tracks(dets, frame_idx, frame)

    for track in finalized:
        if track["category"] == "animal":
            if track["frames"] >= MIN_TRACK_FRAMES:
                log_sighting(track)
        elif track["category"] == "human":
            eid = track.get("event_id")
            if eid is None or eid not in st.session_state.human_ids:
                if eid is not None:
                    st.session_state.human_ids.add(eid)
                st.session_state.humans_filtered += 1

    for det in dets:
        if det.get("category") == "empty":
            eid = det.get("event_id")
            if eid is not None and eid not in st.session_state.false_ids:
                st.session_state.false_ids.add(eid)
                st.session_state.empty_filtered += 1

            clipped = clip_box(det["box"], frame.shape[1], frame.shape[0])
            if clipped is not None:
                x1, y1, x2, y2 = clipped
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (120, 120, 120), 1)
                cv2.putText(annotated, "Motion ignored", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

    for track in st.session_state.tracks:
        clipped = clip_box(track["box"], frame.shape[1], frame.shape[0])
        if clipped is None:
            continue

        x1, y1, x2, y2 = clipped
        color = (0, 220, 0) if track["category"] == "animal" else (0, 165, 255)
        label = f"{track['species']} {track['best_conf']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

    return annotated


def get_recent_html(limit=8):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT timestamp, species, confidence, image FROM sightings ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except Exception:
        return "<i>Sighting log unavailable.</i>"

    if not rows:
        return "<i>Waiting for first confirmed animal sighting...</i>"

    html = "<div style='display:flex;flex-direction:column;gap:8px;'>"

    for ts, species, conf, img in rows:
        b64 = base64.b64encode(img).decode() if img else ""
        html += f"""
        <div style="display:flex;gap:10px;border:1px solid #2c3e50;border-radius:10px;padding:8px;background:#111418;">
            <img src="data:image/jpeg;base64,{b64}" style="width:84px;height:84px;object-fit:cover;border-radius:8px;"/>
            <div style="font-size:0.85rem;">
                <div style="font-weight:600;">{species}</div>
                <div>Confidence: {float(conf):.2f}</div>
                <div style="color:#9aa4b2;">{ts}</div>
            </div>
        </div>
        """

    html += "</div>"
    return html


def counts_df():
    counts = st.session_state.species_counts
    if not counts:
        return pd.DataFrame({"Species": [], "Sightings": []})

    df = pd.DataFrame(list(counts.items()), columns=["Species", "Sightings"])
    df = df.sort_values("Sightings", ascending=False).reset_index(drop=True)
    return df


def stats_markdown():
    total = sum(st.session_state.species_counts.values())
    return (
        f"**Confirmed sightings:** {total}  \n"
        f"**Humans filtered:** {st.session_state.humans_filtered}  \n"
        f"**Empty motion filtered:** {st.session_state.empty_filtered}  \n"
        f"**Active tracks:** {len(st.session_state.tracks)}"
    )


def save_clip(frames):
    if not frames:
        return None, None

    h, w = frames[0].shape[:2]

    mp4_path = VIDEO_DIR / "live_demo.mp4"
    try:
        writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()

            if mp4_path.exists() and mp4_path.stat().st_size > 5000:
                return mp4_path, "video"
    except Exception:
        pass

    try:
        from PIL import Image

        gif_path = VIDEO_DIR / "live_demo.gif"
        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        imgs[0].save(
            gif_path,
            save_all=True,
            append_images=imgs[1:],
            duration=int(1000 / FPS),
            loop=0,
            optimize=True,
        )

        if gif_path.exists() and gif_path.stat().st_size > 5000:
            return gif_path, "gif"
    except Exception:
        pass

    return None, None


def show_clip(placeholder):
    path_str = st.session_state.get("clip_path")
    if not path_str:
        return

    path = Path(path_str)
    if not path.exists():
        return

    fmt = st.session_state.get("clip_format", "video")

    try:
        if fmt == "gif":
            placeholder.image(str(path), caption="Auto demo clip (real footage)", use_container_width=True)
        else:
            placeholder.video(path.read_bytes())
    except Exception:
        try:
            placeholder.download_button("Download demo clip", path.read_bytes(), file_name=path.name)
        except Exception:
            pass


def reset_all():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM sightings")
    except Exception:
        pass

    st.session_state.running = True
    st.session_state.source_selector = DEMO_LABEL
    st.session_state.previous_source_choice = DEMO_LABEL

    st.session_state.tracks = []
    st.session_state.track_id = 1

    st.session_state.species_counts = {}
    st.session_state.species_last = {}

    st.session_state.false_ids = set()
    st.session_state.human_ids = set()

    st.session_state.humans_filtered = 0
    st.session_state.empty_filtered = 0

    st.session_state.clip_buffer = []
    st.session_state.clip_path = None
    st.session_state.clip_format = None
    st.session_state.last_frame = None

    st.session_state.demo_source = None
    release_external()


DEFAULTS = {
    "running": True,
    "source_selector": DEMO_LABEL,
    "previous_source_choice": DEMO_LABEL,
    "tracks": [],
    "track_id": 1,
    "species_counts": {},
    "species_last": {},
    "false_ids": set(),
    "human_ids": set(),
    "humans_filtered": 0,
    "empty_filtered": 0,
    "clip_buffer": [],
    "clip_path": None,
    "clip_format": None,
    "external_source": None,
    "last_frame": None,
    "demo_source": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("Wildlife Camera-Trap Analysis Pipeline")
st.caption(
    "The live demo starts automatically. It prefers REAL wildlife footage downloaded from Wikimedia Commons (no API key); "
    "if video decoding is unavailable it animates REAL wildlife photos over a REAL forest photo; fully offline it uses a synthetic scene."
)

st.sidebar.title("Camera-trap controls")

source_choice = st.sidebar.selectbox(
    "Video source",
    [DEMO_LABEL, "Upload video", "Webcam", "Video URL/RTSP"],
    key="source_selector",
)

uploaded_file = None
url_text = ""

if source_choice == "Upload video":
    uploaded_file = st.sidebar.file_uploader("Upload trail-cam video", type=["mp4", "avi", "mov", "mkv"])
elif source_choice == "Video URL/RTSP":
    url_text = st.sidebar.text_input("Stream URL", placeholder="https://... or rtsp://...")

use_ai = st.sidebar.checkbox("Use AI models if installed (YOLOv8 + ResNet18)", value=False)

models = None
if use_ai:
    models = get_ai_models()
    if models.messages:
        st.sidebar.caption("; ".join(models.messages))

if st.session_state.previous_source_choice != source_choice:
    release_external()
    st.session_state.previous_source_choice = source_choice
    st.session_state.running = source_choice == DEMO_LABEL
    st.session_state.tracks = []
    st.session_state.clip_buffer = []
    st.session_state.clip_path = None
    st.session_state.clip_format = None
    st.session_state.last_frame = None

bcol1, bcol2 = st.sidebar.columns(2)

if bcol1.button("Start / resume"):
    st.session_state.running = True
    if source_choice != DEMO_LABEL and st.session_state.external_source is None:
        st.session_state.external_source = open_external_source(source_choice, uploaded_file, url_text)

if bcol2.button("Stop"):
    st.session_state.running = False

st.sidebar.button("Reset log / restart demo", on_click=reset_all)
st.sidebar.button("Re-download real demo media", on_click=refresh_assets)

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Live annotated feed")
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

with right:
    st.subheader("Sighting log")
    log_placeholder = st.empty()
    st.subheader("Session population count")
    counts_placeholder = st.empty()

clip_placeholder = st.sidebar.empty()

source = None
source_error = None

if source_choice == DEMO_LABEL:
    loading_msg = st.sidebar.info("Loading real trail-cam media (first run downloads real footage)...")
    source = get_demo_source()
    loading_msg.empty()
    st.sidebar.success(MODE_TEXT.get(getattr(source, "mode", "synthetic"), ""))
else:
    source = st.session_state.external_source

    if source is None and st.session_state.running:
        source = open_external_source(source_choice, uploaded_file, url_text)
        st.session_state.external_source = source

    if source is None and st.session_state.running:
        st.session_state.running = False
        source_error = "Could not open the selected external video source."

if st.session_state.running and source is not None:
    frames_processed = 0
    clip_shown = bool(st.session_state.clip_path)

    while frames_processed < CHUNK_FRAMES and st.session_state.running:
        t0 = time.time()

        ret, frame, dets = source.read()
        if not ret:
            st.session_state.running = False
            st.info("Video source ended.")
            break

        if use_ai and models is not None and getattr(models, "ready", False):
            try:
                ai_dets = models.detect(frame)
                if ai_dets:
                    dets = ai_dets
            except Exception:
                pass

        frame_idx = getattr(source, "frame_idx", frames_processed)
        annotated = update_and_annotate(frame, dets, frame_idx)

        st.session_state.last_frame = annotated

        if st.session_state.clip_path is None:
            st.session_state.clip_buffer.append(cv2.resize(annotated, (480, 320)))
            if len(st.session_state.clip_buffer) > CLIP_FRAMES:
                st.session_state.clip_buffer.pop(0)

            if len(st.session_state.clip_buffer) >= CLIP_FRAMES:
                clip_path, clip_format = save_clip(st.session_state.clip_buffer)
                if clip_path:
                    st.session_state.clip_path = str(clip_path)
                    st.session_state.clip_format = clip_format
                    st.session_state.clip_buffer = []

        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        stats_placeholder.markdown(stats_markdown())

        if frames_processed % 6 == 0:
            log_placeholder.markdown(get_recent_html(), unsafe_allow_html=True)
            counts_placeholder.dataframe(counts_df(), use_container_width=True)

        if st.session_state.clip_path and not clip_shown:
            show_clip(clip_placeholder)
            clip_shown = True

        frames_processed += 1

        elapsed = time.time() - t0
        delay = max(0.005, (1.0 / FPS) - elapsed)
        time.sleep(delay)

    if st.session_state.running:
        do_rerun()

else:
    if source_error:
        st.sidebar.warning(source_error)

    if st.session_state.last_frame is not None:
        frame_placeholder.image(cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        frame_placeholder.info("Press Start / resume to begin, or choose the built-in demo.")

    stats_placeholder.markdown(stats_markdown())
    log_placeholder.markdown(get_recent_html(), unsafe_allow_html=True)
    counts_placeholder.dataframe(counts_df(), use_container_width=True)

    if st.session_state.clip_path:
        show_clip(clip_placeholder)