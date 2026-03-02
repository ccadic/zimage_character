import os
import re
import gc
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk, Image
import torch
from diffusers import ZImagePipeline


MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
BASE_NAME = "milo_"
EXT = ".png"


# -----------------------------
# VRAM / RAM cleanup + stats
# -----------------------------
def vram_cleanup():
    """Nettoyage agressif VRAM + RAM (utile entre itérations)."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def vram_stats():
    if not torch.cuda.is_available():
        return "CUDA: off"
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    return f"VRAM alloc={alloc:.0f}MB reserved={reserved:.0f}MB"


def vram_percent(mode="reserved"):
    """
    mode:
      - 'allocated' : mémoire réellement allouée (tensors)
      - 'reserved'  : mémoire réservée par le cache PyTorch (souvent plus proche de ce qui coince)
    Retour: (pct:int, used_bytes:int, total_bytes:int, free_bytes:int)
    """
    if not torch.cuda.is_available():
        return 0, 0, 0, 0

    # mem_get_info: (free, total) en bytes
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    used = reserved if mode == "reserved" else allocated
    pct = int((used / total) * 100) if total > 0 else 0
    return pct, used, total, free


# -----------------------------
# Utilitaires fichiers
# -----------------------------
def next_filename(base_name=BASE_NAME, extension=EXT, folder=".") -> str:
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if re.match(rf"^{re.escape(base_name)}\d+{re.escape(extension)}$", f)]
    if not existing:
        n = 1
    else:
        nums = [int(re.search(r"\d+", f).group()) for f in existing]
        n = max(nums) + 1
    return os.path.join(folder, f"{base_name}{n:04d}{extension}")


def clamp_int(v, lo, hi, default):
    try:
        v = int(v)
        return max(lo, min(hi, v))
    except Exception:
        return default


def clamp_float(v, lo, hi, default):
    try:
        v = float(v)
        return max(lo, min(hi, v))
    except Exception:
        return default


# -----------------------------
# Prompt builder (modulaire)
# -----------------------------
def build_prompt(character_name: str, view: str, emotion: str, action: str, scene: str, extra: str) -> str:
    # ADN stable du personnage (IMPORTANT: pas de fond ici, sinon ça écrase les décors)
    dna = f"""
Full body original children book character named {character_name},
3D stylized toy character, cheerful fantasy story mascot,
toy-like smooth geometry, rounded simplified shapes, no sharp edges,
soft clay rendering style, matte fabric materials, subtle subsurface scattering skin,
head slightly oversized (40% of total height), compact rounded body, short soft limbs,
short fluffy light blond hair, warm brown eyes, subtle rosy cheeks, gentle smile baseline,
wearing a light green and dark purple jester-style hat with two curved soft points,
each tip ending with a small matte silver sphere (not bell),
bright yellow textured sweater (soft felt material), no scarf,
light pastel green shorts, chunky violet shoes with rounded soles, no visible laces,
distinct color palette: light green, vibrant yellow, deep violet,
original standalone mascot, unique silhouette, NOT inspired by any existing character.
"""

    # Turnaround / angle
    view_map = {
        "Face": "front view, symmetric posture, character facing camera",
        "Profil": "side profile view, 90-degree turn, clear silhouette profile",
        "Dos": "back view, character facing away, show clothing and hat from behind",
    }
    view_text = view_map.get(view, "front view")

    # Émotions
    emotion_map = {
        "Neutre": "neutral friendly expression, calm eyes, small smile",
        "Joyeux": "happy expression, brighter smile, slightly raised cheeks",
        "Surpris": "surprised expression, slightly open mouth, widened eyes",
        "Triste": "sad expression, small downturned mouth, gentle watery eyes",
        "Colère": "angry expression but child-friendly, furrowed brows, pout mouth",
        "Peur": "scared expression, tense shoulders, worried eyes",
        "Malicieux": "mischievous grin, one eyebrow slightly raised, playful vibe",
        "Fatigué": "sleepy expression, half-closed eyes, relaxed mouth",
    }
    emotion_text = emotion_map.get(emotion, emotion_map["Neutre"])

    # Actions / poses
    action_map = {
        "Statique (salut)": "standing pose, waving hand, balanced stance",
        "Courir": "running pose, one leg forward, clear readable pose, no heavy motion blur",
        "Sauter": "jumping pose, feet off ground, joyful motion, clear silhouette",
        "Dormir": "sleeping pose, sitting or lying down, eyes closed, peaceful",
        "Lire": "reading a small book, focused calm expression",
        "Pointer": "pointing gesture, educational mascot pose",
        "Tenir un objet": "holding a small object with both hands (generic prop)",
    }
    action_text = action_map.get(action, action_map["Statique (salut)"])

    # Décors (plus directifs + lisibles)
    scene_map = {
        "Studio (fond gris)": (
            "clean light grey studio background, seamless backdrop, "
            "soft ground shadow, warm neutral studio lighting"
        ),
        "Village": (
            "cozy small European village street background, pastel houses, cobblestone road, "
            "soft morning sunlight, gentle depth of field, storybook environment, "
            "character standing on ground, background clearly visible"
        ),
        "Forêt": (
            "friendly forest clearing background, soft sun rays through leaves, "
            "green foliage, mossy ground, subtle bokeh, child-friendly nature scene, "
            "character standing on ground, background clearly visible"
        ),
        "Chambre": (
            "cute child bedroom interior background, bed and plush toys, warm lamp light, "
            "soft cozy atmosphere, character standing on floor, background clearly visible"
        ),
        "École": (
            "simple classroom background, chalkboard, wooden desks, warm friendly colors, "
            "character standing on floor, background clearly visible"
        ),
    }
    scene_text = scene_map.get(scene, scene_map["Studio (fond gris)"])

    # Qualité / rendu
    quality = """
high detail 3D render, ultra clean,
cinematic soft lighting, subtle rim light,
full body in frame, centered composition,
background is clearly visible and matches the chosen scene,
consistent character design reference quality.
"""

    # Pour forcer la rupture "studio" quand on n'est pas en studio
    extra_effective = (extra or "").strip()
    if scene != "Studio (fond gris)":
        extra_effective = (extra_effective + " no studio background, no plain grey background").strip()

    prompt = f"""{dna}
{view_text}.
{emotion_text}.
{action_text}.
{scene_text}.
{quality}
{extra_effective}
"""
    return " ".join(prompt.split())


def build_negative_prompt() -> str:
    return (
        "realistic human anatomy, adult proportions, hyper realistic skin pores, wrinkles, "
        "horror, creepy, dark gore, anime, manga, cyberpunk, low poly, flat 2D watercolor, "
        "photorealistic photography, brand logos, text, watermark, signature"
    )


# -----------------------------
# App
# -----------------------------
class ZImageStudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ZImage Character Studio — Turnaround + Emotions + Actions")
        self.geometry("1100x780")

        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = tk.StringVar(value="./outputs")

        # UI vars
        self.character_name = tk.StringVar(value="Milo the Meadow Sprout")

        self.view_face = tk.BooleanVar(value=True)
        self.view_profile = tk.BooleanVar(value=True)
        self.view_back = tk.BooleanVar(value=True)

        self.emotion = tk.StringVar(value="Joyeux")
        self.action = tk.StringVar(value="Statique (salut)")
        self.scene = tk.StringVar(value="Studio (fond gris)")
        self.extra = tk.StringVar(value="")

        self.seed = tk.StringVar(value="42")
        self.steps = tk.StringVar(value="14")
        self.guidance = tk.StringVar(value="4.0")
        self.width = tk.StringVar(value="768")
        self.height = tk.StringVar(value="1024")
        self.batch = tk.StringVar(value="1")

        self.use_negative = tk.BooleanVar(value=True)
        self.use_xformers = tk.BooleanVar(value=False)

        # VRAM monitor
        self.vram_mode = tk.StringVar(value="reserved")  # reserved | allocated
        self.vram_refresh_ms = 10_000  # 10s
        self.preview_imgtk = None
        self.is_busy = False

        self._build_ui()
        self._update_status(f"Device: {self.device} — Modèle non chargé. — {vram_stats()}")
        self._start_vram_monitor()

    def _update_status(self, txt):
        self.status.set(txt)

    # -------- VRAM UI --------
    def _start_vram_monitor(self):
        self._update_vram_ui()
        self.after(self.vram_refresh_ms, self._start_vram_monitor)

    def _update_vram_ui(self):
        if not torch.cuda.is_available():
            self.vram_label.configure(text="VRAM: CUDA off")
            self.vram_bar["value"] = 0
            return

        pct, used, total, free = vram_percent(mode=self.vram_mode.get())
        used_gb = used / (1024**3)
        total_gb = total / (1024**3)
        free_gb = free / (1024**3)

        self.vram_bar["value"] = pct
        self.vram_label.configure(
            text=f"VRAM {pct}% | used={used_gb:.2f}GB / {total_gb:.2f}GB | free={free_gb:.2f}GB | mode={self.vram_mode.get()}"
        )

    def _build_ui(self):
        left = ttk.Frame(self, padding=12)
        left.pack(side="left", fill="y")

        right = ttk.Frame(self, padding=12)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="Personnage", font=("Arial", 12, "bold")).pack(anchor="w")
        ttk.Label(left, text="Nom (stabilise la cohérence)").pack(anchor="w", pady=(6, 0))
        ttk.Entry(left, textvariable=self.character_name, width=36).pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(left, text="Turnaround", font=("Arial", 12, "bold")).pack(anchor="w")
        ttk.Checkbutton(left, text="Face", variable=self.view_face).pack(anchor="w")
        ttk.Checkbutton(left, text="Profil", variable=self.view_profile).pack(anchor="w")
        ttk.Checkbutton(left, text="Dos", variable=self.view_back).pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(left, text="Émotion", font=("Arial", 12, "bold")).pack(anchor="w")
        emotions = ["Neutre", "Joyeux", "Surpris", "Triste", "Colère", "Peur", "Malicieux", "Fatigué"]
        ttk.OptionMenu(left, self.emotion, self.emotion.get(), *emotions).pack(anchor="w", fill="x")

        ttk.Label(left, text="Action / Attitude", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 0))
        actions = ["Statique (salut)", "Courir", "Sauter", "Dormir", "Lire", "Pointer", "Tenir un objet"]
        ttk.OptionMenu(left, self.action, self.action.get(), *actions).pack(anchor="w", fill="x")

        ttk.Label(left, text="Décor", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 0))
        scenes = ["Studio (fond gris)", "Village", "Forêt", "Chambre", "École"]
        ttk.OptionMenu(left, self.scene, self.scene.get(), *scenes).pack(anchor="w", fill="x")

        ttk.Label(left, text="Extra prompt (optionnel)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(left, textvariable=self.extra, width=36).pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(left, text="Paramètres", font=("Arial", 12, "bold")).pack(anchor="w")

        grid = ttk.Frame(left)
        grid.pack(anchor="w", pady=(6, 0), fill="x")

        ttk.Label(grid, text="Seed").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.seed, width=10).grid(row=0, column=1, padx=6)

        ttk.Label(grid, text="Steps").grid(row=1, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.steps, width=10).grid(row=1, column=1, padx=6)

        ttk.Label(grid, text="Guidance").grid(row=2, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.guidance, width=10).grid(row=2, column=1, padx=6)

        ttk.Label(grid, text="W").grid(row=3, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.width, width=10).grid(row=3, column=1, padx=6)

        ttk.Label(grid, text="H").grid(row=4, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.height, width=10).grid(row=4, column=1, padx=6)

        ttk.Label(grid, text="Nb images").grid(row=5, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.batch, width=10).grid(row=5, column=1, padx=6)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Checkbutton(left, text="Negative prompt (recommandé)", variable=self.use_negative).pack(anchor="w")
        ttk.Checkbutton(left, text="xFormers attention (si installé)", variable=self.use_xformers).pack(anchor="w")

        ttk.Label(left, text="Dossier sortie").pack(anchor="w", pady=(10, 0))
        ttk.Entry(left, textvariable=self.output_dir, width=36).pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        btns = ttk.Frame(left)
        btns.pack(fill="x")

        self.btn_load = ttk.Button(btns, text="Charger modèle", command=self.load_model)
        self.btn_load.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.btn_gen = ttk.Button(btns, text="Générer", command=self.generate)
        self.btn_gen.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.btn_purge = ttk.Button(btns, text="Purge VRAM", command=self.purge_vram)
        self.btn_purge.pack(side="left", expand=True, fill="x")

        ttk.Label(right, text="Aperçu (dernière image)", font=("Arial", 12, "bold")).pack(anchor="w")
        self.preview_label = ttk.Label(right)
        self.preview_label.pack(anchor="center", pady=10)

        ttk.Label(right, text="Prompt généré", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 0))
        self.prompt_box = tk.Text(right, height=10, wrap="word")
        self.prompt_box.pack(fill="both", expand=True)

        # ---- VRAM Monitor UI ----
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(right, text="VRAM Monitor", font=("Arial", 12, "bold")).pack(anchor="w")

        row = ttk.Frame(right)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="Mode:").pack(side="left")
        ttk.OptionMenu(row, self.vram_mode, self.vram_mode.get(), "reserved", "allocated").pack(side="left", padx=6)

        self.vram_label = ttk.Label(right, text="VRAM: n/a")
        self.vram_label.pack(anchor="w", pady=(6, 0))

        self.vram_bar = ttk.Progressbar(right, orient="horizontal", length=520, mode="determinate", maximum=100)
        self.vram_bar.pack(anchor="w", pady=(6, 0))

        self.status = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.status).pack(anchor="w", pady=(10, 0))

    def set_busy(self, busy: bool):
        self.is_busy = busy
        state = "disabled" if busy else "normal"
        self.btn_load.configure(state=state)
        self.btn_gen.configure(state=state)
        self.btn_purge.configure(state="normal" if not busy else "disabled")

    def purge_vram(self):
        vram_cleanup()
        self._update_status(f"Purge VRAM ✅ — {vram_stats()}")
        self._update_vram_ui()

    def load_model(self):
        if self.is_busy:
            return

        def _load():
            self.set_busy(True)
            try:
                self._update_status("Chargement du modèle…")
                self.update_idletasks()

                vram_cleanup()

                pipe = ZImagePipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )

                pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing()
                try:
                    pipe.vae.enable_slicing()
                    pipe.vae.enable_tiling()
                except Exception:
                    pass

                if self.use_xformers.get():
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass

                self.pipe = pipe
                vram_cleanup()
                self._update_status(f"Modèle chargé ✅ — {vram_stats()}")
                self._update_vram_ui()
            except Exception as e:
                self._update_status("Erreur chargement ❌")
                messagebox.showerror("Erreur", str(e))
            finally:
                self.set_busy(False)

        threading.Thread(target=_load, daemon=True).start()

    def _selected_views(self):
        views = []
        if self.view_face.get():
            views.append("Face")
        if self.view_profile.get():
            views.append("Profil")
        if self.view_back.get():
            views.append("Dos")
        return views

    def generate(self):
        if self.is_busy:
            return
        if self.pipe is None:
            messagebox.showwarning("Modèle", "Clique d'abord sur 'Charger modèle'.")
            return

        views = self._selected_views()
        if not views:
            messagebox.showwarning("Turnaround", "Coche au moins une vue (Face/Profil/Dos).")
            return

        seed = clamp_int(self.seed.get(), 0, 2_147_483_647, 42)
        steps = clamp_int(self.steps.get(), 4, 60, 14)
        guidance = clamp_float(self.guidance.get(), 0.0, 12.0, 4.0)
        w = clamp_int(self.width.get(), 256, 2048, 768)
        h = clamp_int(self.height.get(), 256, 2048, 1024)
        batch = clamp_int(self.batch.get(), 1, 20, 1)

        out_dir = self.output_dir.get().strip() or "./outputs"
        os.makedirs(out_dir, exist_ok=True)

        character_name = self.character_name.get().strip() or "Milo the Meadow Sprout"
        emotion = self.emotion.get()
        action = self.action.get()
        scene = self.scene.get()
        extra = self.extra.get().strip()

        neg = build_negative_prompt() if self.use_negative.get() else None

        sample_prompt = build_prompt(character_name, views[0], emotion, action, scene, extra)
        self.prompt_box.delete("1.0", "end")
        self.prompt_box.insert("1.0", sample_prompt)

        def _run():
            self.set_busy(True)
            try:
                self._update_status(f"Génération… — {vram_stats()}")
                self.update_idletasks()

                device = "cuda" if torch.cuda.is_available() else "cpu"
                g = torch.Generator(device).manual_seed(seed)

                saved_paths = []

                with torch.inference_mode():
                    for b in range(batch):
                        for view in views:
                            prompt = build_prompt(character_name, view, emotion, action, scene, extra)
                            kwargs = dict(
                                prompt=prompt,
                                height=h,
                                width=w,
                                num_inference_steps=steps,
                                guidance_scale=guidance,
                                generator=g,
                            )
                            if neg is not None:
                                kwargs["negative_prompt"] = neg

                            out = self.pipe(**kwargs)
                            img = out.images[0]

                            filename = next_filename(base_name=BASE_NAME, extension=EXT, folder=out_dir)
                            img.save(filename)
                            saved_paths.append(filename)

                            del img
                            del out
                            vram_cleanup()
                            # refresh monitor plus souvent pendant génération
                            self.after(0, self._update_vram_ui)

                last = saved_paths[-1]
                self._set_preview(last)
                self._update_status(
                    f"OK ✅ {len(saved_paths)} image(s) — Dernière: {os.path.basename(last)} — {vram_stats()}"
                )
                self.after(0, self._update_vram_ui)
            except Exception as e:
                self._update_status("Erreur génération ❌")
                messagebox.showerror("Erreur", str(e))
            finally:
                self.set_busy(False)

        threading.Thread(target=_run, daemon=True).start()

    def _set_preview(self, path):
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
            img.thumbnail((650, 450))
            self.preview_imgtk = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.preview_imgtk)
            del img
        except Exception:
            pass


if __name__ == "__main__":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    app = ZImageStudioApp()
    app.mainloop()
