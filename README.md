# Hybrid Neural World Models — project page

Static project page for the paper *Hybrid Neural World Models for
Physical Dynamics* (Lakshmanan & Chopra, lossfunk).

Live: <https://pranavlakshmanan.github.io/hnwm/>

## What's in here

```
.
├── index.html        single-page site
├── style.css         all styling (no build step)
├── .nojekyll         skip Jekyll, serve files as-is
└── assets/           images, posters, and MP4 videos
```

Vanilla HTML and CSS. No JS framework, no build step, no dependencies
beyond Google Fonts (Source Serif 4, Inter, JetBrains Mono).

## Local preview

```bash
cd hnwm
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy to GitHub Pages

1. Create an empty public repo at `github.com/<your-username>/hnwm`.
2. Push the contents of this folder to the repo root.

   ```bash
   cd hnwm
   git init
   git add .
   git commit -m "initial commit"
   git branch -M main
   git remote add origin git@github.com:<your-username>/hnwm.git
   git push -u origin main
   ```

3. In the repo, go to **Settings → Pages**. Under *Build and
   deployment*, set the source to **Deploy from a branch**, branch
   `main`, folder `/ (root)`.
4. Site goes live at `https://<your-username>.github.io/hnwm/` in a
   minute or two.

## Asset notes

* `assets/explainer_v7.mp4` — the 30-second explainer animation
  (autoplays muted on page load).
* `assets/oregonator.mp4`, `euler.mp4`, `ball3d.mp4` — short looping
  trajectory clips, one per environment.
* `*_poster.jpg` files — fallback frames shown until the corresponding
  video has buffered. Safe to drop if you don't care about the
  pre-playback state.
* Wide multi-panel paper figures (e.g. `02_oregonator_visual.png`) are
  linked so a tap or click opens the full-resolution PNG in a new tab.
