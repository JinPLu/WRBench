# WRBench project page

Static site for **Current World Models Lack a Persistent State Core** (WRBench),
served by **GitHub Pages** from this `docs/` folder at
<https://jinplu.github.io/WRBench/>.

## Files

- `index.html` — the single-page site.
- `assets/` — figures, plots, forensics montages, logos, and case PNGs.
- `videos/` — paper case-study clips referenced inline.
- `iclr2026_conference.pdf` — archived local paper PDF; the hero links to arXiv.
- `.nojekyll` — disables Jekyll so `assets/` and underscores are served verbatim.
- `robots.txt` — allows indexing.

## Local preview

```bash
cd docs
python3 -m http.server 8080
# open http://127.0.0.1:8080/
```

## Publish (GitHub Pages)

The site deploys automatically once GitHub Pages is configured to serve the
`docs/` folder of the `main` branch:

1. Push to `https://github.com/JinPLu/WRBench`.
2. Repo **Settings → Pages → Build and deployment**:
   - **Source**: *Deploy from a branch*
   - **Branch**: `main` / `/docs`
3. The site goes live at `https://jinplu.github.io/WRBench/`.

No build step is required — the page is plain HTML/CSS/JS with static assets.
