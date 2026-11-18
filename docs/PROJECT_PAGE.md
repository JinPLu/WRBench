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

## Traffic Counter

The footer embeds a lightweight `visitor-badge.laobi.icu` SVG badge for page
visits. It works on GitHub Pages without JavaScript or a backend, but it is a
public external hit counter rather than strict unique-user analytics.

## Local preview

```bash
cd docs
python3 -m http.server 8080
# open the local preview URL shown by Python
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

## Public artifact links

Keep the project page, GitHub README, Hugging Face cards, and issue replies
aligned on these canonical URLs:

- Paper: <https://arxiv.org/abs/2606.20545>
- Hugging Face paper page: <https://huggingface.co/papers/2606.20545>
- Hugging Face Collection: <https://huggingface.co/collections/WRBench/wrbench-current-world-models-lack-a-persistent-state-core>
- Leaderboard: <https://huggingface.co/spaces/WRBench/wrbench-leaderboard>
- Natural-25: <https://huggingface.co/datasets/WRBench/wrbench-natural25>
- Results: <https://huggingface.co/datasets/WRBench/wrbench-results>
- Human annotations: <https://huggingface.co/datasets/WRBench/wrbench-human-annotations>
- Videos: <https://huggingface.co/datasets/WRBench/wrbench-videos>

The project's Release resources section links directly to the public datasets
and leaderboard. The project-page table preserves the 23-model paper results.
The Hugging Face results dataset and leaderboard track subsequent public
releases. Text-to-video addenda, including `minwm-wan-action2v`, are released
separately in the results dataset as `wrbench_t2v_results.json` until the T2V
target-certification policy is promoted into the main table.
