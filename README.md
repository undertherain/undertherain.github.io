# My homepage (blackbird.pw)

This is a Hugo-based static website utilizing the Blowfish theme.

## Deployment

The site is automatically built and deployed to **GitHub Pages** via a GitHub Actions workflow whenever changes are pushed to the `master` branch. 

* **Domain:** [blackbird.pw](https://blackbird.pw/) (The DNS records point to GitHub Pages IP addresses like `185.199.108.153`. Cloudflare is likely only being used for domain registration or DNS management, not hosting.)
* **Hosting:** GitHub Pages
* **Repository:** `undertherain/undertherain.github.io`
* **Workflow File:** `.github/workflows/hugo.yaml`

To deploy updates to the live site, simply commit your changes and push to the `master` branch:

```bash
git add .
git commit -m "Update site"
git push origin master
```

The GitHub Actions workflow will automatically handle building the Hugo site and publishing it to GitHub Pages.

## Local Development
To compile CSS changes locally, you can use the provided script:
```bash
./compile.sh
```

## Current direction (July 2026)
- Rediscovered the deployment workflow (GitHub Pages via Actions).