# GlossWerk Deployment Guide

## Step 1: Domain

1. Go to [Cloudflare Registrar](https://domains.cloudflare.com/)
2. Search for `glosswerk.com` (also check `.ai` and `.io` as backups)
3. Register (~$10/year for .com)
4. Cloudflare gives you free DNS + SSL automatically

## Step 2: Landing Page (Cloudflare Pages)

The landing page is a static HTML file — no server needed.

1. Create a GitHub repo: `glosswerk-website`
2. Push `website/index.html` to the repo
3. Go to Cloudflare Dashboard → Pages → Create a project
4. Connect your GitHub repo
5. Build settings: leave blank (static site, no build step)
6. Deploy
7. In Cloudflare DNS, add a CNAME: `glosswerk.com` → your Pages URL
8. Done — landing page live at glosswerk.com with free SSL

## Step 3: Streamlit App (Railway)

Railway is the fastest way to deploy a Streamlit app ($5/month).

### 3a. Create `requirements.txt` in your glosswerk repo:

```
streamlit>=1.30
anthropic
python-docx
spacy>=3.7
```

### 3b. Create `Procfile`:

```
web: streamlit run scripts/glosswerk_app.py --server.port=$PORT --server.address=0.0.0.0
```

### 3c. Create `setup.sh` for spaCy model:

```bash
#!/bin/bash
python -m spacy download de_core_news_lg
```

### 3d. Deploy to Railway:

1. Go to [railway.app](https://railway.app) and sign up
2. New Project → Deploy from GitHub repo
3. Set environment variables:
   - `ANTHROPIC_API_KEY` = your key
   - `GLOSSWERK_DEMO` = `true`
4. Railway auto-detects Python and deploys
5. Get your app URL: `glosswerk-xxx.up.railway.app`

### 3e. Connect subdomain:

1. In Cloudflare DNS, add CNAME: `app.glosswerk.com` → your Railway URL
2. In Railway, add custom domain: `app.glosswerk.com`
3. Demo app now lives at `app.glosswerk.com`

### 3f. Update landing page login link:

Change the "Login to Demo" link in `index.html` to point to `https://app.glosswerk.com`

## Step 4: Generate Demo Invite Codes

From your local machine:

```bash
cd glosswerk/scripts
python demo_auth.py generate --company "Welocalize" --email "john@welocalize.com"
# Output: GW-A1B2C3D4

python demo_auth.py generate --company "RWS Patent" --email "jane@rws.com"
# Output: GW-E5F6G7H8

python demo_auth.py list
# Shows all codes, usage, expiry
```

Send each contact their unique code. They enter it at `app.glosswerk.com` to access the demo.

## Step 5: Cost Monitoring

### API costs
- Set a spending limit on your Anthropic account (Dashboard → Billing)
- Start with $100/month — enough for ~25-40 full patent runs
- Each patent costs ~$2-5 in API calls

### Railway
- $5/month base, scales with usage
- Set a spending limit in Railway settings

### Total monthly cost: ~$20-30

## Alternative: DigitalOcean ($6/month)

If you prefer more control:

```bash
# Create a $6/month droplet (1 vCPU, 1GB RAM)
# SSH in and:
sudo apt update && sudo apt install python3-pip python3-venv -y
git clone your-repo
cd glosswerk
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download de_core_news_lg

# Run with systemd or use screen:
GLOSSWERK_DEMO=true ANTHROPIC_API_KEY=sk-xxx \
  streamlit run scripts/glosswerk_app.py --server.port=8501

# Put nginx in front for SSL (use certbot for free certs)
```

## Local Development (No Auth)

For local testing without the auth gate:

```bash
# Don't set GLOSSWERK_DEMO, or set to false
cd glosswerk/scripts
streamlit run glosswerk_app.py
```

The auth gate only activates when `GLOSSWERK_DEMO=true`.
