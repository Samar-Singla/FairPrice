# Cloudflare demonstration

Public URL: https://fairprice-demo.chatbot-sockscarving.workers.dev

Deployed on 17 November 2026 to the user's existing Cloudflare account with its existing Wrangler OAuth session. No password is stored in this repository. No paid hosting was enabled.

## Architecture

Browser → account-owned `fairprice-demo` Worker → temporary Cloudflare Quick Tunnel → FastAPI on this Mac, port 8001. The same server serves the built React frontend and Python ML APIs. The Worker displays an offline page if the origin fails.

The demo has an isolated `demo-data/` directory containing synthetic sample transactions, a trained neural model, checkpoints and subsequent visitor-created records. It is a shared public demonstration: visitors can generate/import data, train models and create reviews. Do not upload confidential data. The private local workspace remains in `data/` on port 8000.

This is not always-on cloud hosting. The Mac must be awake and online. The Worker URL stays constant, but a restarted Quick Tunnel gets a new origin URL. Republish the Worker after restarting the tunnel. Quick Tunnels have no uptime guarantee. No custom domain was present in the account.

## Operations

For a fresh clone, copy `deploy/wrangler.example.jsonc` to `deploy/wrangler.jsonc` and sign in to your own Wrangler account.

From the project directory:

```sh
.venv/bin/python scripts/demo_services.py status
.venv/bin/python scripts/demo_services.py start
.venv/bin/python scripts/publish_demo.py
```

`start` loads three user-session launchd jobs: local app, demo app and tunnel. They survive closing the terminal and restart crashed processes within the current login session. After logout/reboot, run `start` and `publish_demo.py` again. If the tunnel process restarts and gets a new URL, run `publish_demo.py` again. It checks the tunnel's demo health before deploying. Logs and service definitions are in `.runtime/`.

To stop all three services:

```sh
.venv/bin/python scripts/demo_services.py stop
```

The Worker remains deployed and shows an offline page. `./run.sh` remains available for a manual local-only server; do not run it on port 8000 while the local launchd job is running.

## Verification

Production frontend build, 48 backend tests and Python lint pass. Public root, health, prediction, training replay and numerical verification were tested through the Worker. Browser visual verification was unavailable in this chat.

Sources: [Cloudflare Tunnel setup](https://developers.cloudflare.com/tunnel/get-started/), [Cloudflare Containers storage limitations](https://developers.cloudflare.com/containers/faq/).
