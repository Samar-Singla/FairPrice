"""Point the account-owned Worker to the current Mac tunnel and publish it."""

import json
import re
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for attempt in range(20):
    log = (root / ".runtime/tunnel.log").read_text()
    urls = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
    if urls:
        check = subprocess.run(
            ["curl", "-fsS", "--max-time", "5", urls[-1] + "/api/health"],
            check=False,
            capture_output=True,
            text=True,
        )
        if check.returncode == 0:
            try:
                if json.loads(check.stdout).get("demo") is True:
                    break
            except json.JSONDecodeError:
                pass
    time.sleep(1)
else:
    raise SystemExit(
        "Demo tunnel is not healthy. Check .runtime/tunnel.log before publishing."
    )
config_path = root / "deploy/wrangler.jsonc"
config = json.loads(config_path.read_text())
config["vars"]["DEMO_ORIGIN"] = urls[-1]
config_path.write_text(json.dumps(config, indent=2) + "\n")
subprocess.run(
    ["wrangler", "deploy", "--config", str(config_path)], cwd=root, check=True
)
