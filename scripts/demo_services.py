"""Start/stop Mac demonstration services. Cloudflare origin must be republished after tunnel restart."""

import os
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
DOMAIN = f"gui/{os.getuid()}"


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    RUNTIME.mkdir(exist_ok=True)
    specs = {
        "local": (
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "uvicorn",
                "backend.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            {},
        ),
        "demo": (
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "uvicorn",
                "backend.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8001",
            ],
            {"FAIRPRICE_DATA": str(ROOT / "demo-data"), "FAIRPRICE_DEMO": "1"},
        ),
        "tunnel": (
            [
                shutil.which("cloudflared") or "cloudflared",
                "tunnel",
                "--no-autoupdate",
                "--url",
                "http://127.0.0.1:8001",
            ],
            {},
        ),
    }
    for name, (args, env) in specs.items():
        label = f"com.fairprice.{name}"
        path = RUNTIME / f"{label}.plist"
        if action == "start":
            if (
                subprocess.run(
                    ["launchctl", "print", f"{DOMAIN}/{label}"],
                    check=False,
                    capture_output=True,
                ).returncode
                == 0
            ):
                print(name, "already running")
                continue
            if name == "tunnel":
                (RUNTIME / "tunnel.log").write_text("")
            path.write_bytes(
                plistlib.dumps(
                    {
                        "Label": label,
                        "ProgramArguments": args,
                        "WorkingDirectory": str(ROOT),
                        "EnvironmentVariables": env,
                        "RunAtLoad": True,
                        "KeepAlive": True,
                        "StandardOutPath": str(RUNTIME / f"{name}.log"),
                        "StandardErrorPath": str(RUNTIME / f"{name}.log"),
                    }
                )
            )
            subprocess.run(["launchctl", "bootstrap", DOMAIN, str(path)], check=True)
            print(name, "started")
        elif action == "stop":
            subprocess.run(["launchctl", "bootout", f"{DOMAIN}/{label}"], check=False)
            for _ in range(50):
                if (
                    subprocess.run(
                        ["launchctl", "print", f"{DOMAIN}/{label}"],
                        check=False,
                        capture_output=True,
                    ).returncode
                    != 0
                ):
                    break
                time.sleep(0.1)
        elif action == "status":
            result = subprocess.run(
                ["launchctl", "print", f"{DOMAIN}/{label}"],
                check=False,
                capture_output=True,
                text=True,
            )
            print(name, "loaded" if result.returncode == 0 else "stopped")
        else:
            raise SystemExit("Use start, stop or status")


if __name__ == "__main__":
    main()
