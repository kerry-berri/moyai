"""Run the Moyai application (API, worker, bundled Temporal, SQLite) as one always-on Modal container.

Deploy from a checkout:  modal deploy deploy/modal/moyai_modal.py
(use a Modal client new enough for the workspace's image builder version, e.g. `pip install -U modal`).
Then drive it over HTTP: POST /api/login, POST /api/sessions, POST /api/sessions/{id}/messages.
Settings come from the Modal Secret named by MOYAI_MODAL_SECRET (default "moyai-env"):
WORKSPACE_PASSWORD, LITELLM_API_BASE, LITELLM_API_KEY, AGENT_MODEL, MODAL_TOKEN_ID,
MODAL_TOKEN_SECRET, and optionally MOYAI_AGENT_ENGINE, LITELLM_TRACE_ENDPOINT, LITELLM_TRACE_API_KEY.
"""
import os
import secrets
import subprocess
from pathlib import Path

import modal

PORT = 8787
RUNTIME_ONLY = ("PYTHONPATH", "MODAL_IS_REMOTE", "MODAL_SERVER_URL")

app = modal.App(os.environ.get("MOYAI_MODAL_CONTROL_APP", "moyai-control"))
# Inside the container this file is imported alone, so the build inputs are resolved only on deploy.
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/app")
image = modal.Image.from_dockerfile(
    ROOT / "Dockerfile",
    context_dir=ROOT,
    build_args={"MOYAI_BUILD_SHA": os.environ.get("MOYAI_BUILD_SHA", "")},
).entrypoint([])
data = modal.Volume.from_name("moyai-data", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": data},
    secrets=[modal.Secret.from_name(os.environ.get("MOYAI_MODAL_SECRET", "moyai-env"))],
    # SQLite and the bundled Temporal server have one writer: exactly one container.
    min_containers=1,
    max_containers=1,
    timeout=24 * 60 * 60,
    cpu=2,
    memory=4096,
)
@modal.concurrent(max_inputs=1000)
@modal.web_server(PORT, startup_timeout=300)
def serve() -> None:
    # Modal mounts a Volume through a symlink; the start script only accepts the real directory.
    directory = Path(os.path.realpath("/data"))
    # Without these, the app's own Modal client (which starts agent computers) is replaced by the
    # container runtime's copy and the MODAL_TOKEN_* settings are ignored.
    inherited = {name: value for name, value in os.environ.items() if name not in RUNTIME_ONLY}
    environment = {**inherited, "DATA_DIR": str(directory)}
    # The pinned Modal client cannot build with a workspace default newer than this.
    environment.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")
    if not environment.get("ENCRYPTION_KEY"):
        # Moyai creates its local key with a hard link, which a Volume does not support.
        key = directory / ".encryption-key"
        if not key.is_file():
            key.write_text(secrets.token_urlsafe(48))
            key.chmod(0o600)
            data.commit()
        environment["ENCRYPTION_KEY"] = key.read_text().strip()
    subprocess.Popen(
        ["/app/.venv/bin/python", "/app/docker_start.py", "moyai", "dev", "--host", "0.0.0.0", "--port", str(PORT)],
        env=environment,
    )
