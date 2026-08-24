"""
Publish the model and the browser demo to HuggingFace.

Two repos, both created under the account named in `.env`:

  model   <user>/clothing-color-recognition          weights + model card
  space   <user>/clothing-color-recognition  (static) the browser demo

A HuggingFace repo is a git repo with git-lfs for large files. `huggingface_hub`
wraps that, so nothing here shells out to git and no lfs setup is needed.

Credentials come from `.env` at the repo root, which is gitignored:

  hf_token=hf_...
  hf_username=...

Usage
-----
  python scripts/publish_hf.py --dry-run     # print the plan, touch nothing
  python scripts/publish_hf.py --model       # weights + model card only
  python scripts/publish_hf.py --space       # demo only
  python scripts/publish_hf.py --all
  python scripts/publish_hf.py --all --private
"""
import argparse
import io
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

REPO_NAME = "clothing-color-recognition"

# (path on disk, path in the repo). The ONNX is what the Space actually loads;
# the two .pth files are for anyone wanting to continue from the checkpoints.
MODEL_FILES = [
    ("checkpoints/distilV4_student.onnx", "distilV4_student.onnx"),
    ("checkpoints/distilV4_student_fp32.pth", "distilV4_student_fp32.pth"),
    ("checkpoints/distilV4_student_int8.pth", "distilV4_student_int8.pth"),
    # The fitted Gaussians, so anyone can reproduce the labels these were
    # trained against. Without it the posterior targets cannot be recomputed.
    ("datasets/category_mixture_v4.json", "category_mixture_v4.json"),
    ("web/MODEL_CARD.md", "README.md"),
]

SPACE_FILES = [
    ("web/README.md", "README.md"),
    ("web/index.html", "index.html"),
    ("web/style.css", "style.css"),
    ("web/app.js", "app.js"),
    ("web/model.onnx", "model.onnx"),
    # Vendored so the demo never calls Google at runtime. Fetch it with:
    #   curl -o web/pose_landmarker_lite.task https://storage.googleapis.com/\
    #   mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/\
    #   pose_landmarker_lite.task
    ("web/pose_landmarker_lite.task", "pose_landmarker_lite.task"),
]


def load_env(path):
    if not os.path.exists(path):
        sys.exit("ERROR: no .env at %s. It needs hf_token and hf_username." % path)
    env = {}
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for key in ("hf_token", "hf_username"):
        if not env.get(key):
            sys.exit("ERROR: %s missing from .env" % key)
    return env


def check(files):
    """Every file must exist before anything is created remotely."""
    missing, plan = [], []
    for local, remote in files:
        p = os.path.join(PROJECT_ROOT, local)
        if os.path.exists(p):
            plan.append((p, remote, os.path.getsize(p)))
        else:
            missing.append(local)
    if missing:
        sys.exit("ERROR: missing files:\n  " + "\n  ".join(missing))
    return plan


def show(kind, repo_id, plan):
    total = sum(size for _, _, size in plan)
    print("\n  %s  %s" % (kind.upper().ljust(6), repo_id))
    for _, remote, size in plan:
        print("      %-28s %8.2f MB" % (remote, size / 1e6))
    print("      %-28s %8.2f MB total" % ("", total / 1e6))


def crlf_html(path):
    """Return `path`'s bytes with CRLF line endings, for HTML only.

    A Space's index.html gets a `window.huggingface = {...}` script injected into
    its <head> at serve time. That injector computes the insertion offset as if
    the file used CRLF, so against an LF file it lands N bytes early, where N is
    the number of newlines before <head>. With two newlines it splits the tag
    into `<hea` + script + `d>`, the browser gives up on the head, and the script
    source renders as visible text at the top of the page.

    Normalizing here rather than on disk means an editor writing LF cannot
    reintroduce it.
    """
    with io.open(path, "rb") as fh:
        data = fh.read()
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def push(api, repo_id, repo_type, plan, private, message):
    from huggingface_hub import CommitOperationAdd

    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private,
                    exist_ok=True, space_sdk="static" if repo_type == "space" else None)
    ops = []
    for local, remote, _ in plan:
        payload = crlf_html(local) if remote.endswith(".html") else local
        ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=payload))
    api.create_commit(repo_id=repo_id, repo_type=repo_type, operations=ops,
                      commit_message=message)
    kind = {"model": "", "space": "spaces/", "dataset": "datasets/"}[repo_type]
    return "https://huggingface.co/%s%s" % (kind, repo_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--space", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--private", action="store_true",
                    help="create as private; you can flip to public on the site")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--message", default="Add model and browser demo")
    args = ap.parse_args()

    do_model = args.model or args.all
    do_space = args.space or args.all
    if not (do_model or do_space):
        ap.print_help()
        return

    env = load_env(os.path.join(PROJECT_ROOT, ".env"))
    user = env["hf_username"]
    repo_id = "%s/%s" % (user, REPO_NAME)

    from huggingface_hub import HfApi
    api = HfApi(token=env["hf_token"])
    who = api.whoami()["name"]
    if who != user:
        sys.exit("ERROR: token belongs to %r but .env says %r" % (who, user))
    print("  authenticated as %s" % who)
    print("  visibility       %s" % ("private" if args.private else "PUBLIC"))

    jobs = []
    if do_model:
        jobs.append(("model", repo_id, check(MODEL_FILES)))
    if do_space:
        jobs.append(("space", repo_id, check(SPACE_FILES)))
    for kind, rid, plan in jobs:
        show(kind, rid, plan)

    if args.dry_run:
        print("\n  dry run, nothing was created or uploaded")
        return

    print()
    for kind, rid, plan in jobs:
        print("  pushing %s ..." % kind)
        url = push(api, rid, kind, plan, args.private, args.message)
        print("    %s" % url)

    if do_space:
        print("\n  The Space builds in about a minute. A static Space serves the")
        print("  files as-is, so if it looks wrong, it is the files and not a build.")


if __name__ == "__main__":
    main()
