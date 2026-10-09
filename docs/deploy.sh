#!/usr/bin/env bash
# Publish docs/_build/html to the gh-pages branch.
#
#   ./deploy.sh              update the root of the site (the "latest" docs)
#   ./deploy.sh --release    also freeze a copy at /v<version>/ for the version switcher
#   ./deploy.sh --no-push    commit to gh-pages locally but do not push
#
# Run `make html` first (the Makefile's deploy targets do this for you).
# Version folders (v*/) already on gh-pages are never touched by a normal deploy.
set -euo pipefail

RELEASE=0
PUSH=1
for arg in "$@"; do
    case "$arg" in
        --release) RELEASE=1 ;;
        --no-push) PUSH=0 ;;
        *) echo "unknown option: $arg" >&2; exit 1 ;;
    esac
done

DOCS_DIR="$(cd "$(dirname "$0")" && pwd)"
HTML="$DOCS_DIR/_build/html"
WORKTREE="$DOCS_DIR/_build/gh-pages"
BASE_URL="https://natashabatalha.github.io/picaso"
REMOTE="${PICASO_DOCS_REMOTE:-origin}"

if [ ! -f "$HTML/index.html" ]; then
    echo "No build found at $HTML. Run 'make html' first." >&2
    exit 1
fi

VERSION="$(python -c "import picaso.justdoit as jdi; print(jdi.__version__)" 2>/dev/null)" || true
if [ -z "$VERSION" ]; then
    echo "Could not import picaso to read its version. Activate the docs environment first." >&2
    exit 1
fi
SHA="$(git -C "$DOCS_DIR" rev-parse --short HEAD)"
BRANCH="$(git -C "$DOCS_DIR" rev-parse --abbrev-ref HEAD)"

if [ -n "$(git -C "$DOCS_DIR" status --porcelain -- .)" ]; then
    echo "warning: docs/ has uncommitted changes; the deployed site will not match $SHA" >&2
fi

# Check out gh-pages in a separate worktree so the current branch is untouched.
git -C "$DOCS_DIR" fetch "$REMOTE" gh-pages
git -C "$DOCS_DIR" worktree remove --force "$WORKTREE" 2>/dev/null || true
git -C "$DOCS_DIR" worktree add --force -B gh-pages "$WORKTREE" FETCH_HEAD
trap 'git -C "$DOCS_DIR" worktree remove --force "$WORKTREE"' EXIT

# Root of the site = latest docs. Keep version folders and site-level files.
rsync -a --delete \
    --exclude '/.git' --exclude '/v*/' --exclude '/switcher.json' \
    --exclude '/CNAME' --exclude '/.nojekyll' --exclude '/.doctrees' \
    "$HTML/" "$WORKTREE/"

if [ "$RELEASE" -eq 1 ]; then
    rsync -a --delete --exclude '/.doctrees' "$HTML/" "$WORKTREE/v$VERSION/"
fi

touch "$WORKTREE/.nojekyll"

# Rebuild switcher.json from the version folders that exist on gh-pages.
python - "$WORKTREE" "$VERSION" "$BASE_URL" <<'EOF'
import json, os, sys
from packaging.version import InvalidVersion, Version

root, latest, base = sys.argv[1:]

def key(v):
    try:
        return Version(v)
    except InvalidVersion:
        return Version('0')

frozen = sorted(
    (d[1:] for d in os.listdir(root) if d.startswith('v') and os.path.isdir(os.path.join(root, d))),
    key=key, reverse=True,
)
entries = [{'name': f'{latest} (latest)', 'version': latest, 'url': f'{base}/', 'preferred': True}]
entries += [{'name': v, 'version': v, 'url': f'{base}/v{v}/'} for v in frozen if v != latest]

with open(os.path.join(root, 'switcher.json'), 'w') as f:
    json.dump(entries, f, indent=2)
    f.write('\n')
EOF

cd "$WORKTREE"
git add -A
if git diff --cached --quiet; then
    echo "gh-pages is already up to date."
    exit 0
fi
MSG="Deploy docs for picaso $VERSION from $BRANCH@$SHA"
[ "$RELEASE" -eq 1 ] && MSG="$MSG (release v$VERSION)"
git commit -q -m "$MSG"
echo "Committed to gh-pages: $MSG"

if [ "$PUSH" -eq 1 ]; then
    git push "$REMOTE" gh-pages
    echo "Pushed. The site updates at $BASE_URL/ within a few minutes."
else
    echo "Not pushed (--no-push). To publish later: git push $REMOTE gh-pages"
fi
