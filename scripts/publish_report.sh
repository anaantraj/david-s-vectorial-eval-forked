#!/usr/bin/env bash
# Publish the generated experimental notes to Wasabi object storage.
#
#   scripts/publish_report.sh [timestamp] [source-directory]
#
# Copies the selected generated report (experimental-notes/ by default) into a
# timestamped folder under reports/, then
# mirrors that folder to s3://vectorial-reports/<timestamp>/ using the `wasabi`
# AWS profile. Objects are served publicly by the bucket policy, so no per-object
# ACL is set.
#
# Content types are declared explicitly because the S3 API defaults to
# application/octet-stream, which browsers download rather than render.

set -euo pipefail

BUCKET="vectorial-reports"
ENDPOINT="https://s3.us-west-1.wasabisys.com"
PROFILE="wasabi"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${1:-$(date +%Y-%m-%dT%H-%M-%S)}"
SRC_ARG="${2:-experimental-notes}"
if [[ "$SRC_ARG" = /* ]]; then
  SRC="$SRC_ARG"
else
  SRC="$ROOT/$SRC_ARG"
fi
DEST="$ROOT/reports/$STAMP"

if [ ! -f "$SRC/index.html" ]; then
  echo "error: $SRC/index.html not found; run experimental-notes/build.py first" >&2
  exit 1
fi

echo "==> staging $DEST"
mkdir -p "$DEST"
cp "$SRC"/*.html "$SRC"/*.css "$DEST/"
for builder in "$SRC"/build.py "$SRC"/build_*.py; do
  if [ -f "$builder" ]; then cp "$builder" "$DEST/"; fi
done

# Uploaded one object at a time. `aws s3 cp --recursive` segfaults against this
# endpoint, and per-object calls also let each content type be set explicitly.
upload() {  # upload <local-file> <key-prefix>
  local file="$1" prefix="$2" name ctype
  name="$(basename "$file")"
  case "$name" in
    *.html) ctype="text/html; charset=utf-8" ;;
    *.css)  ctype="text/css; charset=utf-8" ;;
    *.py)   ctype="text/plain; charset=utf-8" ;;
    *)      ctype="application/octet-stream" ;;
  esac
  aws --profile "$PROFILE" --endpoint-url "$ENDPOINT" s3api put-object \
    --bucket "$BUCKET" --key "$prefix/$name" --body "$file" \
    --content-type "$ctype" >/dev/null
  echo "    $prefix/$name"
}

echo "==> uploading to s3://$BUCKET/$STAMP/"
for f in "$DEST"/*; do upload "$f" "$STAMP"; done

# `latest/` always mirrors the most recent run so a single link stays valid.
echo "==> refreshing s3://$BUCKET/latest/"
for f in "$DEST"/*; do upload "$f" "latest"; done

URL="$ENDPOINT/$BUCKET/$STAMP/index.html"
echo
echo "published:  $URL"
echo "latest:     $ENDPOINT/$BUCKET/latest/index.html"
echo "local copy: $DEST"
