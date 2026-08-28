# 6. Reporting and publication

The presentation site in `experimental-notes/` is generated from the run JSON by
`experimental-notes/build.py`. It is raw HTML with inline SVG and one stylesheet:
no JavaScript, no external assets, so the folder opens from disk, zips, or serves
from any static host.

```bash
.venv/bin/python experimental-notes/build.py
```

**Never hand-edit the generated HTML.** Every figure and every quoted number is
derived from the reports, which is what prevents the site from drifting away from
the data it describes.

## Which reports the site reads

```python
RUN            = ROOT / "runs" / "scaled"       # full report, includes the oracle
RUN_SYS        = ROOT / "runs" / "scaled_sys"   # systems-only report
SPLIT          = "heldout"
SPACE_SEQUENCE = [...]                           # the embedding spaces to compare
NEURAL_REF_KEY = "gemma-512"                      # reference for two-column figures
```

Two reports are needed because the oracle produces no output on cells held out for
generalisation. Including it in the common-cell intersection shrinks the
group-similarity metric from 10 cells to 6. The site therefore takes the cell base
from the systems-only report and overlays the oracle's own values, printing its
coverage as a superscript. See the common-cell rule in
[03-metrics.md](03-metrics.md).

Producing both is two evaluate calls over the same outputs, and they can run
concurrently by symlinking a second run directory at the first's outputs.

The embedding-space page reads the same two run directories scored again in each
space named in `SPACE_SEQUENCE`. The scaled study compares six: TF-IDF (the default,
which keeps the unqualified report name) and five neural spaces — EmbeddingGemma at
128/256/512/768 and MiniLM-L6 at 384 — each produced by adding
`--embedding-model <checkpoint> --embedding-dim <d>` to the evaluate call. A space
is only shown if both its reports are present, so the page degrades to whatever the
sweep produced; missing spaces are simply omitted. The two-column figures (slopes,
style gap, side-by-side table) use `NEURAL_REF_KEY` as the neural reference, chosen
as EmbeddingGemma at 512 rather than 256 because 256 is the one neural width that
does not cleanly rank the wrong-topic control worst on the triangle score.

The page never compares a value from one space against a value from another as if
they were on one scale. It compares the *ordering* of the columns, the
supported/not-resolved verdicts, and — for the triangle score specifically — where
the wrong-topic `shuffle_control` lands, which is the harness's own test (§2.2) for
whether a score measures content or register. Which rows depend on the space is
computed by comparing the reports rather than listed by hand, and the rows that do
not depend on it are the control: if any of them ever moved, the runs would differ
in more than the space.

## Figure conventions

These are requirements, not preferences. Each exists because its absence produced
a misleading figure.

**Every comparison figure shows all three reference columns** — `identity`,
`shuffle_control`, `target_sample` — each labelled with its role via `rlabel()`.
Dataset-construction figures (coverage, funnel) are exempt because they have no
system columns. Omitting a reference hides the calibration the reader needs: the
per-habit figure looked like a Claude-only failure until the oracle was plotted
and revealed a noise floor of 1.12 rather than 0.

**Axis limits derive from the data.** A hardcoded ceiling of 0.95 on the per-cell
figure pushed the largest values off the top of the plot when the run scaled to
1.33. Compute the maximum from the series being drawn and add headroom.

**Counts and ranges in captions are computed, never typed.** A caption reading
"all five scored cells" silently became false when coverage doubled.

**Text must fit the viewBox.** `text_block()` wraps a run of text to a pixel
width using the per-class glyph estimates in `_CHAR_W`. Long subtitles and notes
must go through it. A layout audit that measures rendered text width against the
viewBox catches regressions; see the audit snippet in
[00-experimentation-guide.md](00-experimentation-guide.md).

**Anchors are drawn distinctly.** `identity` and `target_sample` define one axis
of the trade-off scatter by construction, so they are drawn as rings rather than
filled circles, with an in-figure note. Marking them as ordinary systems invites
the reading that the oracle "failed" at content preservation.

Note that CSS in the stylesheet overrides SVG presentation attributes. A blanket
`.pt { stroke: #fff }` rule once rendered the anchor rings white on white.

## The resolution table

`resolution_table()` runs the interval-overlap test over the comparisons declared
in `CLAIMS` and prints a supported / not-resolved verdict for each. When a new
claim enters the prose, add it to `CLAIMS` so the verdict is computed rather than
asserted.

## Diagnostic rows in the results table

Rows whose direction is `0` have no "best" value in the usual sense, but they do
have a target, supplied as the fourth element of the `TABLE_ROWS` tuple:

- `"oracle"` — the level authentic target posts reach (`classifier.target_rate`).
- a float — a fixed level, such as ⅓ for `trm.rank_i2`.

The mark goes to the column closest to that reference. Leaving these rows unmarked
was confusing, since a reader expects every row to indicate something.

## Determinism

Rebuilding without changing the reports must produce byte-identical HTML:

```bash
.venv/bin/python experimental-notes/build.py && md5 -q experimental-notes/*.html | md5 -q
.venv/bin/python experimental-notes/build.py && md5 -q experimental-notes/*.html | md5 -q
```

Element ids are derived from a `blake2b` digest of the figure title, not from
`hash()`, which is salted per process. A non-deterministic build usually means an
unsorted dict or a stray `hash()`.

## Writing style

The audience is a technical collaborator who does not know this harness.

- Formal register, complete sentences, no split clauses or em-dash asides.
- Plain vocabulary. Say "outputs are too similar to each other" rather than
  "homogenisation". Avoid *register, dispersion, instrumentation, provenance,
  residualisation, meta-metric* in reader-facing prose.
- Interval and significance discussion belongs in the resolution table and the
  error bars, not in the running text.
- Limitations live in the limitations page.
- State supported findings plainly, without hedging.

## Publishing

```bash
./scripts/publish_report.sh            # timestamped
./scripts/publish_report.sh 2026-07-22T14-48-59   # explicit stamp
```

The script stages the built site into `reports/<timestamp>/`, uploads it to
`s3://vectorial-reports/<timestamp>/` under the `wasabi` AWS profile, and
refreshes `latest/` so a single link stays valid.

Two constraints are baked in and must be preserved:

- **`aws s3 cp --recursive` segfaults against the Wasabi endpoint.** Uploads go
  one object at a time via `s3api put-object`.
- **Content types must be set explicitly.** The S3 default of
  `application/octet-stream` makes browsers download the HTML rather than render
  it.

The bucket carries a public-read policy, so no per-object ACL is set. Verify
public access without credentials after any bucket change:

```bash
env -u AWS_PROFILE curl -s -o /dev/null -w '%{http_code} %{content_type}\n' \
  https://s3.us-west-1.wasabisys.com/vectorial-reports/latest/index.html
```

If an upload fails partway, remove the orphaned prefix rather than leaving a
partial report that renders without its stylesheet.
