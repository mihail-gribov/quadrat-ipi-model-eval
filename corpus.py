"""The input rows: which payloads the bench runs, and where their text comes from.

The repository does not carry the letters. It carries the LIST -- `data/labels_money.jsonl`, one
line per payload with its Quadrat-IPI id and the demand label that selected it -- and a
fingerprint of the rows that list resolves to. The text is fetched from the dataset itself at a
pinned revision, filtered to the list, checked against the fingerprint, and cached locally
(`data/quadrat-money.jsonl`, not tracked). Two people running this a year apart therefore run the
same 420 letters, and a silently re-released dataset would be caught rather than measured.

    python3 corpus.py            # fetch (if needed), verify, report
    python3 corpus.py --refresh  # discard the cache and fetch again
"""

import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
LIST = HERE / "data" / "labels_money.jsonl"
CACHE = HERE / "data" / "quadrat-money.jsonl"
FINGERPRINT = HERE / "data" / "quadrat-money.sha256"

REPO = "mihailgribov/quadrat-ipi"
REVISION = "v1.0.2"
FILE = "data/positives.jsonl"

# The fields the bench reads. The fingerprint covers exactly these, in this order, so a change to
# a field the bench never looks at (a style annotation, say) does not invalidate the cache.
FIELDS = ("id", "text", "injection", "inj_span", "family", "action", "host_type", "locality")


def ids():
    out = []
    for line in LIST.open():
        line = line.strip()
        if line:
            out.append(json.loads(line)["id"])
    return out


def fingerprint(rows):
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["id"]):
        h.update(json.dumps([r.get(k) for k in FIELDS], ensure_ascii=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def fetch():
    """Pull the pinned revision from Hugging Face and keep only the listed rows."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is needed to fetch the corpus: pip install huggingface_hub")
    print(f"fetching {REPO}@{REVISION}:{FILE} ...", flush=True)
    path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=FILE, revision=REVISION)
    want = set(ids())
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in want:
                rows[r["id"]] = r
    missing = sorted(want - set(rows))
    if missing:
        sys.exit(f"{len(missing)} listed ids are not in {REPO}@{REVISION}: {missing[:5]} ...")
    out = [rows[i] for i in sorted(rows)]
    CACHE.parent.mkdir(exist_ok=True)
    with CACHE.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def rows(refresh=False):
    """The listed rows, fetched on first use and verified every time."""
    if refresh or not CACHE.exists():
        got = fetch()
    else:
        got = [json.loads(line) for line in CACHE.read_text().splitlines() if line.strip()]
    want = ids()
    if sorted(r["id"] for r in got) != sorted(want):
        if not refresh:
            return rows(refresh=True)
        sys.exit("the cached rows do not match the list even after a fresh fetch")
    fp = fingerprint(got)
    if FINGERPRINT.exists():
        expected = FINGERPRINT.read_text().split()[0]
        if fp != expected:
            sys.exit(f"corpus fingerprint mismatch: got {fp[:16]}..., expected {expected[:16]}... "
                     f"-- the rows behind the list have changed; do not compare against the "
                     f"published sweeps")
    return got


def main():
    refresh = "--refresh" in sys.argv[1:]
    got = rows(refresh=refresh)
    print(f"{len(got)} rows from {REPO}@{REVISION}, cached at {CACHE.relative_to(HERE)}")
    print(f"fingerprint {fingerprint(got)}")
    if not FINGERPRINT.exists():
        print(f"no {FINGERPRINT.name} yet -- write one with: python3 corpus.py --write-fingerprint")
    if "--write-fingerprint" in sys.argv[1:]:
        FINGERPRINT.write_text(fingerprint(got) + f"  {REPO}@{REVISION}:{FILE}\n")
        print(f"written {FINGERPRINT.relative_to(HERE)}")


if __name__ == "__main__":
    main()


def read_labels(path=LIST, where=""):
    """{id: label record} for the label lines that satisfy `where`.

    `where` is `field=value[|value]` pairs, comma separated: `demand=money_out,executor=agent`.
    One parser for the runner and the scorers, so the slice a run was fed is the slice its
    tables admit.
    """
    cond = []
    for part in (where or "").split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            cond.append((k.strip(), {x for x in v.split("|") if x}))
    out = {}
    for line in pathlib.Path(path).read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("id") and all(str(d.get(k)) in vals for k, vals in cond):
            out[d["id"]] = d
    return out
