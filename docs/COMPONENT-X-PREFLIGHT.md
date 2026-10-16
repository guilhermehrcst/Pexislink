# Scania Component X v3: source evidence and local preflight (v0.3 Stage A)

Plan: [`docs/plans/0002-component-x-v03-preflight.md`](plans/0002-component-x-v03-preflight.md).
Tracking issue: [#4](https://github.com/guilhermehrcst/Pexislink/issues/4).
Code: `src/pexislink/componentx.py`. Tests: `tests/test_componentx.py` (synthetic fixtures only).

> [!IMPORTANT]
> **No real Scania file has been downloaded, inspected or hashed in this work.** Every
> test runs on synthetic fixtures written by this project. Nothing here is a result
> about the dataset, about vehicles or about predictive value.

## 1. What the preflight is, and is not

It is a read-only, offline, streaming check of files **you** have obtained from the
publisher. It answers: are these the files the publisher describes, are they
structurally sound, and are they fit for vehicle-isolated research later?

It is not a loader for modelling. It derives no feature, trains nothing, predicts
nothing, does not interpret anonymous columns, and does not read outcome values.

## 2. Gate 0: source evidence (updated 2026-10-15)

### 2.1 Publisher catalog independently verified

The [official Researchdata.se version 3 catalog](https://researchdata.se/en/catalogue/dataset/2024-34/3)
was retrieved and inspected on 2026-10-15 outside Claude's restricted development
environment. It directly confirms:

- Dataset title: *SCANIA Component X Dataset: A Real-World Multivariate Time Series
  Dataset for Predictive Maintenance*.
- Version **3**, DOI **10.5878/bnh5-ka77**, published **2025-04-09**.
- Access **open**; license **Creative Commons Attribution 4.0 International
  (CC BY 4.0)**. Credit the five named creators, cite the versioned DOI, link
  to the license, and indicate changes in any reused material.
- **11 files** totaling approximately **1.54 GiB**, comprising **nine CSV data
  files and two documentation PDFs**.
- The v2 release introduced test labels; v3 replaced the documentation
  manuscript with the final journal article.

The **exact nine CSV filenames**, confirmed from the v3 catalog, are:


| Split | Operational | Specifications | Outcomes (do not inspect labels during Stage A) |
|---|---|---|---|
| Train | `train_operational_readouts.csv` | `train_specifications.csv` | `train_tte.csv` |
| Validation | `validation_operational_readouts.csv` | `validation_specifications.csv` | `validation_labels.csv` |
| Test | `test_operational_readouts.csv` | `test_specifications.csv` | `test_labels.csv` |

Documentation filenames: `2024_IDA_challenge_v2.pdf` and `Scania_Component_X.pdf`.

The `publisher_claim` status remains correct for source metadata: it describes
**published facts**, not an independently inspected dataset archive. The
`LICENSE.verification` field now records `verified_official_catalog`.
**No CSV payload, file checksum, CSV header, encoding or delimiter has been
independently examined.** The existing expected row/column/size numbers remain
claims from the dataset descriptor, not local validation results.

### 2.2 Prior access attempts and remaining blocker

The previous Claude development environment could not fetch the publisher:
`curl: (56) CONNECT tunnel failed, response 403` and DNS failures on the
Researchdata.se/SND/DOI hosts. Those attempts remain historical evidence of an
environment-specific network restriction, **not evidence that the publisher
website is unavailable**.

In the current working container, DNS access to
`researchdata.se`, `api.researchdata.se`, `doris.snd.se` and `snd.se`
also failed. A web research client could read the official catalog page but
could not transfer a CSV file into the preflight runtime. No official
`train_specifications.csv` bytes or SHA-256 were obtained.

**Next evidence gate:** via an approved environment with direct publisher file
access, obtain **one small non-label file**, preferably
`train_specifications.csv`, confirm its exact publisher link and local SHA-256,
and run the offline preflight. The expected manifest status for other missing
files is `incomplete`; do not create dummy "Scania" files or open any test
label values. Large operational downloads require a separate resource check.

### 2.3 What remains scientifically unverified

Real-file header schemas, encoding, per-file sizes/checksums, outcome label
boundaries and temporal feature semantics have not been established by actual
file inspection. Neither model training nor predictive claims are authorized.
The published license does not resolve these technical validity gates.


## 3. Gate 1: what the preflight checks

```bash
pexislink dataset preflight --source component-x-v3 --input-dir DIR --out MANIFEST.json \
    [--max-file-mb 2048] [--max-line-bytes 65536] [--max-rows 5000000] \
    [--max-vehicles 100000] [--max-columns 1024] [--max-dir-entries 10000] [--max-categories 256]
```

Files are recognised by exact name, top level of `DIR` only. Symbolic links are not
followed (`symlink_not_followed`): the entry is `lstat`ed and must be the file then
opened, and `O_NOFOLLOW` is used where the platform has it (two independent barriers). Directories, devices and FIFOs are opened
non-blocking only to be identified with `fstat` and are never read
(`not_a_regular_file`). Every regular file is read once, in a stream; its SHA-256
covers every byte even when parsing stops early, except when the file exceeds
`--max-file-mb` (then it is not read and `sha256` is `not_checked`) or a read error
occurs.

| Role | Files | Checks |
|---|---|---|
| operational | `*_operational_readouts.csv` | strict UTF-8 (BOM noted), no NUL, bounded line length and bounded record length (a quoted field spanning lines counts towards one record), strict CSV quoting, header (safe names, unique, `vehicle_id` and `time_step` present), field count, vehicle-id syntax, strict numbers (no `nan`/`inf`/overflow/underflow to 0/`1_000`/non-ASCII digits/blanks; nothing coerced to 0), `time_step` rejected if beyond 2^53 and not exactly representable, per vehicle in file order: backwards `time_step`, duplicate `(vehicle_id, time_step)`, non-contiguous vehicle rows, positive gap histogram (powers of two, no unit assumed); per column: missing, non-numeric, non-finite, underflow, over-long, values beyond 2^53 not exactly representable, min, max, and *candidate* decreases within a vehicle's contiguous rows |
| specifications | `*_specifications.csv` | as above for structure; duplicate `vehicle_id`; per column missing count and number of distinct values (capped; values are kept only as 16-byte digests) |
| outcome | `train_tte.csv`, `validation_labels.csv`, `test_labels.csv` | **metadata only**: header line, byte length, SHA-256, physical line count after the header and how many of those lines are not blank (a byte-level test; neither is a parsed row count; a file with no non-blank line blocks as `no_data_rows`). The first line is accepted as a header only if it names every column the publisher describes (otherwise `unexpected_header`, and no names are reported, since a headerless file's first line is data). Values are never decoded (a test feeds invalid UTF-8 after the header to prove it) |

Across files: vehicle-ID overlap between the train, validation and test operational
files, and operational-versus-specification coverage per split, as counts. Label
coverage is deliberately not computed (outcome files are not parsed).

**Duplicate-key completeness.** Without sorting or storing every key, duplicates are
found when adjacent within a vehicle. If a vehicle's `time_step` never goes
backwards, this is exhaustive (any duplicate would be a non-increase); otherwise the
count is reported as a `lower_bound`.

**Candidate decreases** are counted, not interpreted: the tool does not assume any
column is a counter, and decreases across a vehicle's non-contiguous rows are not
compared.

### Failure semantics

| Situation | Effect |
|---|---|
| A file fails structurally (encoding, CSV syntax, header, limits, compression, symlink, not a regular file, read error, changed while read: size, mtime or ctime differ, or it grew past `--max-file-mb`) | that file `check: fail` with `fatal: {code, line}`; it contributes no vehicle set; manifest still written; exit 3 |
| A row is rejected (field count, vehicle id, `time_step`) | counted with the first 5 line numbers; `blocking: rejected_rows`; exit 3 |
| A duplicate or backwards `(vehicle_id, time_step)` key | counted; `blocking: time_step_order`; exit 3 |
| A data file with no data rows | `blocking: no_data_rows`; exit 3 |
| A feature cell is missing or invalid | counted per column; the row is kept; not blocking |
| A publisher claim does not match (rows, vehicles, columns, bytes) | listed in `overall.publisher_claim_mismatches`; not blocking, because the claims are unverified |
| Expected files absent, nothing blocking | `overall: incomplete`, exit 4 |
| All nine expected files present and clean | `overall: pass`, exit 0 (a structural statement only) |
| Bad arguments, missing input directory, manifest inside the input directory, manifest path is not a regular file | exit 2, no manifest |
| Manifest cannot be written (missing or unwritable parent, full disk) | exit 1 |

Compressed files (gzip, zip, xz, bzip2, zstd) are detected by magic bytes and not
decompressed (no decompression bombs); decompress them yourself and re-run.

## 4. Gate 2: the manifest

Deterministic JSON (sorted keys; same inputs and runtime give identical bytes; only
`tool.python` differs between Python versions, checked on 3.11, 3.12 and 3.13). It
records: dataset identity, licence and evidence with explicit statuses
(`publisher_claim`, `verified_local`, `not_available`, `not_checked`), per file the
byte length, SHA-256, content type, header, checks and a comparison with each
publisher claim, split checks, limits, the command (paths replaced by placeholders),
the PexisLink version, a SHA-256 of the package sources, the git HEAD read from
files, and the Python version.

It contains **no row values, no vehicle identifiers, no local paths and no names of
unrecognised local files** (only their count). Error messages on stderr do not echo
paths either (errno text only). Time and peak memory go to stderr only, so they do
not break determinism.

## 5. Resources (synthetic data, this container: 4 CPUs, 16 GB RAM)

`scripts/componentx_synthetic_scale.py` writes a synthetic file of the described
shape (107 columns). With 200,000 rows and 4,000 vehicles (124 MiB):

| Python | Time | Peak RSS |
|---|---|---|
| 3.11.17 | 14.1 s | 19.1 MB |
| 3.12.3 | 12.9 s | 21.7 MB |
| 3.13.16 | 16.3 s | 21.6 MB |

Memory is bounded by vehicles, columns and the configured limits, not by rows or
file size (tested with `tracemalloc`, including a single record spanning 200,000
lines and specification values of 60 kB each).
Linear extrapolation to the described training file (1.12 M rows, about 1.14 GB):
roughly 70-95 s and a peak RSS in the tens of MB, single core. This is an
extrapolation from synthetic data, not a measurement on the real file.

**Download feasibility.** About 1.54 GiB of disk and no meaningful RAM are needed.
The plan forbids downloading the full bundle without explicit authorisation, so the
recommended order is: read the licence on the catalog page; download the small files
first (`test_labels.csv` is *not* needed and should not be opened: the tool would only
hash it); run the preflight; only then, with authorisation, the operational files.

## 6. How to run with legally obtained files

1. Cite the confirmed v3 CC BY 4.0 license and dataset DOI from the [official catalog](https://researchdata.se/en/catalogue/dataset/2024-34/3); record actual download URL and original file checksum.
2. Put the files in a directory outside the repository, or in `data/` (git-ignored). Do not commit them.
3. `pexislink dataset preflight --source component-x-v3 --input-dir DIR --out manifest.json`.
4. Compare `files.*.sha256` with any checksum the publisher provides (none was found yet).
5. Review `overall`, every `fatal`, and every `publisher_claims.*.match: false` before using the data.

## 7. Limitations and what is not validated

- **The real files have never been seen by this tool.** File names, headers, encoding,
  delimiter, quoting and numeric format are assumptions from secondary sources. A
  real header that differs will fail visibly (`missing_key_column`,
  `unsupported_column_name`), not silently.
- Column names must match `[A-Za-z0-9_.-]{1,64}`; others fail the file. Vehicle IDs
  must match the same pattern.
- Files are matched by exact name; a renamed file is reported as unrecognised.
- Rejecting any row, or any duplicate or backwards key, fails the file. That is
  deliberately strict; a real file with a few malformed rows or unsorted vehicles
  will need a documented decision, not a silent filter or sort.
- `time_step` values beyond 2^53 that are not exactly representable as a float are
  rejected, and numeric literals are limited to 64 characters.
- The git HEAD (also resolved through a linked worktree's `gitdir` file) does not
  reveal uncommitted changes; `package_sources_sha256` covers the code actually run.
- Change detection (fail closed, `file_changed_during_read`): size, mtime and ctime
  of the open file before and after the read, the path still naming the same file
  (device and inode) after the read, and the byte cap while streaming. Remaining
  TOCTOU gaps: a writer that restores size, mtime and ctime, or a change on a file
  system with coarse timestamps within one tick, is not detected; a swap between
  `lstat` and `open` is caught by the same-file check.
- Cross-split overlap needs every split's vehicle IDs in memory (bounded by
  `--max-vehicles`; about 34,000 described in total).
- No checksum from the publisher was found to compare against.
- Synthetic fixtures follow the *described* layout; they cannot reveal problems the
  description does not mention.

## 8. Independent adversarial review

A separate reviewer with no part in writing the code attacked the first
implementation (`46aacf6`). Every finding below was reproduced, has a regression
test in `ReviewRegressionTests` that fails on `46aacf6`, and is fixed.

| ID | Severity | Finding | Fix |
|---|---|---|---|
| H1 | High | A headerless outcome file put a vehicle ID and a label value into the manifest as "column names", and the run passed | the header must name every described column, else `unexpected_header` with no names reported |
| H2 | High | One CSV record spanning many short lines had no size bound (memory about 8× file size) | record length bounded by `--max-line-bytes` (`record_too_long`) |
| M1 | Medium | Specification category sets kept raw values (memory up to file size) | 16-byte digests |
| M2 | Medium | Duplicate or backwards keys did not block a `pass` | `blocking: time_step_order` |
| L1 | Low | Outcome line count off by one for header-only files | corrected; documented as physical lines, blank lines included |
| L2 | Low | Distinct `time_step` values beyond 2^53 became false duplicates | rejected as `time_step_precision_lost` |
| L3 | Low | An infinite gap fell in a wrong histogram bucket | `overflow` bucket (now unreachable through input, kept defensive) |
| L4 | Low | `1e-400` silently became 0; precision loss detected only for pure-digit literals | `underflow`; exact `Decimal` comparison for any notation |
| L5 | Low | A field above the csv module's limit was reported as `csv_syntax` | `field_too_large` |
| L6 | Low | Header-only data files passed; claim mismatches were not surfaced | `no_data_rows` blocks; mismatches listed in `overall` |
| L7 | Low | Unread entries had no claims block; a symlink showed only `errno 40` | accurate note; `symlink_not_followed` |
| L8 | Low | The no-process spy ran the preflight on no expected file; `os.system`-style calls were not checked | spy runs on the synthetic dataset; AST check for `os` process functions |
| L9 | Low | Two limits were not configurable | `--max-dir-entries`, `--max-categories` |
| L10 | Low | A read error aborted the whole run; Ctrl-C left a temporary file | per-file `read_error`; temporary file discarded |
| L11 | Low | Names of unrecognised local files were written to the manifest | count only |

The reviewer also confirmed: determinism across runs and Python versions, exit codes,
FIFO and symlink handling, byte-range arithmetic for the rounded publisher sizes, limit
boundaries (N allowed, N+1 fails), strict number parsing, and that the manifest cannot be
written inside the input directory (also via `..` or a symlink).

## 9. Automated review on PR #5

Codex raised 8 findings (all P2) on `c0a9724`. All were real; each fix has a test that
fails on `c0a9724` (`CodexReviewTests`).

| Finding | Fix |
|---|---|
| A same-size rewrite during the read was not detected | size, mtime and ctime compared before and after; the path must still name the file read (added after the owner's review gate) |
| A file growing after `fstat` was read without bound | byte cap enforced while streaming; reported as `file_changed_during_read` |
| An unwritable manifest parent exited 2 instead of the documented 1 | exit 1 (a non-regular destination stays a usage error, exit 2) |
| Without `O_NOFOLLOW` (Windows) a symlink would be followed | `lstat` check and same-file check before reading; `O_NOFOLLOW` kept as a second barrier, tested separately |
| A file with rejected rows still contributed its partial vehicle set | any blocking file is withheld from split checks |
| A linked worktree's `.git` file gave `git_head: not_available` | `gitdir` and `commondir` resolved |
| `--max-columns` was not applied to outcome headers | applied |
| The synthetic scale script wrote `vehicles x floor(rows/vehicles)` rows | exactly `--rows` (the measurements above used divisible values and are unaffected) |

## 10. Independent re-review of the final diff

After the owner's review gate, a fresh reviewer re-checked `39c2677`: all eight gate items
passed with reproductions (streaming cap on the parse and drain paths, same-size rewrite,
rename-over and deletion, split checks only from passing files, exit codes for early and
late manifest failures, symlinks without `O_NOFOLLOW`, exact synthetic row counts, outcome
column limit, linked worktrees with packed refs). It found four new defects, all fixed with
tests that fail on `39c2677` (`ReReviewTests`):

| Severity | Finding | Fix |
|---|---|---|
| Medium | A symlink loop in `--out` (or in a worktree `gitdir`) raised `RuntimeError` as a traceback on Python 3.11 | `realpath`, so every version reports `cannot write manifest` (exit 1) without traceback; `git_head: not_available` |
| Low | An outcome file with only blank lines after the header passed | non-blank lines counted at byte level; `no_data_rows` |
| Low | Error messages echoed local paths | errno text only |
| Low | The synthetic script created `--out` before validating its arguments | validation first |

Also from that review: the "manifest inside the input directory" guard now compares device
and inode of each parent (robust to case-insensitive or aliased paths), the parse-path
growth case has its own test, and a test assertion that could not fail was removed.
