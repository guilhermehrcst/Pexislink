"""Component X v3 preflight. Every input here is a SYNTHETIC fixture
(``componentx_fixtures``); no test reads, downloads or imitates real Scania data."""

import hashlib
import io
import json
import os
import tempfile
import tracemalloc
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from componentx_fixtures import FEATURES, OPERATIONAL_HEADER, SPLIT_IDS, clean_files, csv_text, operational_rows, \
    write_dataset

from pexislink import componentx
from pexislink.cli import main
from pexislink.componentx import EXPECTED, PreflightLimits, parse_number, precision_lost, run_preflight

TRAIN_OPS = "train_operational_readouts.csv"


def ops(rows, header=OPERATIONAL_HEADER, **kwargs) -> bytes:
    return csv_text(header, rows, **kwargs).encode()


class PreflightCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = self.root / "data"
        self.out = self.root / "manifest.json"

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *extra, data=None):
        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            code = main(["dataset", "preflight", "--source", "component-x-v3", "--input-dir",
                         str(data or self.data), "--out", str(self.out), *extra])
        manifest = json.loads(self.out.read_text()) if self.out.is_file() else None
        return code, manifest, stderr.getvalue()

    def preflight(self, *, limits=None, **dataset):
        write_dataset(self.data, **dataset)
        return run_preflight(self.data, limits or PreflightLimits(), ["test"])

    def train(self, **dataset):
        return self.preflight(**dataset)["files"][TRAIN_OPS]


class CleanDatasetTests(PreflightCase):
    def test_clean_synthetic_dataset_passes_with_exit_0(self):
        write_dataset(self.data)
        code, manifest, stderr = self.run_cli()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(manifest["overall"]["status"], "pass")
        for name, report in manifest["files"].items():
            self.assertEqual(report["status"], "verified_local", name)
            expected_check = "metadata_only" if EXPECTED[name].role == "outcome" else "pass"
            self.assertEqual(report["check"], expected_check, name)
            data = (self.data / name).read_bytes()
            self.assertEqual(report["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(report["bytes"], len(data))
        train = manifest["files"][TRAIN_OPS]
        self.assertEqual(train["rows"]["data_rows"], 12)
        self.assertEqual(train["vehicles"]["distinct"], 3)
        self.assertEqual(train["features"]["count"], len(FEATURES))
        self.assertEqual(train["time_step"]["duplicate_keys"]["completeness"], "exhaustive")
        overlap = manifest["split_checks"]["operational_vehicle_overlap_between_splits"]
        self.assertEqual({v["shared_vehicle_ids"] for v in overlap.values()}, {0})

    def test_manifest_is_deterministic_and_carries_no_ids_values_or_paths(self):
        write_dataset(self.data)
        self.run_cli()
        first = self.out.read_bytes()
        self.out.unlink()
        self.run_cli()
        self.assertEqual(first, self.out.read_bytes())
        text = first.decode()
        for vid in (v for ids in SPLIT_IDS.values() for v in ids):
            self.assertNotIn(str(vid), text)
        self.assertNotIn("Cat", text)  # specification values
        self.assertNotIn(str(self.root), text)

    def test_evidence_statuses_are_explicit(self):
        manifest = self.preflight()
        self.assertEqual(manifest["license"]["status"], "publisher_claim")
        self.assertEqual(manifest["license"]["verification"], "verified_official_catalog")
        self.assertIn("researchdata.se", manifest["license"]["evidence"])
        self.assertEqual(len(componentx.EXPECTED), 9)
        self.assertIn("official version 3 catalog", componentx._CATALOG_EVIDENCE)
        for claim in manifest["source"].values():
            self.assertEqual(claim["status"], "publisher_claim")
        claims = manifest["files"][TRAIN_OPS]["publisher_claims"]
        self.assertEqual(claims["data_rows"], {"publisher_claim": 1_122_452, "verified_local": 12, "match": False})
        self.assertEqual(manifest["files"]["test_labels.csv"]["publisher_claims"]["bytes"]["match"], False)
        self.assertEqual(manifest["tool"]["pexislink_version"], componentx.__version__)

    def test_quoted_fields_crlf_and_bom_are_accepted(self):
        rows = ['"900001","1.5","100","0","0"', '"900001","2.5","101","1","2"']
        data = b"\xef\xbb\xbf" + ops(rows, newline="\r\n")
        report = self.train(overrides={TRAIN_OPS: data})
        self.assertEqual(report["check"], "pass", report)
        self.assertTrue(report["utf8_bom"])
        self.assertEqual(report["rows"]["accepted"], 2)


class MissingInputTests(PreflightCase):
    def test_missing_files_are_not_available_and_exit_4(self):
        write_dataset(self.data, omit=("test_operational_readouts.csv", "test_labels.csv"))
        code, manifest, _ = self.run_cli()
        self.assertEqual(code, 4)
        self.assertEqual(manifest["overall"]["status"], "incomplete")
        self.assertEqual(manifest["files"]["test_labels.csv"]["status"], "not_available")
        overlap = manifest["split_checks"]["operational_vehicle_overlap_between_splits"]
        self.assertEqual(overlap["train__test"], "not_checked")
        self.assertEqual(overlap["train__validation"]["status"], "verified_local")

    def test_unrecognized_entries_are_counted_not_named(self):
        write_dataset(self.data)
        (self.data / "truck_ABC123_owner_smith.csv").write_text("x")
        (self.data / "private name\x1b.csv").write_text("x")
        manifest = run_preflight(self.data, PreflightLimits(), ["test"])
        self.assertEqual(manifest["input_directory"]["unrecognized_entries"], 2)
        self.assertNotIn("smith", json.dumps(manifest))

    def test_symlink_and_directory_are_not_followed(self):
        write_dataset(self.data, omit=(TRAIN_OPS, "test_labels.csv"))
        target = self.root / "elsewhere.csv"
        target.write_bytes(clean_files()[TRAIN_OPS])
        (self.data / TRAIN_OPS).symlink_to(target)
        (self.data / "test_labels.csv").mkdir()
        manifest = run_preflight(self.data, PreflightLimits(), ["test"])
        for name in (TRAIN_OPS, "test_labels.csv"):
            self.assertEqual(manifest["files"][name]["status"], "not_checked", name)
            self.assertIsNotNone(manifest["files"][name]["fatal"])
        self.assertEqual(manifest["overall"]["status"], "fail")


class FileLevelFailureTests(PreflightCase):
    def assert_fatal(self, data: bytes, code: str, line=None, limits=None):
        report = self.preflight(overrides={TRAIN_OPS: data}, limits=limits)["files"][TRAIN_OPS]
        self.assertEqual(report["check"], "fail")
        self.assertEqual(report["fatal"]["code"], code, report["fatal"])
        if line is not None:
            self.assertEqual(report["fatal"]["line"], line)
        if code != "file_too_large":  # the whole file is still hashed for provenance
            self.assertEqual(report["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(report["bytes_read"], len(data))
        return report

    def test_header_contract(self):
        self.assert_fatal(b"", "empty_file", 1)
        self.assert_fatal(ops([], header=("vehicle_id",) + FEATURES), "missing_key_column", 1)
        self.assert_fatal(ops([], header=OPERATIONAL_HEADER + ("vehicle_id",)), "duplicate_column_name", 1)
        self.assert_fatal(ops([], header=OPERATIONAL_HEADER + ("bad name",)), "unsupported_column_name", 1)
        self.assert_fatal(ops([], header=OPERATIONAL_HEADER + ("\x1b[31m",)), "unsupported_column_name", 1)

    def test_encoding_and_syntax(self):
        good = operational_rows((900001,))
        self.assert_fatal(ops(good[:1]) + b"900001,9.5,\xff,0,0\n", "invalid_utf8", 3)
        self.assert_fatal(ops(good[:1]) + b"900001,9.5,1\x00,0,0\n", "nul_byte", 3)
        self.assert_fatal(ops(good[:1]) + b'900001,9.5,"1"x,0,0\n', "csv_syntax")
        self.assert_fatal(ops(good[:1]) + b'900001,9.5,"unterminated\n', "csv_syntax")

    def test_compressed_input_is_not_decompressed(self):
        self.assert_fatal(b"\x1f\x8b\x08\x00" + b"\x00" * 32, "unsupported_compression")
        self.assert_fatal(b"PK\x03\x04" + b"\x00" * 32, "unsupported_compression")

    def test_limits_fail_the_file_explicitly(self):
        rows = operational_rows((900001, 900002, 900003))
        self.assert_fatal(ops(rows), "row_limit_exceeded", limits=PreflightLimits(max_rows=5))
        self.assert_fatal(ops(rows), "vehicle_limit_exceeded", limits=PreflightLimits(max_vehicles=2))
        self.assert_fatal(ops(rows), "too_many_columns", 1, limits=PreflightLimits(max_columns=4))
        self.assert_fatal(ops(rows[:1]) + b"9" * 300 + b"\n", "line_too_long", 3,
                          limits=PreflightLimits(max_line_bytes=128))
        report = self.assert_fatal(ops(rows), "file_too_large", limits=PreflightLimits(max_file_bytes=64))
        self.assertEqual(report["sha256"]["status"], "not_checked")

    def test_file_changed_during_read_is_detected(self):
        write_dataset(self.data)
        real = os.fstat
        calls = {"n": 0}

        def growing(fd):
            info = real(fd)
            calls["n"] += 1
            if calls["n"] == 2:  # the size re-check after the read
                return os.stat_result((info.st_mode, 0, 0, 0, 0, 0, info.st_size + 1, 0, 0, 0))
            return info

        with mock.patch.object(componentx.os, "fstat", side_effect=growing):
            report = componentx.check_file(self.data / TRAIN_OPS, TRAIN_OPS, PreflightLimits(), set())
        self.assertEqual(report["fatal"]["code"], "file_changed_during_read")

    def test_failed_file_contributes_no_vehicle_set(self):
        manifest = self.preflight(overrides={TRAIN_OPS: b"\x1f\x8b broken"})
        overlap = manifest["split_checks"]["operational_vehicle_overlap_between_splits"]
        self.assertEqual(overlap["train__validation"], "not_checked")
        self.assertEqual(manifest["split_checks"]["operational_vs_specifications_coverage"]["train"], "not_checked")


class RowLevelTests(PreflightCase):
    def test_rejected_rows_are_reported_never_coerced(self):
        rows = operational_rows((900001,))[:1] + [
            "900001,9.5,1,2",              # line 3: wrong field count
            "9000 01,9.5,1,2,3",           # line 4: invalid vehicle id
            "900001,,1,2,3",               # line 5: missing time_step
            "900001,nan,1,2,3",            # line 6
            "900001,1e400,1,2,3",          # line 7: overflow
            "900001,1_0,1,2,3",            # line 8
            "900001,١٢,1,2,3",   # line 9: Arabic-Indic digits
        ]
        report = self.train(overrides={TRAIN_OPS: ops(rows)})
        issues = report["rows"]["issues"]
        self.assertEqual(issues["wrong_field_count"]["first_lines"], [3])
        self.assertEqual(issues["invalid_vehicle_id"]["first_lines"], [4])
        self.assertEqual(issues["time_step_missing"]["first_lines"], [5])
        self.assertEqual(issues["time_step_nonfinite"]["first_lines"], [6, 7])
        self.assertEqual(issues["time_step_non_numeric"]["first_lines"], [8, 9])
        self.assertEqual((report["rows"]["accepted"], report["rows"]["rejected"]), (1, 7))
        self.assertEqual(report["check"], "fail")

    def test_missing_and_invalid_feature_cells_are_counted_per_column(self):
        big = "9" * 70
        rows = ["900001,1,,1_000,nan", "900001,2,5, 5,inf", f"900001,3,9007199254740993,{big},1e400",
                "900001,4,6,1e-400,0"]
        report = self.train(overrides={TRAIN_OPS: ops(rows)})
        columns = report["features"]["columns"]
        self.assertEqual(report["check"], "pass")  # cell problems are reported, rows are not rejected
        self.assertEqual(columns["synthetic_counter_a"]["missing"], 1)
        self.assertEqual(columns["synthetic_counter_a"]["precision_lost_beyond_2pow53"], 1)
        self.assertEqual(columns["synthetic_bin_b_0"]["non_numeric"], 2)
        self.assertEqual(columns["synthetic_bin_b_0"]["number_too_long"], 1)
        self.assertEqual(columns["synthetic_bin_b_0"]["underflow"], 1)
        self.assertEqual(columns["synthetic_bin_b_1"]["nonfinite"], 3)
        self.assertEqual(columns["synthetic_bin_b_1"]["min"], 0.0)

    def test_time_order_is_checked_per_vehicle_without_sorting(self):
        rows = ["900001,1,1,0,0", "900001,3,2,0,0", "900001,3,3,0,0",  # duplicate key at line 4
                "900002,1,1,0,0", "900001,2,4,0,0",                    # reappears (line 6) and goes back
                "900002,5,0,0,0"]
        report = self.train(overrides={TRAIN_OPS: ops(rows)})
        time = report["time_step"]
        self.assertEqual(time["duplicate_keys"], {"adjacent_equal": 1, "completeness": "lower_bound"})
        self.assertEqual(time["backwards_steps"], 1)
        self.assertEqual(report["vehicles"]["non_contiguous_reappearances"], 2)  # 900001, then 900002
        self.assertEqual(report["rows"]["issues"]["time_step_backwards"]["first_lines"], [6])
        self.assertEqual(time["positive_delta_histogram"], {"[2^1,2^2)": 1, "[2^2,2^3)": 1})  # 1->3, 1->5
        self.assertEqual(report["rows"]["rejected"], 0)

    def test_candidate_decreases_stay_within_contiguous_blocks(self):
        rows = ["900001,1,10,0,0", "900001,2,4,0,0",   # decrease
                "900002,1,1,0,0", "900001,3,2,0,0"]    # lower than 4, but in a new block: not compared
        columns = self.train(overrides={TRAIN_OPS: ops(rows)})["features"]["columns"]
        self.assertEqual(columns["synthetic_counter_a"]["candidate_decreases"], 1)

    def test_truncated_final_line(self):
        rows = operational_rows((900001,))
        report = self.train(overrides={TRAIN_OPS: ops(rows, final_newline=False)})
        self.assertFalse(report["final_newline"])
        self.assertEqual(report["rows"]["accepted"], len(rows))
        cut = ops(rows)[:-6]  # cut mid-row
        report = self.train(overrides={TRAIN_OPS: cut})
        self.assertEqual(report["rows"]["issues"]["wrong_field_count"]["count"], 1)

    def test_multiline_quoted_record_is_flagged(self):
        rows = ["900001,1,1,0,0", '900001,2,"1\n2",0,0']
        report = self.train(overrides={TRAIN_OPS: ops(rows)})
        self.assertEqual(report["rows"]["multiline_records"], 1)

    def test_specification_duplicates_and_categories(self):
        data = csv_text(("vehicle_id", "synthetic_spec_0"),
                        ["900001,A", "900002,", "900001,B", "900003,C"]).encode()
        report = self.preflight(overrides={"train_specifications.csv": data})["files"]["train_specifications.csv"]
        self.assertEqual(report["rows"]["issues"]["duplicate_vehicle_id"]["first_lines"], [4])
        self.assertEqual(report["columns"]["synthetic_spec_0"], {"missing": 1, "distinct_values": 2})
        self.assertEqual(report["check"], "fail")


class SplitTests(PreflightCase):
    def test_shared_vehicle_is_counted_not_named(self):
        test_ops = ops(operational_rows((920001, 900002)))
        manifest = self.preflight(overrides={"test_operational_readouts.csv": test_ops})
        overlap = manifest["split_checks"]["operational_vehicle_overlap_between_splits"]
        self.assertEqual(overlap["train__test"]["shared_vehicle_ids"], 1)
        self.assertEqual(overlap["validation__test"]["shared_vehicle_ids"], 0)
        coverage = manifest["split_checks"]["operational_vs_specifications_coverage"]["test"]
        self.assertEqual((coverage["only_in_operational"], coverage["only_in_specifications"]), (1, 1))


class OutcomeIsolationTests(PreflightCase):
    def test_outcome_values_are_never_decoded(self):
        # invalid UTF-8 and nonsense after the header would fail any parser that read the values
        data = b"vehicle_id,class_label\n920001,\xff\xfe\n920002,0\n"
        report = self.preflight(overrides={"test_labels.csv": data})["files"]["test_labels.csv"]
        self.assertEqual(report["check"], "metadata_only")
        self.assertIsNone(report["fatal"])
        self.assertEqual(report["header"]["names"], ["vehicle_id", "class_label"])
        self.assertEqual(report["physical_lines_after_header"], 2)
        self.assertEqual(report["sha256"], hashlib.sha256(data).hexdigest())
        self.assertNotIn("rows", report)

    def test_outcome_row_count_comparison_handles_missing_final_newline(self):
        data = b"vehicle_id,class_label\n920001,0\n920002,0"
        report = self.preflight(overrides={"test_labels.csv": data})["files"]["test_labels.csv"]
        self.assertEqual(report["publisher_claims"]["data_rows"]["verified_local"], 2)

    def test_outcome_header_must_be_safe(self):
        report = self.preflight(overrides={"test_labels.csv": b"\xff\n1\n"})["files"]["test_labels.csv"]
        self.assertEqual(report["fatal"]["code"], "invalid_header")


class CliSemanticsTests(PreflightCase):
    def test_blocking_issue_exits_3(self):
        write_dataset(self.data, overrides={TRAIN_OPS: b""})
        code, manifest, stderr = self.run_cli()
        self.assertEqual(code, 3)
        self.assertIn(f"{TRAIN_OPS}: empty_file", manifest["overall"]["blocking_issues"])
        self.assertIn("preflight: fail", stderr)

    def test_usage_errors_exit_2_without_manifest(self):
        write_dataset(self.data)
        self.assertEqual(self.run_cli(data=self.root / "missing")[0], 2)
        self.assertEqual(self.run_cli("--max-rows", "0")[0], 2)
        self.assertFalse(self.out.exists())
        inside = self.data / "manifest.json"
        with redirect_stderr(io.StringIO()):
            code = main(["dataset", "preflight", "--source", "component-x-v3", "--input-dir", str(self.data),
                         "--out", str(inside)])
        self.assertEqual(code, 2)
        self.assertFalse(inside.exists())
        self.out.mkdir()
        self.assertEqual(self.run_cli()[0], 2)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["dataset", "preflight", "--source", "other", "--input-dir", str(self.data), "--out", "x"])
        self.assertEqual(caught.exception.code, 2)

    def test_unwritable_manifest_exits_1(self):
        write_dataset(self.data)
        from pexislink import cli
        with mock.patch.object(cli._PendingFile, "commit", side_effect=OSError(28, "No space left on device")):
            code, manifest, stderr = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("cannot write manifest", stderr)

    def test_input_files_are_not_modified(self):
        write_dataset(self.data)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.data.iterdir()}
        self.run_cli()
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.data.iterdir()}
        self.assertEqual(before, after)


class StreamingTests(PreflightCase):
    def peak_bytes(self, rows: int) -> int:
        ids = tuple(range(900000, 900050))
        per_vehicle = rows // len(ids)
        lines = []
        for vid in ids:
            lines += [f"{vid},{step},{step * 3},{step % 7},{step % 5}" for step in range(per_vehicle)]
        directory = self.root / f"big{rows}"
        directory.mkdir()
        (directory / TRAIN_OPS).write_bytes(ops(lines))
        tracemalloc.start()
        try:
            report = run_preflight(directory, PreflightLimits(), ["test"])
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(report["files"][TRAIN_OPS]["rows"]["accepted"], per_vehicle * len(ids))
        return peak

    def test_memory_does_not_grow_with_rows(self):
        small, large = self.peak_bytes(5_000), self.peak_bytes(50_000)
        self.assertLess(large, small * 1.5 + 256 * 1024, (small, large))
        self.assertLess(large, 4 * 1024 * 1024)


class ParserUnitTests(unittest.TestCase):
    def test_parse_number(self):
        self.assertEqual(parse_number("1.5e3"), (1500.0, None))
        self.assertEqual(parse_number("-0"), (-0.0, None))
        self.assertEqual(parse_number("0e-400"), (0.0, None))
        for cell, issue in (("", "missing"), ("abc", "non_numeric"), ("1_0", "non_numeric"),
                            (" 1", "non_numeric"), ("١", "non_numeric"), ("NaN", "nonfinite"),
                            ("-inf", "nonfinite"), ("1e999", "nonfinite"), ("1" * 65, "number_too_long"),
                            ("0x10", "non_numeric"), ("1e-400", "underflow"), ("0.5e-400", "underflow")):
            self.assertEqual(parse_number(cell), (None, issue), cell)

    def test_precision_lost(self):
        self.assertTrue(precision_lost("9007199254740993", 9007199254740992.0))
        self.assertFalse(precision_lost("9007199254740992", 9007199254740992.0))
        self.assertTrue(precision_lost("9007199254740993.0", 9007199254740992.0))
        self.assertTrue(precision_lost("9.007199254740993e15", 9007199254740992.0))
        self.assertFalse(precision_lost("1e16", 1e16))
        self.assertFalse(precision_lost("123.25", 123.25))

    def test_limits_are_validated(self):
        with self.assertRaises(ValueError):
            PreflightLimits(max_rows=0)
        with self.assertRaises(ValueError):
            PreflightLimits(max_vehicles=True)


class ReviewRegressionTests(PreflightCase):
    """Findings of the independent adversarial review (docs/COMPONENT-X-PREFLIGHT.md §8).
    Each test failed on the first implementation (46aacf6)."""

    def test_H1_headerless_outcome_file_fails_and_reports_no_names(self):
        manifest = self.preflight(overrides={"validation_labels.csv": b"910001,3\n910002,0\n",
                                             "train_tte.csv": b"900001,510,1\n"})
        for name in ("validation_labels.csv", "train_tte.csv"):
            report = manifest["files"][name]
            self.assertEqual(report["fatal"]["code"], "unexpected_header", name)
            self.assertNotIn("header", report)
        self.assertEqual(manifest["overall"]["status"], "fail")
        text = json.dumps(manifest)
        self.assertNotIn("910001", text)
        self.assertNotIn("900001", text)

    def test_H2_a_record_spanning_lines_is_bounded(self):
        body = b'1,"xy\n' + b'xy","xy\n' * 10_000  # one record of ~90 kB, every line short
        data = ops([]) + body
        report = self.train(overrides={TRAIN_OPS: data})
        self.assertEqual(report["fatal"]["code"], "record_too_long")
        tracemalloc.start()
        try:
            report = self.train(overrides={TRAIN_OPS: ops([]) + b'1,"xy\n' + b'xy","xy\n' * 200_000})
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(report["fatal"]["code"], "record_too_long")
        self.assertLess(peak, 4 * 1024 * 1024)

    def test_M1_specification_values_are_not_kept_verbatim(self):
        rows = [f"{900000 + i},{'v' * 60_000}{i}" for i in range(200)]  # 200 distinct 60 kB values
        data = csv_text(("vehicle_id", "synthetic_spec_0"), rows).encode()
        tracemalloc.start()
        try:
            report = self.preflight(overrides={"train_specifications.csv": data})["files"]["train_specifications.csv"]
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(report["columns"]["synthetic_spec_0"]["distinct_values"], 200)
        self.assertLess(peak, len(data) // 2)

    def test_M2_time_order_defects_block(self):
        rows = ["900001,5,1,0,0", "900001,5,2,0,0", "900001,2,3,0,0"]
        write_dataset(self.data, overrides={TRAIN_OPS: ops(rows)})
        code, manifest, _ = self.run_cli()
        self.assertEqual(code, 3)
        self.assertEqual(manifest["files"][TRAIN_OPS]["blocking"], ["time_step_order"])
        self.assertIn(f"{TRAIN_OPS}: time_step_order", manifest["overall"]["blocking_issues"])

    def test_L1_outcome_line_count(self):
        def lines(data):
            report = self.preflight(overrides={"test_labels.csv": data})["files"]["test_labels.csv"]
            return report.get("physical_lines_after_header"), report["blocking"]
        self.assertEqual(lines(b"vehicle_id,class_label"), (0, ["no_data_rows"]))
        self.assertEqual(lines(b"vehicle_id,class_label\n"), (0, ["no_data_rows"]))
        self.assertEqual(lines(b"vehicle_id,class_label\n920001,0"), (1, []))
        self.assertEqual(lines(b"vehicle_id,class_label\n920001,0\n\n\n"), (3, []))  # blank lines are lines

    def test_L2_time_step_beyond_float_precision_is_rejected(self):
        rows = ["900001,9007199254740992,1,0,0", "900001,9007199254740993,1,0,0"]
        report = self.train(overrides={TRAIN_OPS: ops(rows)})
        self.assertEqual(report["rows"]["issues"]["time_step_precision_lost"]["first_lines"], [3])
        self.assertEqual(report["time_step"]["duplicate_keys"]["adjacent_equal"], 0)

    def test_L3_overflowing_gap_has_its_own_bucket(self):
        # Through the CLI an infinite gap is now unreachable: time steps >= 2**53 must be exact
        # and at most 64 characters (L2). The bucket function stays defensive.
        self.assertEqual(componentx._delta_bucket(float("inf")), "overflow")
        self.assertEqual(componentx._delta_bucket(3.0), "[2^1,2^2)")
        big = str(2 ** 200)  # exact, 61 digits: accepted, finite gap
        report = self.train(overrides={TRAIN_OPS: ops([f"900001,-{big},1,0,0", f"900001,{big},1,0,0"])})
        self.assertEqual(report["time_step"]["positive_delta_histogram"], {"[2^201,2^202)": 1})

    def test_L5_field_limit_is_named(self):
        big = "1" * 140_000
        report = self.train(overrides={TRAIN_OPS: ops([f'900001,1,"{big}",0,0'])},
                            limits=PreflightLimits(max_line_bytes=1 << 20))
        self.assertEqual(report["fatal"]["code"], "field_too_large")

    def test_L6_empty_data_files_block(self):
        report = self.train(overrides={TRAIN_OPS: ops([])})
        self.assertEqual(report["blocking"], ["no_data_rows"])
        self.assertEqual(report["check"], "fail")

    def test_L6_claim_mismatches_are_listed(self):
        manifest = self.preflight()
        self.assertIn(f"{TRAIN_OPS}: data_rows", manifest["overall"]["publisher_claim_mismatches"])

    def test_L7_unread_entries_have_accurate_claims_and_codes(self):
        write_dataset(self.data, omit=(TRAIN_OPS,))
        (self.data / TRAIN_OPS).symlink_to(self.root / "nowhere.csv")
        report = run_preflight(self.data, PreflightLimits(), ["t"])["files"][TRAIN_OPS]
        self.assertEqual(report["fatal"]["code"], "symlink_not_followed")
        self.assertEqual(report["publisher_claims"]["note"], "file present but not read; nothing compared")

    def test_L10_read_error_fails_one_file_not_the_run(self):
        write_dataset(self.data)
        def failing(*args):
            raise OSError(5, "Input/output error")

        with mock.patch.object(componentx, "_check_operational", failing):
            manifest = run_preflight(self.data, PreflightLimits(), ["t"])
        self.assertEqual(manifest["files"][TRAIN_OPS]["fatal"]["code"], "read_error")
        self.assertEqual(manifest["files"]["test_labels.csv"]["check"], "metadata_only")


class CodexReviewTests(PreflightCase):
    """Automated review of PR #5 (c0a9724). Each test failed on c0a9724."""

    def mutate_during_read(self, change):
        original = componentx._check_operational

        def wrapper(source, limits, report, vehicles):
            original(source, limits, report, vehicles)
            if "900001" in vehicles:  # only while the training file itself is open
                change(self.data / TRAIN_OPS)

        return mock.patch.object(componentx, "_check_operational", wrapper)

    def test_same_size_rewrite_during_read_is_detected(self):
        write_dataset(self.data)

        def rewrite(path):
            data = path.read_bytes()
            path.write_bytes(data.replace(b"900001", b"900009"))  # same length, new content
            stamp = path.stat().st_mtime_ns + 5_000_000_000
            os.utime(path, ns=(stamp, stamp))

        with self.mutate_during_read(rewrite):
            report = run_preflight(self.data, PreflightLimits(), ["t"])["files"][TRAIN_OPS]
        self.assertEqual(report["fatal"]["code"], "file_changed_during_read")

    def test_path_replaced_during_read_is_detected(self):
        write_dataset(self.data)

        def replace(path):
            fresh = path.with_name("fresh.tmp")
            fresh.write_bytes(path.read_bytes())  # identical bytes, different file
            os.replace(fresh, path)

        with self.mutate_during_read(replace):
            report = run_preflight(self.data, PreflightLimits(), ["t"])["files"][TRAIN_OPS]
        self.assertEqual(report["fatal"]["code"], "file_changed_during_read")

    def test_file_growing_past_the_size_cap_is_bounded(self):
        write_dataset(self.data)
        size = len(clean_files()[TRAIN_OPS])

        def grow(path):
            with open(path, "ab") as handle:
                handle.write(b"x" * 100_000)

        with self.mutate_during_read(grow):
            report = run_preflight(self.data, PreflightLimits(max_file_bytes=size + 1000), ["t"])["files"][TRAIN_OPS]
        self.assertEqual(report["fatal"]["code"], "file_changed_during_read")
        self.assertEqual(report["sha256"]["status"], "not_checked")

    def test_outcome_file_growing_during_drain_is_bounded(self):
        """Outcome files are hashed by drain() after the header: the cap must hold there too."""
        write_dataset(self.data)
        name = "test_labels.csv"
        size = len(clean_files()[name])
        original = componentx._check_outcome

        def wrapper(source, expected, *rest):
            if expected.column_names == ("class_label",) and expected.split == "test":
                with open(self.data / name, "ab") as handle:  # grows after the size check
                    handle.write(b"920003,0\n" * 2000)
            original(source, expected, *rest)

        with mock.patch.object(componentx, "_check_outcome", wrapper):
            report = run_preflight(self.data, PreflightLimits(max_file_bytes=size + 100), ["t"])["files"][name]
        self.assertEqual(report["fatal"]["code"], "file_changed_during_read")
        self.assertEqual(report["sha256"]["status"], "not_checked")

    def test_unwritable_manifest_destination_exits_1(self):
        write_dataset(self.data)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(["dataset", "preflight", "--source", "component-x-v3", "--input-dir", str(self.data),
                         "--out", str(self.root / "missing" / "m.json")])
        self.assertEqual(code, 1)
        self.assertIn("cannot write manifest", stderr.getvalue())

    def test_symlink_rejected_without_o_nofollow(self):
        write_dataset(self.data, omit=(TRAIN_OPS,))
        target = self.root / "elsewhere.csv"
        target.write_bytes(clean_files()[TRAIN_OPS])
        (self.data / TRAIN_OPS).symlink_to(target)
        with mock.patch.object(componentx.os, "O_NOFOLLOW", 0):  # as on Windows: flag unavailable
            report = run_preflight(self.data, PreflightLimits(), ["t"])["files"][TRAIN_OPS]
        self.assertEqual(report["fatal"]["code"], "symlink_not_followed")
        self.assertNotIn("sha256", report)

    def test_o_nofollow_still_guards_a_swap_after_lstat(self):
        """Second, independent barrier: the entry becomes a symlink between lstat and open."""
        write_dataset(self.data, omit=(TRAIN_OPS,))
        target = self.root / "elsewhere.csv"
        target.write_bytes(clean_files()[TRAIN_OPS])
        (self.data / TRAIN_OPS).symlink_to(target)
        with mock.patch.object(componentx.os, "lstat", side_effect=lambda path: os.stat(target)):
            report = run_preflight(self.data, PreflightLimits(), ["t"])["files"][TRAIN_OPS]
        self.assertEqual(report["fatal"]["code"], "symlink_not_followed")

    def test_split_checks_withheld_when_rows_were_rejected(self):
        bad = ops(operational_rows((900001,)) + ["920001,not-a-number,1,0,0"])
        manifest = self.preflight(overrides={TRAIN_OPS: bad})
        overlap = manifest["split_checks"]["operational_vehicle_overlap_between_splits"]
        self.assertEqual(overlap["train__test"], "not_checked")

    def test_git_head_resolves_linked_worktree(self):
        root = self.root / "worktree"
        (root / "pkg").mkdir(parents=True)
        gitdir = self.root / "common" / "worktrees" / "wt"
        gitdir.mkdir(parents=True)
        (root / ".git").write_text(f"gitdir: {gitdir}\n")
        (gitdir / "HEAD").write_text("ref: refs/heads/feature\n")
        (gitdir / "commondir").write_text("../..\n")
        (self.root / "common" / "refs" / "heads").mkdir(parents=True)
        (self.root / "common" / "refs" / "heads" / "feature").write_text("a" * 40 + "\n")
        self.assertEqual(componentx._git_head(root / "pkg"),
                         {"status": "verified_local", "sha": "a" * 40, "ref": "refs/heads/feature"})

    def test_outcome_header_respects_max_columns(self):
        data = b"vehicle_id,class_label,synthetic_x,synthetic_y\n920001,0,1,2\n"
        manifest = self.preflight(overrides={"test_labels.csv": data}, limits=PreflightLimits(max_columns=3))
        self.assertEqual(manifest["files"]["test_labels.csv"]["fatal"]["code"], "too_many_columns")


class ReReviewTests(PreflightCase):
    """Independent re-review of the final diff (39c2677). Each test failed on 39c2677."""

    def test_symlink_loop_in_out_path_fails_cleanly_on_every_version(self):
        write_dataset(self.data)
        loop = self.root / "loop"
        loop.symlink_to(loop)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(["dataset", "preflight", "--source", "component-x-v3", "--input-dir", str(self.data),
                         "--out", str(loop / "m.json")])
        self.assertEqual(code, 1)  # the manifest cannot be written there (3.11 used to raise RuntimeError)
        self.assertIn("cannot write manifest", stderr.getvalue())
        self.assertNotIn(str(self.root), stderr.getvalue())

    def test_git_head_symlink_loop_is_not_available(self):
        root = self.root / "repo"
        (root / "pkg").mkdir(parents=True)
        (self.root / "l1").symlink_to(self.root / "l2")
        (self.root / "l2").symlink_to(self.root / "l1")
        (root / ".git").write_text("gitdir: ../l1\n")
        self.assertEqual(componentx._git_head(root / "pkg")["status"], "not_available")

    def test_outcome_with_only_blank_lines_has_no_data(self):
        write_dataset(self.data, overrides={"test_labels.csv": b"vehicle_id,class_label\n\n\r\n\n"})
        code, manifest, _ = self.run_cli()
        report = manifest["files"]["test_labels.csv"]
        self.assertEqual(code, 3)
        self.assertEqual(report["non_blank_lines_after_header"], 0)
        self.assertEqual(report["blocking"], ["no_data_rows"])

    def test_non_blank_count_spans_chunks_and_unterminated_lines(self):
        source = componentx._HashedFile(io.BytesIO(b""), 64, 1 << 30)
        source.count_content = True
        for chunk in (b"\n92", b"0001,0\n\n", b"\r\n9200", b"02,1"):  # lines split across chunks
            source._count_content(chunk)
        source.handle = io.BytesIO(b"")
        source.drain()
        self.assertEqual(source.content_lines, 2)

    def test_errors_do_not_echo_local_paths(self):
        marker = self.root / "secret_owner_ABC123"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(main(["dataset", "preflight", "--source", "component-x-v3", "--input-dir",
                                   str(marker), "--out", str(self.out)]), 2)
            write_dataset(self.data)
            self.assertEqual(main(["dataset", "preflight", "--source", "component-x-v3", "--input-dir",
                                   str(self.data), "--out", str(marker / "deep" / "m.json")]), 1)
        self.assertNotIn("secret_owner", stderr.getvalue())

    def test_growth_during_parsing_is_bounded(self):
        """The cap holds on the parse path too: the training file 'grows' past it while parsed."""
        write_dataset(self.data)
        size = len(clean_files()[TRAIN_OPS])
        real = os.fstat
        state = {"patched": False}

        def small_first(fd):
            info = real(fd)
            if not state["patched"] and info.st_size == size:  # its size when opened: under the cap
                state["patched"] = True
                values = list(info)
                values[6] = 10
                return os.stat_result(values)
            return info

        with mock.patch.object(componentx.os, "fstat", side_effect=small_first):
            report = run_preflight(self.data, PreflightLimits(max_file_bytes=size // 2), ["t"])["files"][TRAIN_OPS]
        self.assertEqual(report["fatal"]["code"], "file_changed_during_read")
        self.assertEqual(report["sha256"]["status"], "not_checked")
        self.assertNotIn("rows", report)  # parsing stopped at the cap


if __name__ == "__main__":
    unittest.main()
