import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from MUSE import muse, train_uda
from MUSE.prepare_bridge import prepare
import test_bridge
from test_documented_commands import source_parser


def manifest(root, dataset="office31", task="A2DW"):
    source, targets = muse.TASK_PRESETS[dataset][task]
    aliases = muse.DOMAIN_PRESETS[dataset]
    report = {"schema": 2, "dataset": dataset, "task": task, "source": aliases[source], "targets": {}}
    for target_key in targets:
        target = aliases[target_key]
        report["targets"][target] = {"bridges": {}}
        for pair in muse.bridge_pairs():
            report["targets"][target]["bridges"][pair["name"]] = {
                "images": 2, "classes": 1,
                "path": f"uda/{muse._safe_name(target)}/{pair['name']}/images",
                "inversion_strength": pair["inversion_strength"], "semantic_scale": pair["semantic_scale"],
            }
    (root / "bridge.json").write_text(json.dumps(report), encoding="utf-8")
    return report


class UdaGridTests(unittest.TestCase):
    def test_separate_jobs_paths_and_logs_for_all_three_datasets(self):
        for dataset, task, expected in [("office31", "A2DW", 50), ("officehome", "P2ACR", 75),
                                        ("minidomainnet", "S2CPR", 75)]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest(root, dataset, task)
                for method, arch in [("els", "resnet50"), ("ssrt", "vit_b_16")]:
                    args = train_uda.build_parser().parse_args([
                        "--bridge_root", str(root), "--data_root", str(root / "real"),
                        "--method", method, "--dry_run",
                    ])
                    extra = ["--arch", arch, "--seed", "0"] + (["--eps", ".1"] if method == "els" else [])
                    jobs = train_uda.commands(args, extra)
                    self.assertEqual(len(jobs), expected)
                    self.assertEqual(len({c[c.index("--source-train-dir") + 1] for c in jobs}), expected)
                    self.assertEqual(len({c[c.index("--log") + 1] for c in jobs}), expected)
                    parser = source_parser("UDA/ELS/cdan_mcc_sdat.py" if method == "els" else "UDA/SSRT/ssrt_sdat.py")
                    for job in jobs:
                        parsed = parser.parse_args(job[2:])
                        self.assertEqual(parsed.phase, "train")
                        self.assertIn(method, Path(parsed.log).parts)
                        if dataset == "minidomainnet":
                            target = Path(parsed.target_train_dir).name
                            self.assertEqual(Path(parsed.target_train_list).name, f"{target}_train.txt")
                            self.assertEqual(Path(parsed.target_test_list).name, f"{target}_test.txt")

    def test_old_manifest_missing_pair_and_overrides_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = train_uda.build_parser().parse_args([
                "--bridge_root", str(root), "--data_root", "real", "--method", "els", "--dry_run",
            ])
            report = manifest(root)
            for option in ["--source-train-dir", "--source-train-dirs", "--target-train-list", "--log", "--phase", "--source-tr"]:
                with self.assertRaises(ValueError):
                    train_uda.commands(args, [option, "wrong"])
            report["schema"] = 1
            (root / "bridge.json").write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                train_uda.commands(args, [])
            report["schema"] = 2
            report["targets"]["dslr"]["bridges"].pop("inv_0.6_semantic_0.6")
            (root / "bridge.json").write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                train_uda.commands(args, [])

    def test_preflight_requires_complete_images_before_any_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest(root)
            with patch.object(train_uda.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    train_uda.main(["--bridge_root", str(root), "--data_root", str(root / "missing"), "--method", "els"])
            run.assert_not_called()

    def test_generated_bridges_feed_25_independent_training_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_bridge.BridgeTests().fixture(root)
            prepare(root)
            real = root / "real" / "dslr" / "images"
            real.mkdir(parents=True)
            with patch.object(train_uda.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                train_uda.main([
                    "--bridge_root", str(root), "--data_root", str(root / "real"),
                    "--method", "els", "--log_root", str(root / "logs"), "--eps", "0.1",
                ])
            self.assertEqual(run.call_count, 25)
            parser = source_parser("UDA/ELS/cdan_mcc_sdat.py")
            seen = set()
            for call in run.call_args_list:
                parsed = parser.parse_args(call.args[0][2:])
                self.assertEqual(Path(parsed.target_train_dir), real)
                output = Path(parsed.source_train_dir).parent
                rows = [json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines()]
                self.assertEqual(len({row["semantic_scale"] for row in rows}), 1)
                seen.add(output.name)
            self.assertEqual(len(seen), 25)


if __name__ == "__main__":
    unittest.main()
