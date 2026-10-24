import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from MUSE.muse import bridge_groups, bridge_pairs, main
from MUSE.prepare_bridge import prepare
from MUSE.generation_io import complete_target, write_config


class BridgeTests(unittest.TestCase):
    def test_grid_contains_all_independent_combinations(self):
        groups = bridge_groups()
        ddim = [g for g in groups if g["generation"] == "ddim_inversion"]
        pure = [g for g in groups if g["generation"] == "pure_noise"]
        strengths = {0.6, 0.7, 0.8, 0.9, 1.0}
        self.assertEqual(len({g["name"] for g in groups}), 30)
        self.assertEqual({(g["inversion_strength"], g["semantic_scale"]) for g in ddim},
                         {(i, a) for i in strengths for a in strengths})
        self.assertEqual({g["semantic_scale"] for g in pure}, strengths)
        self.assertTrue(all(g["adapter_strength"] == g["style_scale"] == 1.0 for g in groups))
        pairs = bridge_pairs()
        self.assertEqual(len(pairs), 25)
        by_name = {g["name"]: g for g in groups}
        for pair in pairs:
            ddim, pure = [by_name[name] for name in pair["groups"]]
            self.assertEqual(ddim["inversion_strength"], pair["inversion_strength"])
            self.assertEqual(ddim["semantic_scale"], pure["semantic_scale"])
            self.assertEqual(pure["semantic_scale"], pair["semantic_scale"])

    def test_dry_run_all_datasets_without_models_or_outputs(self):
        for dataset, task in [("office31", "A2DW"), ("officehome", "P2ACR"), ("minidomainnet", "S2CPR")]:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "not-created"
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    status = main(["bridge", "--dataset", dataset, "--task", task,
                                   "--data_root", "data", "--pretrained_model_name_or_path", "model",
                                   "--checkpoint_dir", "checkpoint", "--output_dir", str(output), "--dry_run"])
                text = stream.getvalue()
                self.assertEqual(status, 0)
                self.assertEqual(text.count("generate_ddim_inversion.py"), 25)
                self.assertEqual(text.count("generate_pure_noise.py"), 5)
                self.assertIn("prepare_bridge.py", text)
                self.assertFalse(output.exists())

    def fixture(self, root):
        plan = {"schema": 2, "dataset": "office31", "task": "A2DW", "source": "amazon", "targets": ["dslr"],
                "lora_scale": 1.0, "style_scale": 1.0,
                "num_images_per_class": 1, "groups": bridge_groups(), "bridges": bridge_pairs()}
        (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        for group in plan["groups"]:
            domain = root / "groups" / group["name"] / "dslr"
            config = {"mode": "class_conditioned" if group["generation"] == "pure_noise" else "ddim_inversion",
                      "lora_scale": 1.0, "semantic_scale": group["semantic_scale"], "style_scale": 1.0,
                      "source_domain": "amazon", "dataset": "office31", "target_domains": "dslr"}
            if group["generation"] == "ddim_inversion":
                config.update(inversion_end_ratio=group["inversion_strength"], inversion_lora_scale=1.0)
            else:
                config["num_images_per_class"] = 1
            write_config(domain.parent, config)
            rows = []
            for index, cls in enumerate(["Desk_Lamp", "bike"]):
                folder = domain / "images" / cls
                folder.mkdir(parents=True)
                (folder / "sample.png").write_bytes(b"fixture")
                rows.append({"image": f"images/{cls}/sample.png", "class_name": cls, "label": index,
                             "target_domain": "dslr", "mode": config["mode"],
                             "adapter_strength": 1.0, "semantic_scale": group["semantic_scale"], "style_scale": 1.0,
                             "inversion_strength": group["inversion_strength"]})
            (domain / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            complete_target(domain, 2, ["Desk_Lamp", "bike"])
        return plan

    def test_pool_preserves_classes_groups_and_metadata_without_copying(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.fixture(root)
            result = prepare(root)
            bridges = result["targets"]["dslr"]["bridges"]
            self.assertEqual(len(bridges), 25)
            for pair in plan["bridges"]:
                entry = bridges[pair["name"]]
                self.assertEqual(entry["images"], 4)
                output = (root / entry["path"]).parent
                rows = [json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines()]
                self.assertEqual({r["class_name"]: r["label"] for r in rows}, {"Desk_Lamp": 0, "bike": 1})
                self.assertEqual(len(rows), 4)
                self.assertEqual({r["semantic_scale"] for r in rows}, {pair["semantic_scale"]})
                self.assertEqual({r["inversion_strength"] for r in rows}, {None, pair["inversion_strength"]})
                self.assertEqual({r["generation"] for r in rows}, {"pure_noise", "ddim_inversion"})
                for row in rows:
                    self.assertTrue(os.path.samefile(output / row["file_name"], root / row["group_image"]))
            self.assertFalse((root / "uda" / "dslr" / "images").exists())
            self.assertEqual(prepare(root), result)

    def test_incomplete_or_mismatched_groups_fail_before_pooling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.fixture(root)
            sample = root / "groups" / plan["groups"][-1]["name"] / "dslr" / "images" / "bike" / "sample.png"
            sample.unlink()
            with self.assertRaises(ValueError):
                prepare(root)
            self.assertFalse((root / "uda").exists())

    def test_wrong_strength_and_changed_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.fixture(root)
            domain = root / "groups" / plan["groups"][0]["name"] / "dslr"
            config_path = domain.parent / "generation_config.json"
            config = json.loads(config_path.read_text())
            config["lora_scale"] = 0.7
            config_path.write_text(json.dumps(config), encoding="utf-8")
            complete_target(domain, 2, ["Desk_Lamp", "bike"])
            with self.assertRaises(ValueError):
                prepare(root)
            self.assertFalse((root / "uda").exists())

    def test_copy_mode_resumes_partial_merge_and_refuses_extra_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            prepare(root, "copy")
            output = root / "uda" / "dslr" / "inv_0.6_semantic_0.6"
            manifest = output / "metadata.jsonl"
            manifest.unlink()
            result = prepare(root, "copy")
            self.assertEqual(result["targets"]["dslr"]["bridges"][output.name]["images"], 4)
            (output / "images" / "bike" / "extra.png").write_bytes(b"extra")
            with self.assertRaises(ValueError):
                prepare(root, "copy")

    def test_old_pooling_plan_and_mismatched_pairing_are_rejected(self):
        for change in [lambda p: p.update(schema=1),
                       lambda p: p["bridges"][0]["groups"].__setitem__(1, "class_condition_semantic_0.7")]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan = self.fixture(root)
                change(plan)
                (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
                with self.assertRaises(ValueError):
                    prepare(root)
                self.assertFalse((root / "uda").exists())

    def test_wrong_semantic_metadata_is_rejected_even_with_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.fixture(root)
            target = root / "groups" / plan["groups"][-1]["name"] / "dslr"
            metadata = target / "metadata.jsonl"
            rows = [json.loads(line) for line in metadata.read_text().splitlines()]
            rows[0]["semantic_scale"] = 0.6
            metadata.write_text("".join(json.dumps(row) + "\n" for row in rows))
            complete_target(target, 2, ["Desk_Lamp", "bike"])
            with self.assertRaises(ValueError):
                prepare(root)
            self.assertFalse((root / "uda").exists())


if __name__ == "__main__":
    unittest.main()
