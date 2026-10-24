"""Check README commands against real argument definitions without ML imports."""

import argparse
import ast
import contextlib
import copy
import io
import re
import shlex
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from MUSE import muse


ROOT = Path(__file__).resolve().parents[1]


def source_parser(script):
    tree = ast.parse((ROOT / script).read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "parse_args" in functions:
        statements = functions["parse_args"].body
    elif script == "MUSE/prepare_bridge.py":
        statements = functions["main"].body
    else:
        statements = next(
            node.body for node in tree.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
        )
    body = []
    if "str2bool" in functions:
        body.append(copy.deepcopy(functions["str2bool"]))
    for statement in statements:
        if any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "parser"
            and node.func.attr in {"parse_args", "parse_known_args"}
            for node in ast.walk(statement)
        ):
            break
        body.append(copy.deepcopy(statement))

    # ELS obtains dynamic choices from imported model/dataset registries. Use
    # their actual local exports; building those registries would import torch.
    def exports(relative):
        source = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        return next(
            ast.literal_eval(node.value) for node in source.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        )

    namespace = {
        "argparse": argparse, "Path": Path,
        "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
        "utils": types.SimpleNamespace(
            get_model_names=lambda: exports("UDA/ELS/common/vision/models/resnet.py"),
            get_dataset_names=lambda: exports("UDA/ELS/common/vision/datasets/__init__.py"),
        ),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), script, "exec"), namespace)
    return namespace["parser"]


class DocumentedCommandsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.blocks = re.findall(r"```bash\n(.*?)```", readme, flags=re.DOTALL)
        cls.commands = [
            shlex.split(command)
            for block in cls.blocks
            for command in block.replace("\\\n", " ").splitlines()
            if command.strip()
        ]

    def test_readme_commands_and_launched_children_parse(self):
        checked_scripts = set()
        for command in self.commands:
            if command[0] != "python" or command[1] == "-m":
                continue
            script = command[1]
            checked_scripts.add(script)
            with self.subTest(script=script, mode=command[2] if len(command) > 2 else None):
                self.assertTrue((ROOT / script).is_file())
                if script == "MUSE/muse.py":
                    launched = []
                    with contextlib.redirect_stdout(io.StringIO()), patch.object(
                        muse, "_run", side_effect=lambda args, dry_run: launched.append(args) or 0
                    ):
                        muse.main(command[2:] + ["--dry_run"])
                    for child in launched:
                        child_script = Path(child[1]).relative_to(ROOT).as_posix()
                        source_parser(child_script).parse_args(child[2:])
                else:
                    source_parser(script).parse_args(command[2:])
        self.assertEqual(checked_scripts, {
            "MUSE/muse.py", "MUSE/prepare_bridge.py",
            "UDA/ELS/cdan_mcc_sdat.py", "UDA/SSRT/ssrt_sdat.py",
        })

    def test_documented_minidomainnet_arguments_and_evaluation_options(self):
        fragments = [command for command in self.commands if command[0] == "--target-train-dir"]
        self.assertEqual(len(fragments), 1)
        for script in ["UDA/ELS/cdan_mcc_sdat.py", "UDA/SSRT/ssrt_sdat.py"]:
            command = next(command for command in self.commands if len(command) > 1 and command[1] == script)
            options = command[2:] + fragments[0] + ["--phase", "test"]
            if script.startswith("UDA/SSRT"):
                options += ["--checkpoint", "logs/ssrt_office31_A2D/best.pth"]
            with self.subTest(script=script):
                parsed = source_parser(script).parse_args(options)
                self.assertEqual(parsed.phase, "test")
                self.assertTrue(parsed.target_train_list.endswith("real_train.txt"))
                self.assertTrue(parsed.target_test_list.endswith("real_test.txt"))
                self.assertEqual(parsed.target_train_dir, "data/minidomainnet/domainnet/real")

    def test_training_recipe_and_resume_flags_parse(self):
        command = next(command for command in self.commands if command[:3] == ["python", "MUSE/muse.py", "train"])
        for dataset, task in [("office31", "A2DW"), ("officehome", "P2ACR"), ("minidomainnet", "S2CPR")]:
            options = list(command[2:])
            options[options.index("--dataset") + 1] = dataset
            options[options.index("--task") + 1] = task
            options[options.index("--data_root") + 1] = {
                "office31": "data/Office-31", "officehome": "data/OfficeHome",
                "minidomainnet": "data/minidomainnet/domainnet",
            }[dataset]
            options += ["--resume_from_checkpoint", "latest"]
            if dataset == "minidomainnet":
                options += ["--minidomainnet_split_dir", "data/minidomainnet/domainnet/splits_mini"]
            launched = []
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                muse, "_run", side_effect=lambda args, dry_run: launched.append(args) or 0
            ):
                muse.main(options + ["--dry_run"])
            parsed = source_parser("MUSE/train_MUSE.py").parse_args(launched[0][2:])
            with self.subTest(dataset=dataset):
                self.assertEqual(parsed.dataset, dataset)
                self.assertEqual(parsed.resume_from_checkpoint, "latest")
                self.assertEqual(parsed.learning_rate * parsed.lr_theta, .005)
                self.assertFalse(parsed.train_text_encoder)
                self.assertTrue(parsed.freeze_semantic_on_target)
                self.assertTrue(parsed.enable_adaptive_target_sampling)


if __name__ == "__main__":
    unittest.main()
