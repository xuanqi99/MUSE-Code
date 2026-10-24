"""Generation contracts that can run without torch, model weights, or datasets."""

import argparse
import ast
import hashlib
import json
import math
import sys
import tempfile
import types
import unittest
from typing import Optional
from pathlib import Path
from unittest.mock import patch

from MUSE import generation_io, muse


ROOT = Path(__file__).resolve().parents[1]


def generation_parser(filename):
    tree = ast.parse((ROOT / "MUSE" / filename).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    function.body[-1] = ast.Return(value=ast.Name(id="parser", ctx=ast.Load()))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "argparse": argparse,
        "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    }
    exec(compile(module, filename, "exec"), namespace)
    return namespace["parse_args"]()


class Scalar(int):
    def item(self):
        return int(self)


class Timesteps(list):
    def flip(self, dimension):
        return Timesteps(reversed(self))

    def tolist(self):
        return list(self)

    def __getitem__(self, index):
        if isinstance(index, list):
            return Timesteps(value for value, keep in zip(self, index) if keep)
        result = super().__getitem__(index)
        return Timesteps(result) if isinstance(index, slice) else Scalar(result)

    def __le__(self, endpoint):
        return [value <= endpoint for value in self]


class SchedulerStub:
    def set_timesteps(self, count, device=None):
        self.num_inference_steps = count
        self.timesteps = Timesteps(reversed([index * (1000 // count) for index in range(count)]))


class ScalarTensor(float):
    device = "cpu"
    dtype = "fp32"

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self

    def __add__(self, other):
        return ScalarTensor(float(self) + float(other))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return ScalarTensor(float(self) - float(other))

    def __rsub__(self, other):
        return ScalarTensor(float(other) - float(self))

    def __mul__(self, other):
        return ScalarTensor(float(self) * float(other))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return ScalarTensor(float(self) / float(other))


class GenerationContractTest(unittest.TestCase):
    def test_all_bridge_child_commands_parse(self):
        parsers = {
            filename: generation_parser(filename)
            for filename in ("generate_pure_noise.py", "generate_ddim_inversion.py")
        }
        for dataset, task in [("office31", "A2DW"), ("officehome", "A2CPR"), ("minidomainnet", "S2CPR")]:
            commands = []
            with patch.object(muse, "_run", side_effect=lambda command, dry_run: commands.append(command) or 0):
                muse.main([
                    "bridge", "--dataset", dataset, "--task", task,
                    "--data_root", "data", "--checkpoint_dir", "checkpoint",
                    "--pretrained_model_name_or_path", "sdxl", "--dry_run",
                ])
            self.assertEqual(len(commands), 31)
            ddim_combinations = set()
            class_conditioned_scales = set()
            for command in commands[:-1]:
                filename = Path(command[1]).name
                with self.subTest(dataset=dataset, filename=filename, output=command[command.index("--output_dir") + 1]):
                    parsed = parsers[filename].parse_args(command[2:])
                    self.assertEqual(parsed.lora_scale, 1.0)
                    self.assertEqual(parsed.style_scale, 1.0)
                    self.assertIn(parsed.semantic_scale, muse.STRENGTHS)
                    self.assertIsNotNone(parsed.source_domain)
                    if filename == "generate_ddim_inversion.py":
                        self.assertEqual(parsed.inversion_lora_scale, 1.0)
                        ddim_combinations.add((parsed.inversion_end_ratio, parsed.semantic_scale))
                    else:
                        class_conditioned_scales.add(parsed.semantic_scale)
            self.assertEqual(ddim_combinations, {(i, s) for i in muse.STRENGTHS for s in muse.STRENGTHS})
            self.assertEqual(class_conditioned_scales, set(muse.STRENGTHS))

    def test_standalone_scales_default_to_one(self):
        for filename in ["generate_ddim_inversion.py", "generate_pure_noise.py"]:
            with self.subTest(filename=filename):
                parser = generation_parser(filename)
                args = parser.parse_args([
                    "--pretrained_model_name_or_path", "sdxl", "--checkpoint_dir", "checkpoint",
                    "--output_dir", "generated",
                ])
                self.assertEqual((args.lora_scale, args.semantic_scale, args.style_scale), (1.0, 1.0, 1.0))

    def test_partial_ddim_uses_matching_endpoint(self):
        source = ROOT / "MUSE/generate_ddim_inversion.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {"inversion_endpoint", "DDIMFromPriorScheduler"}]
        namespace = {"DDIMScheduler": SchedulerStub}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        for strength, endpoint, length in [(.6, 580, 30), (.7, 680, 35), (.8, 780, 40), (.9, 880, 45), (1., 980, 50)]:
            with self.subTest(strength=strength):
                _, timestep = namespace["inversion_endpoint"](SchedulerStub(), 50, strength)
                self.assertEqual(timestep, endpoint)
                scheduler = namespace["DDIMFromPriorScheduler"]()
                scheduler.prior_timestep = endpoint
                scheduler.set_timesteps(50)
                self.assertEqual(scheduler.timesteps[0], endpoint)
                self.assertEqual(scheduler.timesteps[-1], 0)
                self.assertEqual(len(scheduler.timesteps), length)
                self.assertEqual(scheduler.num_inference_steps, 50)
        scheduler.prior_timestep = 581
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(50)

    def test_inversion_matches_constant_noise_analytic_solution(self):
        source = ROOT / "MUSE/generate_ddim_inversion.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"inversion_endpoint", "ddim_invert_sdxl"}]

        class NumericScheduler(SchedulerStub):
            alphas_cumprod = [ScalarTensor(math.exp(-.005 * (index + 1))) for index in range(1000)]

            def set_timesteps(self, count, device=None):
                super().set_timesteps(count, device=device)
                self.timesteps = Timesteps(value + 1 for value in self.timesteps)

            @classmethod
            def from_config(cls, config):
                return cls()

            def scale_model_input(self, latents, timestep):
                return latents

        class UNet:
            def parameters(self):
                yield types.SimpleNamespace(dtype="fp32")

            def __call__(self, *args, **kwargs):
                return [ScalarTensor(.25)]

        namespace = {
            "torch": types.SimpleNamespace(
                Tensor=ScalarTensor, no_grad=lambda: lambda function: function,
                flip=lambda tensor, dims: tensor.flip(dims[0]),
                sqrt=lambda tensor: ScalarTensor(math.sqrt(tensor)),
            ),
            "StableDiffusionXLPipeline": object, "Optional": Optional,
            "DDIMScheduler": NumericScheduler,
            "_encode_image_to_latents": lambda *args, **kwargs: ScalarTensor(.7),
            "_get_add_time_ids": lambda *args, **kwargs: "time_ids",
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        pipe = types.SimpleNamespace(
            device="cpu", scheduler=types.SimpleNamespace(config={}), unet=UNet(),
            encode_prompt=lambda **kwargs: [types.SimpleNamespace(dtype="fp32"), None, "pooled", None],
        )
        for strength in muse.STRENGTHS:
            with self.subTest(strength=strength):
                _, timestep = namespace["inversion_endpoint"](NumericScheduler(), 50, strength)
                result = namespace["ddim_invert_sdxl"](
                    pipe, types.SimpleNamespace(shape=(1, 3, 8, 8)), "A class", 50, strength, 1., None,
                )
                first_alpha = NumericScheduler.alphas_cumprod[1]
                end_alpha = NumericScheduler.alphas_cumprod[timestep]
                clean = (.7 - math.sqrt(1 - first_alpha) * .25) / math.sqrt(first_alpha)
                expected = math.sqrt(end_alpha) * clean + math.sqrt(1 - end_alpha) * .25
                self.assertAlmostEqual(result, expected, places=12)

    def test_exact_class_names_for_three_datasets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset, domain, name in [("office31", "amazon", "back_pack"), ("officehome", "Art", "Alarm_Clock")]:
                class_dir = root / domain / ("images" if dataset == "office31" else "") / name
                class_dir.mkdir(parents=True)
                (class_dir / "fixture.jpg").write_bytes(b"fixture")
                self.assertEqual(generation_io.source_classes(root, domain, dataset), [name])
            splits = root / "splits_mini"
            splits.mkdir()
            (splits / "sketch_train.txt").write_text("sketch/Alarm_Clock/1.jpg 0\nsketch/Apple/2.jpg 1\n")
            self.assertEqual(generation_io.source_classes(root, "sketch", "minidomainnet"), ["Alarm_Clock", "Apple"])
        self.assertEqual(generation_io.prompt_class("Alarm_Clock"), "alarm clock")
        self.assertNotEqual(generation_io.stable_image_id("class/a.jpg"), generation_io.stable_image_id("class/a.png"))

    def test_config_and_completion_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            group = Path(temporary) / "group"
            generation_io.write_config(group, {"strength": .6})
            generation_io.write_config(group, {"strength": .6})
            with self.assertRaises(ValueError):
                generation_io.write_config(group, {"strength": .7})
            target = group / "target"
            generation_io.begin_target(target)
            with (target / "metadata.jsonl").open("w", encoding="utf-8") as metadata:
                generation_io.write_record(metadata, {"image": "images/Alarm_Clock/0.png"})
            with self.assertRaises(RuntimeError):
                generation_io.complete_target(target, 2, ["Alarm_Clock"])
            generation_io.complete_target(target, 1, ["Alarm_Clock"])
            completion = json.loads((target / "completion.json").read_text(encoding="utf-8"))
            self.assertEqual(completion["metadata_records"], 1)
            self.assertEqual(completion["classes"], ["Alarm_Clock"])
            self.assertEqual(completion["generation_config_sha256"], hashlib.sha256((group / "generation_config.json").read_bytes()).hexdigest())
            generation_io.begin_target(target)
            self.assertFalse((target / "completion.json").exists())
            unknown = Path(temporary) / "old"
            unknown.mkdir()
            (unknown / "old.png").write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                generation_io.write_config(unknown, {"strength": .6})

    def test_legacy_double_prefix_checkpoint(self):
        package = types.ModuleType("safetensors")
        submodule = types.ModuleType("safetensors.torch")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "pytorch_lora_weights.safetensors"
            checkpoint.write_bytes(b"fixture")
            submodule.load_file = lambda *args, **kwargs: {"unet.unet.layer.lora.down.weight": "tensor"}
            with patch.dict(sys.modules, {"safetensors": package, "safetensors.torch": submodule}):
                state = generation_io.load_lora_state(checkpoint)
        self.assertEqual(state, {"unet.layer.lora.down.weight": "tensor"})


if __name__ == "__main__":
    unittest.main()
