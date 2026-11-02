"""Exercise the copy/paste shell interface without running any experiment."""

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ((32, 128), (64, 128), (128, 256))
MODELS = ("histogram_linear", "quantile_wasserstein", "regression", "conditional_vae")
MODEL_LABELS = ("HFM", "QWFM", "Regression", "CVAE")


def capture(path, *extra):
    shell = r'''
python() { printf '%s\0' python "$PWD" "$@"; printf '\0'; }
accelerate() { printf '%s\0' accelerate "$PWD" "$@"; printf '\0'; }
export -f python accelerate
bash "$@"
'''
    with tempfile.TemporaryDirectory() as outside:
        result = subprocess.run(["bash", "-c", shell, "capture", str(path), *extra],
                                cwd=outside, capture_output=True, timeout=10)
    if extra:
        return result
    if result.returncode:
        raise AssertionError(result.stderr.decode())
    return [item.split("\0") for item in result.stdout.decode().split("\0\0") if item]


def invocation(record):
    program, cwd, *arguments = record
    index = next(i for i, arg in enumerate(arguments) if arg.startswith("codes/"))
    return program, cwd, arguments[index], arguments[index + 1:]


def option(arguments, name):
    prefix = "--" + name
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix + "="):
            return argument.split("=", 1)[1]
        if argument == prefix:
            return arguments[index + 1]
    raise AssertionError(f"Missing option {prefix}")


def options(arguments, name):
    """Read a nargs option from the command Bash actually expanded."""
    prefix = "--" + name
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix + "="):
            values = [argument.split("=", 1)[1]]
        elif argument == prefix:
            values = []
        else:
            continue
        for value in arguments[index + 1:]:
            if value.startswith("--"):
                break
            values.append(value)
        return values
    raise AssertionError(f"Missing option {prefix}")


class PaperLauncherTests(unittest.TestCase):
    def test_every_script_runs_from_an_unrelated_directory_without_work(self):
        paths = sorted((ROOT / "scripts/paper").rglob("*.sh"))
        self.assertEqual(len(paths), 92)
        for path in paths:
            with self.subTest(path=path):
                subprocess.run(["bash", "-n", str(path)], check=True)
                records = capture(path)
                self.assertTrue(records)
                for record in records:
                    _, cwd, entry, _ = invocation(record)
                    self.assertEqual(cwd, str(ROOT))
                    self.assertTrue((ROOT / entry).is_file(), entry)
                self.assertEqual(capture(path, "unexpected").returncode, 2)
                self.assertNotIn("/nas/", path.read_text())
                self.assertNotIn("/home/", path.read_text())

    def test_training_settings_match_the_paper(self):
        for m, n in SETTINGS:
            directory = ROOT / f"scripts/paper/m{m}n{n}"
            for i, model in enumerate(MODELS):
                with self.subTest(m=m, n=n, model=model):
                    path = directory / f"{i+1:02d}_train_{model}.sh"
                    record, = capture(path)
                    _, _, _, args = invocation(record)
                    self.assertEqual(option(args, "num_measurements"), str(m))
                    self.assertEqual(option(args, "num_spectrum_bins"), str(n))
                    self.assertEqual(option(args, "num_train_iterations"), "200000")
                    self.assertEqual(float(option(args, "learning_rate")), 1e-4)
                    batch = int(option(args, "train_batch_size"))
                    accumulation = int(option(args, "gradient_accumulation_batches"))
                    self.assertEqual(batch * accumulation * 4, 2048)
                    if model == "quantile_wasserstein":
                        self.assertEqual(option(args, "num_quantiles"), str(n))
                    if model == "regression":
                        self.assertEqual(option(args, "regression_loss"), "mse_w1")
                    if model == "conditional_vae":
                        self.assertEqual(option(args, "latent_dim"), "16")

    def test_sampling_guard_and_pr_protocol(self):
        for m, n in SETTINGS:
            directory = ROOT / f"scripts/paper/m{m}n{n}"
            for i, model in enumerate(MODELS):
                path = directory / f"{20+i:02d}_evaluate_pr_{model}.sh"
                check, evaluate = capture(path)
                _, _, entry, args = invocation(check)
                self.assertEqual(entry, "codes/evaluate/check_paper_checkpoint.py")
                self.assertEqual(option(args, "required_iteration"), "200000")
                _, _, _, args = invocation(evaluate)
                for key, value in {"num_posterior_samples": "256", "reference_repeats": "20",
                                   "nearest_k": "3", "w2_num_quantiles": "2048",
                                   "model_subfolder": "unet_ema"}.items():
                    self.assertEqual(option(args, key), value)
                if model == "quantile_wasserstein":
                    self.assertEqual(option(args, "compare_decoders"), "true")
                    self.assertEqual(option(args, "eval_output_subdir"), "decoder_comparison")
                    self.assertEqual(option(args, "num_quantiles"), str(n))

    def test_independent_samples_and_cpu_reuse_route_qwfm_to_linear_cdf(self):
        for m, n in SETTINGS:
            directory = ROOT / f"scripts/paper/m{m}n{n}"
            for i, model in enumerate(MODELS):
                check, generate = capture(directory / f"{30+i:02d}_generate_{model}_samples.sh")
                self.assertEqual(invocation(check)[2], "codes/evaluate/check_paper_checkpoint.py")
                program, _, entry, args = invocation(generate)
                self.assertEqual(program, "accelerate")
                self.assertEqual(option(args, "num_conditions"), "16384")
                self.assertEqual(option(args, "num_posterior_samples"), "8")
                self.assertEqual(option(args, "eval_ode_steps"),
                                 "1,4,16,64,256" if i < 2 else "1")
                self.assertIn("if [ \"${NUM_PROCESSES}\" = \"1\" ]", (directory / f"{30+i:02d}_generate_{model}_samples.sh").read_text())
                if i == 1:
                    self.assertEqual(entry, "codes/evaluate/generate_qwfm_decoder_comparison.py")
                for stage, suffix in [(40, "scores"), (50, "forward")]:
                    record, = capture(directory / f"{stage+i:02d}_evaluate_{model}_{suffix}.sh")
                    program, _, entry, args = invocation(record)
                    self.assertEqual(program, "python")
                    run = option(args, "run_dir")
                    self.assertEqual("/linear_cdf/" in run, i == 1)
                    self.assertNotIn("hard_bin", run)
                    self.assertEqual(option(args, "sample_count"), "8")
                    if stage == 40:
                        index = args.index("--ground_metrics")
                        self.assertEqual(args[index + 1:index + 3], ["l2", "w2"])
                        self.assertIn("--skip_plots", args)
                    else:
                        self.assertIn("--recompute", args)

    def test_command_lists_cover_every_paper_script_once(self):
        commands = []
        names = [f"run_examples_m{m}n{n}.txt" for m, n in SETTINGS]
        names.append("run_examples_comparisons.txt")
        for name in names:
            commands.extend(line.removeprefix("bash ") for line in (ROOT / name).read_text().splitlines()
                            if line.startswith("bash "))
        expected = {str(p.relative_to(ROOT)) for p in (ROOT / "scripts/paper").rglob("*.sh")}
        self.assertEqual(len(commands), len(set(commands)))
        self.assertEqual(set(commands), expected)

    def test_each_model_pipeline_ends_with_its_own_visualization(self):
        for m, n in SETTINGS:
            setting = f"m{m}n{n}"
            names = [Path(line.removeprefix("bash ")).name
                     for line in (ROOT / f"run_examples_{setting}.txt").read_text().splitlines()
                     if line.startswith("bash ")]
            expected = ["00_prepare_test_conditions.sh", "10_build_reference_run0.sh",
                        "11_build_reference_run1.sh", "12_build_reference_run2.sh", "13_audit_references.sh"]
            for i, model in enumerate(MODELS):
                expected.extend((f"{1+i:02d}_train_{model}.sh",
                                 f"{30+i}_generate_{model}_samples.sh",
                                 f"{40+i}_evaluate_{model}_scores.sh",
                                 f"{50+i}_evaluate_{model}_forward.sh",
                                 f"{20+i}_evaluate_pr_{model}.sh",
                                 f"{60+i}_visualize_{model}.sh"))
            expected.append("70_compare_models.sh")
            self.assertEqual(names, expected, setting)

    def test_visualization_tiers_select_only_their_own_inputs_and_distinct_outputs(self):
        destinations = []
        for m, n in SETTINGS:
            setting = f"m{m}n{n}"
            directory = ROOT / f"scripts/paper/{setting}"
            scripts = [(directory / f"{60+i}_visualize_{model}.sh", [MODEL_LABELS[i]])
                       for i, model in enumerate(MODELS)]
            scripts.append((directory / "70_compare_models.sh", list(MODEL_LABELS)))
            for path, selected in scripts:
                with self.subTest(path=path):
                    metric, reconstruction = capture(path)
                    program, _, entry, args = invocation(metric)
                    self.assertEqual(program, "python")
                    self.assertEqual(entry, "codes/evaluate/visualize_results.py")
                    self.assertEqual(option(args, "experiments_root"), "outputs")
                    self.assertEqual(options(args, "settings"), [setting])
                    self.assertEqual(options(args, "models"), selected)
                    self.assertEqual(options(args, "nfes"), ["1", "4", "16", "64", "256"])
                    metric_output = Path(option(args, "output_dir"))
                    self.assertTrue(metric_output.is_relative_to(Path("outputs") / setting))
                    destinations.append(metric_output)

                    program, _, entry, args = invocation(reconstruction)
                    self.assertEqual(program, "python")
                    self.assertEqual(entry, "codes/evaluate/visualize_reconstructions.py")
                    self.assertEqual(option(args, "experiment_dir"), f"outputs/{setting}")
                    self.assertEqual(options(args, "models"), selected)
                    self.assertEqual(option(args, "num_quantiles"), str(n))
                    self.assertEqual(option(args, "nfe"), "256")
                    self.assertEqual(option(args, "sample_count"), "8")
                    self.assertEqual(option(args, "num_conditions"), "16384")
                    reconstruction_output = Path(option(args, "output_dir"))
                    self.assertTrue(reconstruction_output.is_relative_to(Path("outputs") / setting))
                    destinations.append(reconstruction_output)

        record, = capture(ROOT / "scripts/paper/90_export_results.sh")
        program, _, entry, args = invocation(record)
        self.assertEqual(program, "python")
        self.assertEqual(entry, "codes/evaluate/visualize_results.py")
        self.assertEqual(option(args, "experiments_root"), "outputs")
        self.assertEqual(options(args, "settings"), [f"m{m}n{n}" for m, n in SETTINGS])
        self.assertEqual(options(args, "models"), list(MODEL_LABELS))
        self.assertEqual(options(args, "nfes"), ["1", "4", "16", "64", "256"])
        destinations.append(Path(option(args, "output_dir")))
        self.assertEqual(len(destinations), len(set(destinations)), "Visualization tiers must not overwrite one another.")


if __name__ == "__main__":
    unittest.main()
