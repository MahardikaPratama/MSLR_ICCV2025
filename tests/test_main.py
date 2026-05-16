import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

import main


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, gloss_dict=None, **kwargs):
        self.gloss_dict = gloss_dict
        self.kwargs = kwargs
        self.items = [0, 1, 2]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    @staticmethod
    def collate_fn(batch):
        return batch


class DummyModel:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.loaded_state = None
        self.to_device = None
        self.cuda_called = False

    def to(self, device):
        self.to_device = device
        return self

    def cuda(self):
        self.cuda_called = True
        return self

    def load_state_dict(self, state_dict, strict=False):
        self.loaded_state = (state_dict, strict)

    def state_dict(self):
        return {"weights": 1}


class DummyScheduler:
    def state_dict(self):
        return {"scheduler": 1}


class DummyOptimizer:
    def __init__(self, model, optimizer_args):
        self.model = model
        self.optimizer_args = optimizer_args
        self.scheduler = DummyScheduler()

    def state_dict(self):
        return {"optimizer": 1}


class MainProcessorTestCase(unittest.TestCase):
    def setUp(self):
        self.repo_root = os.path.dirname(os.path.dirname(__file__))
        self.old_cwd = os.getcwd()
        os.chdir(self.repo_root)
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tempdir.cleanup()

    def make_arg(self, **overrides):
        arg = {
            "work_dir": self.tempdir.name,
            "random_fix": False,
            "random_seed": 123,
            "print_log": False,
            "log_interval": 1,
            "dataset": "bisindo",
            "model": "DummyModel",
            "feeder": "DummyDataset",
            "device": "cpu",
            "model_args": {},
            "optimizer_args": {"start_epoch": 0},
            "batch_size": 2,
            "test_batch_size": 3,
            "num_worker": 0,
            "load_weights": None,
            "load_checkpoints": None,
            "ignore_weights": [],
            "feeder_args": {},
            "train_args": {},
            "evaluate_tool": "mock_eval",
            "phase": "train",
            "save_interval": 1,
            "eval_interval": 1,
            "num_epoch": 2,
        }
        arg.update(overrides)
        return SimpleNamespace(**arg)

    def make_processor(self, **overrides):
        processor = object.__new__(main.SLRProcessor)
        processor.arg = self.make_arg(**overrides)
        processor.dataset = {}
        processor.data_loader = {}
        processor.device = mock.MagicMock()
        processor.device.output_device = "cpu"
        processor.recoder = mock.MagicMock()
        processor.recoder.print_log = mock.MagicMock()
        processor.rng = mock.MagicMock()
        processor.rng.save_rng_state.return_value = {"rng": 1}
        processor.gloss_dict = {"gloss2id": {"HELLO": {"index": 0}, "WORLD": {"index": 1}}}
        processor.model = mock.MagicMock()
        processor.model.state_dict.return_value = {"model": 1}
        processor.optimizer = mock.MagicMock()
        processor.optimizer.state_dict.return_value = {"optimizer": 1}
        processor.optimizer.scheduler = mock.MagicMock()
        processor.optimizer.scheduler.state_dict.return_value = {"scheduler": 1}
        processor.best_dev_wer = 1000
        processor.tasks = processor.arg.dataset[-2:]
        return processor

    def test_save_arg_writes_yaml_config(self):
        processor = self.make_processor()
        processor.save_arg()

        config_path = os.path.join(self.tempdir.name, "config.yaml")
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r", encoding="utf-8") as f:
            saved = yaml.safe_load(f)
        self.assertEqual(saved["dataset"], "bisindo")
        self.assertEqual(saved["work_dir"], self.tempdir.name)

    def test_load_dataset_info_reads_dataset_config(self):
        processor = self.make_processor()
        processor.load_dataset_info()

        self.assertEqual(processor.arg.dataset_info["dict_path"], "./datasets/mslr2025/sd_gloss_dict.json")
        self.assertEqual(processor.arg.dataset_info["evaluation_prefix"], "mslr-sd-groundtruth")

    def test_build_module_creates_configured_model(self):
        processor = self.make_processor()
        with mock.patch.object(main.slr_network, "DummyModel", DummyModel, create=True):
            model = processor.build_module({"hidden_dim": 32})

        self.assertIsInstance(model, DummyModel)
        self.assertEqual(model.init_kwargs["hidden_dim"], 32)
        self.assertEqual(model.init_kwargs["gloss_dict"], processor.gloss_dict)

    def test_model_to_device_moves_model_and_calls_cuda(self):
        processor = self.make_processor()
        model = DummyModel()

        returned = processor.model_to_device(model)

        self.assertIs(returned, model)
        self.assertEqual(model.to_device, "cpu")
        self.assertTrue(model.cuda_called)

    def test_load_model_weights_removes_ignored_keys(self):
        processor = self.make_processor(ignore_weights=["remove_me"])
        model = DummyModel()
        weight_path = os.path.join(self.tempdir.name, "weights.pt")
        torch.save({"model_state_dict": {"keep": 1, "remove_me": 2}}, weight_path)

        processor.load_model_weights(model, weight_path)

        self.assertEqual(model.loaded_state, ({"keep": 1}, False))

    def test_build_dataloader_uses_mode_specific_batch_settings(self):
        processor = self.make_processor()
        processor.feeder = DummyDataset
        dataset = DummyDataset()

        train_loader = processor.build_dataloader(dataset, "train", True)
        test_loader = processor.build_dataloader(dataset, "dev", False)

        self.assertEqual(train_loader.batch_size, 2)
        self.assertTrue(train_loader.drop_last)
        self.assertEqual(test_loader.batch_size, 3)
        self.assertFalse(test_loader.drop_last)
        self.assertIs(train_loader.collate_fn, DummyDataset.collate_fn)
        self.assertIs(test_loader.collate_fn, DummyDataset.collate_fn)

    def test_load_data_builds_all_splits(self):
        processor = self.make_processor()
        with mock.patch.object(main.datasets, "DummyDataset", DummyDataset, create=True):
            processor.arg.feeder = "DummyDataset"
            processor.load_data()

        self.assertEqual(set(processor.dataset.keys()), {"train", "dev", "test"})
        self.assertEqual(set(processor.data_loader.keys()), {"train", "dev", "test"})
        self.assertEqual(processor.dataset["train"].kwargs["mode"], "train")
        self.assertTrue(processor.dataset["train"].kwargs["transform_mode"])
        self.assertFalse(processor.dataset["dev"].kwargs["transform_mode"])
        self.assertEqual(processor.dataset["train"].gloss_dict, {"HELLO": 0, "WORLD": 1})
        self.assertEqual(processor.data_loader["train"].batch_size, 2)
        self.assertEqual(processor.data_loader["test"].batch_size, 3)

    def test_judge_save_eval_returns_expected_flags(self):
        processor = self.make_processor(save_interval=2, eval_interval=3, num_epoch=8)

        save_model, eval_model = processor.judge_save_eval(1)
        self.assertFalse(save_model)
        self.assertFalse(eval_model)

        save_model, eval_model = processor.judge_save_eval(4)
        self.assertTrue(save_model)
        self.assertFalse(eval_model)

        save_model, eval_model = processor.judge_save_eval(6)
        self.assertTrue(save_model)
        self.assertTrue(eval_model)

    def test_save_model_serializes_expected_state(self):
        processor = self.make_processor()
        save_path = os.path.join(self.tempdir.name, "model.pt")
        with mock.patch.object(main.torch, "save") as mock_save:
            processor.save_model(7, save_path)

        mock_save.assert_called_once()
        saved_obj, saved_path = mock_save.call_args.args
        self.assertEqual(saved_path, save_path)
        self.assertEqual(saved_obj["epoch"], 7)
        self.assertEqual(saved_obj["model_state_dict"], {"model": 1})
        self.assertEqual(saved_obj["optimizer_state_dict"], {"optimizer": 1})
        self.assertEqual(saved_obj["scheduler_state_dict"], {"scheduler": 1})
        self.assertEqual(saved_obj["rng_state"], {"rng": 1})

    def test_custom_save_model_updates_best_and_current_files(self):
        processor = self.make_processor()
        save_dir = self.tempdir.name
        open(os.path.join(save_dir, "best_dev_10.00_epoch1_model.pt"), "a", encoding="utf-8").close()
        open(os.path.join(save_dir, "cur_dev_10.00_epoch1_model.pt"), "a", encoding="utf-8").close()

        with mock.patch.object(main.os, "system") as mock_system, mock.patch.object(processor, "save_model") as mock_save_model:
            processor.custom_save_model(5.0, 2, save_dir)

        self.assertEqual(processor.best_dev_wer, 5.0)
        self.assertEqual(mock_save_model.call_count, 2)
        saved_paths = [call.args[1] for call in mock_save_model.call_args_list]
        self.assertTrue(any("cur_dev_05.00_epoch2_model.pt" in path for path in saved_paths))
        self.assertTrue(any("best_dev_05.00_epoch2_model.pt" in path for path in saved_paths))
        self.assertEqual(mock_system.call_count, 2)

    def test_custom_save_model_creates_best_and_current_when_empty(self):
        processor = self.make_processor()
        save_dir = os.path.join(self.tempdir.name, "empty_save_dir")
        os.makedirs(save_dir)

        with mock.patch.object(main.os, "system") as mock_system, mock.patch.object(processor, "save_model") as mock_save_model:
            processor.custom_save_model(12.5, 3, save_dir)

        self.assertEqual(processor.best_dev_wer, 12.5)
        self.assertEqual(mock_save_model.call_count, 2)
        self.assertEqual(mock_system.call_count, 0)
        saved_paths = [call.args[1] for call in mock_save_model.call_args_list]
        self.assertTrue(any("cur_dev_12.50_epoch3_model.pt" in path for path in saved_paths))
        self.assertTrue(any("best_dev_12.50_epoch3_model.pt" in path for path in saved_paths))

    def test_loading_initializes_model_optimizer_and_data(self):
        processor = self.make_processor(load_weights="weights.pt")
        dummy_model = DummyModel()

        with mock.patch.object(processor, "build_module", return_value=dummy_model) as mock_build_module, \
            mock.patch.object(main.utils, "Optimizer", DummyOptimizer), \
            mock.patch.object(processor, "load_model_weights") as mock_load_weights, \
            mock.patch.object(processor, "model_to_device", return_value=dummy_model) as mock_model_to_device, \
            mock.patch.object(processor, "load_data") as mock_load_data:
            model, optimizer = processor.loading()

        self.assertIs(model, dummy_model)
        self.assertIsInstance(optimizer, DummyOptimizer)
        mock_build_module.assert_called_once_with(processor.arg.model_args)
        mock_load_weights.assert_called_once_with(dummy_model, "weights.pt")
        mock_model_to_device.assert_called_once_with(dummy_model)
        mock_load_data.assert_called_once()

    def test_train_runs_evaluation_and_save_cycle(self):
        processor = self.make_processor()
        processor.data_loader = {"train": "train_loader"}
        with mock.patch.object(main, "seq_train") as mock_seq_train, \
            mock.patch.object(processor, "test", return_value=23.4) as mock_test, \
            mock.patch.object(processor, "custom_save_model") as mock_custom_save_model:
            processor.train()

        self.assertEqual(mock_seq_train.call_count, 2)
        self.assertEqual(mock_test.call_count, 2)
        mock_custom_save_model.assert_called_once_with(23.4, 1, processor.arg.work_dir)
        self.assertTrue(processor.recoder.print_log.called)

    def test_test_delegates_to_seq_eval(self):
        processor = self.make_processor()
        processor.data_loader = {"dev": "dev_loader"}
        with mock.patch.object(main, "seq_eval", return_value=88.5) as mock_seq_eval:
            result = processor.test("dev", 9)

        self.assertEqual(result, 88.5)
        mock_seq_eval.assert_called_once_with(
            processor.arg,
            "dev_loader",
            processor.model,
            processor.device,
            "dev",
            9,
            processor.arg.work_dir,
            processor.recoder,
            processor.tasks,
            processor.arg.evaluate_tool,
        )

    def test_start_dispatches_train_and_test_modes(self):
        processor = self.make_processor(phase="train")
        with mock.patch.object(processor, "train") as mock_train, \
            mock.patch.object(processor, "test") as mock_test:
            processor.start()

        mock_train.assert_called_once()
        mock_test.assert_not_called()

        processor = self.make_processor(phase="test", load_weights="weights.pt")
        with mock.patch.object(processor, "test") as mock_test:
            processor.start()

        self.assertEqual(mock_test.call_count, 2)
        mock_test.assert_any_call("dev", 6667)
        mock_test.assert_any_call("test", 6667)
        processor.recoder.print_log.assert_any_call("Model:   DummyModel.")
        processor.recoder.print_log.assert_any_call("Weights: weights.pt.")
        processor.recoder.print_log.assert_any_call("Evaluation Done.\n")


if __name__ == "__main__":
    unittest.main()
