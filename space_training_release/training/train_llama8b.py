"""Single-GPU DPO with training-loss plateau stopping and durable checkpoints."""
import argparse
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer
from stage1.dpo_training_chat import _set_reproducible_seed, _build_dpo_dataset_from_raw


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class Progress(TrainerCallback):
    def __init__(self):
        self.stop_requested = False
        self.started = None
        self.initial_step = 0
        signal.signal(signal.SIGUSR1, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, *_):
        self.stop_requested = True

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()
        self.initial_step = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        steps = state.global_step - self.initial_step
        if steps > 0 and steps % 10 == 0:
            seconds = (time.monotonic() - self.started) / steps
            print(f'TIMING step={state.global_step} epoch={state.epoch:.3f} seconds_per_step={seconds:.2f}', flush=True)
        if self.stop_requested:
            control.should_save = True
            control.should_training_stop = True
        return control


class LossPlateau(TrainerCallback):
    """Stop when two consecutive 100-step mean losses fail to improve by 5%."""

    def __init__(self, steps_per_epoch, log_interval=10, window_steps=100,
                 relative_improvement=0.05, patience=2):
        assert window_steps % log_interval == 0
        self.min_step = int(steps_per_epoch)
        self.logs_per_window = window_steps // log_interval
        self.relative_improvement = relative_improvement
        self.patience = patience
        self.losses = []
        self.best = None
        self.stale = 0
        self.triggered = False
        self.stop_record = None

    def _observe(self, step, loss, *, emit):
        self.losses.append(float(loss))
        if len(self.losses) % self.logs_per_window:
            return False
        mean = sum(self.losses[-self.logs_per_window:]) / self.logs_per_window
        if step < self.min_step or self.best is None or mean < self.best * (1 - self.relative_improvement):
            self.best = mean if self.best is None else min(self.best, mean)
            self.stale = 0
        else:
            self.stale += 1
        if emit:
            print(f'LOSS_WINDOW step={step} mean={mean:.6f} best={self.best:.6f} stale={self.stale}/{self.patience}', flush=True)
        if step >= self.min_step and self.stale >= self.patience:
            self.triggered = True
            self.stop_record = {'step': step, 'window_mean': mean, 'best_mean': self.best,
                                'stale_windows': self.stale, 'min_step': self.min_step,
                                'relative_improvement': self.relative_improvement}
            return True
        return False

    def on_train_begin(self, args, state, control, **kwargs):
        # Rebuild the decision after a Slurm restart from checkpoint history.
        for entry in state.log_history:
            if 'loss' in entry:
                self._observe(int(entry['step']), entry['loss'], emit=False)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs and self._observe(state.global_step, logs['loss'], emit=True):
            control.should_save = True
            control.should_training_stop = True
            print(f'EARLY_STOP_LOSS_PLATEAU step={state.global_step} epoch={state.epoch:.3f}', flush=True)
        return control


class DurableTrainer(DPOTrainer):
    def _save_checkpoint(self, model, trial):
        # Rotate only after the new checkpoint has been fully written.
        super()._save_checkpoint(model, trial)
        path = Path(self.args.output_dir) / f'checkpoint-{self.state.global_step}'
        required = ['adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json']
        for name in required:
            if not (path / name).is_file():
                raise RuntimeError(f'Incomplete checkpoint: {path / name}')
        write_json(path / 'COMPLETE.json', {'step': self.state.global_step, 'files': {name: (path / name).stat().st_size for name in required}})
        complete = sorted((p for p in Path(self.args.output_dir).glob('checkpoint-*') if (p / 'COMPLETE.json').exists()), key=lambda p: int(p.name.split('-')[-1]))
        for old in complete[:-2]:
            shutil.rmtree(old)
        print(f'CHECKPOINT_SAVED {path}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', choices=['energy','information','decision','execution'], required=True)
    parser.add_argument('--preferred', required=True)
    parser.add_argument('--base-model-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    base = Path(args.base_model_path).resolve()
    out = Path(args.output).resolve()
    from common.pipeline_utils import opposite_preferred_subtype
    opposite_preferred_subtype(args.dimension, args.preferred)
    out.mkdir(parents=True, exist_ok=True)
    model_manifest = json.loads((base / 'SHA256SUMS.json').read_text())
    for name, expected in model_manifest.items():
        if digest(base / name) != expected:
            raise RuntimeError(f'Model checksum mismatch: {name}')
    recipe = {
        'dimension': args.dimension, 'preferred': args.preferred, 'base_sha256': model_manifest,
        'data_sha256': {str(p): digest(p) for p in sorted(Path('datasets/training_raw').glob(f'en_{args.dimension}_*.json'))},
        'seed': 42, 'batch_size': 16, 'gradient_accumulation_steps': 2,
        'learning_rate': 1e-5, 'lr_scheduler': 'constant', 'beta': 1.0,
        'early_stop': {'metric': 'training_dpo_loss', 'min_epochs': 1,
                       'window_steps': 100, 'relative_improvement': 0.05,
                       'patience_windows': 2},
        'safety_limit_epochs': 20,
        'lora': {'r': 32, 'alpha': 64, 'dropout': 0.1}, 'load_dtype': 'float16',
        'reference_mode': 'same_base_with_adapter_disabled', 'save_steps': 100, 'smoke': args.smoke,
        'versions': {p: md.version(p) for p in ['torch','transformers','trl','peft','accelerate','datasets']},
        'runner_sha256': digest(__file__), 'dataset_builder_sha256': digest('stage1/dpo_training_chat.py'),
    }
    manifest = out / 'recipe.json'
    if manifest.exists() and json.loads(manifest.read_text()) != recipe:
        raise RuntimeError('Existing experiment recipe differs; refusing to mix runs')
    write_json(manifest, recipe)
    if (out / 'TRAINING_COMPLETE.json').exists():
        print('Training already complete; skipping.', flush=True)
        return
    checkpoints = out / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    valid = []
    for p in checkpoints.glob('checkpoint-*'):
        marker = p / 'COMPLETE.json'
        if marker.exists():
            info = json.loads(marker.read_text())
            if all((p / n).exists() and (p / n).stat().st_size == size for n, size in info['files'].items()):
                valid.append(p)
    resume = str(max(valid, key=lambda p: int(p.name.split('-')[-1]))) if valid else None
    print(f'RESUME {resume}', flush=True)
    _set_reproducible_seed()
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    data = _build_dpo_dataset_from_raw(tokenizer, args.dimension, args.preferred)
    steps_per_epoch = math.ceil(len(data) / 32)
    if steps_per_epoch < 1:
        raise RuntimeError('Empty training dataset')
    # Trainer requires a finite max_steps. This is an emergency ceiling, not an
    # epoch target: reaching it without a loss plateau is an incomplete run.
    safety_max_steps = 3 if args.smoke else 20 * steps_per_epoch
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map={'': 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_dropout=0.1, bias='none', task_type='CAUSAL_LM'))
    # TRL uses the unchanged base weights with LoRA disabled as the reference.
    ref = None
    cfg = DPOConfig(output_dir=str(checkpoints), per_device_train_batch_size=16,
        gradient_accumulation_steps=2, max_steps=safety_max_steps,
        learning_rate=1e-5, lr_scheduler_type='constant',
        beta=1.0, save_strategy='steps', save_steps=100, save_total_limit=None,
        save_only_model=False, bf16=False, seed=42, data_seed=42,
        logging_steps=10, report_to='none', logging_dir=str(out / 'trainer-logs'))
    cfg_path = out / 'effective_config.json'
    current_config = json.loads(json.dumps(cfg.to_dict()))
    if cfg_path.exists():
        saved_config = json.loads(cfg_path.read_text())
        changed = [key for key in sorted(set(saved_config) | set(current_config))
                   if saved_config.get(key) != current_config.get(key)]
        if changed:
            raise RuntimeError(f'Effective training configuration changed: {changed}')
    write_json(cfg_path, cfg.to_dict())
    progress = Progress()
    plateau = LossPlateau(steps_per_epoch=steps_per_epoch)
    callbacks = [progress, plateau]
    trainer = DurableTrainer(model=model, ref_model=ref, args=cfg, train_dataset=data, processing_class=tokenizer, callbacks=callbacks)
    print('REFERENCE_MODE adapter_disabled device=' + str(next(model.parameters()).device), flush=True)
    result = trainer.train(resume_from_checkpoint=resume)
    if args.smoke:
        if trainer.state.global_step != 3 or not math.isfinite(result.metrics['train_loss']):
            raise RuntimeError('Smoke validation failed')
        write_json(out / 'SMOKE_COMPLETE.json', {'steps': 3, 'metrics': result.metrics})
        print('SMOKE_COMPLETE', flush=True)
        return
    if progress.stop_requested:
        trainer._save_checkpoint(trainer.model, trial=None)
        print('PAUSED_BY_SIGNAL: checkpoint saved; resubmit to continue.', flush=True)
        raise SystemExit(75)
    if not plateau.triggered:
        trainer._save_checkpoint(trainer.model, trial=None)
        write_json(out / 'SAFETY_LIMIT_OR_UNEXPECTED_STOP.json', {
            'step': trainer.state.global_step, 'safety_max_steps': safety_max_steps,
            'epoch': trainer.state.epoch, 'reason': 'No training-loss plateau detected'})
        print('INCOMPLETE: no loss plateau; inspect before resubmitting.', flush=True)
        raise SystemExit(76)
    # Save the final resumable state before merging; exporting can be retried.
    trainer._save_checkpoint(trainer.model, trial=None)
    write_json(out / 'train_metrics.json', result.metrics)
    final_tmp = out / 'final.incomplete'
    if final_tmp.exists():
        shutil.rmtree(final_tmp)
    final_tmp.mkdir()
    merged = model.merge_and_unload()
    merged.save_pretrained(final_tmp, safe_serialization=True)
    tokenizer.save_pretrained(final_tmp)
    if (out / 'final').exists():
        shutil.rmtree(out / 'final')
    final_tmp.rename(out / 'final')
    write_json(out / 'TRAINING_COMPLETE.json', {
        'step': trainer.state.global_step, 'epoch': trainer.state.epoch,
        'safety_max_steps': trainer.state.max_steps, 'model': str(out / 'final'),
        'stop_reason': 'loss_plateau',
        'loss_plateau': plateau.stop_record})
    print('TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
