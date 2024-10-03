import argparse
import os
import sys
import tempfile
from pathlib import Path

import shutil
import traceback

import torch
import torchaudio

from utils.formatter import format_audio_list, find_latest_best_model, list_audios
from utils.gpt_train import train_gpt

from faster_whisper import WhisperModel

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Clear logs
def remove_log_file(file_path):
    log_file = Path(file_path)
    if log_file.exists() and log_file.is_file():
        log_file.unlink()

# Clear GPU cache
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Global XTTS model variable
XTTS_MODEL = None

def load_model(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        raise ValueError("You need to provide `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path`.")
    
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model...")
    XTTS_MODEL.load_checkpoint(
        config, 
        checkpoint_path=xtts_checkpoint, 
        vocab_path=xtts_vocab,
        speaker_file_path=xtts_speaker, 
        use_deepspeed=False
    )
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()
    print("Model Loaded!")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file, temperature, length_penalty,
            repetition_penalty, top_k, top_p, sentence_split, use_config):
    if XTTS_MODEL is None or not speaker_audio_file:
        raise ValueError("You need to load the model and provide a speaker audio file.")
    
    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=speaker_audio_file, 
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, 
        max_ref_length=XTTS_MODEL.config.max_ref_len, 
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs
    )
    
    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature,
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
            enable_text_splitting=True
        )
    else:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting=sentence_split
        )
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)
    
    print("Speech generated at:", out_path)
    return out_path

def load_params_tts(out_path, version):
    out_path = Path(out_path)
    ready_model_path = out_path / "ready"

    vocab_path = ready_model_path / "vocab.json"
    config_path = ready_model_path / "config.json"
    speaker_path = ready_model_path / "speakers_xtts.pth"
    reference_path = ready_model_path / "reference.wav"

    model_path = ready_model_path / "model.pth"

    if not model_path.exists():
        model_path = ready_model_path / "unoptimize_model.pth"
        if not model_path.exists():
            raise FileNotFoundError("Params for TTS not found.")
    
    return model_path, config_path, vocab_path, speaker_path, reference_path

def preprocess_dataset(audio_files, audio_folder_path, language, whisper_model, out_path):
    clear_gpu_cache()
    
    out_path = os.path.join(out_path, "dataset")
    os.makedirs(out_path, exist_ok=True)

    if audio_folder_path:
        audio_files = list(list_audios(audio_folder_path))
    else:
        audio_files = audio_files

    if not audio_files:
        raise FileNotFoundError("No audio files found! Please provide files or specify a folder path.")
    else:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "float32"
            asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
            train_meta, eval_meta, audio_total_size = format_audio_list(
                audio_files, 
                asr_model=asr_model, 
                target_language=language, 
                out_path=out_path
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Data processing was interrupted due to an error: {e}")
    
    if audio_total_size < 120:
        raise ValueError("The sum of the duration of the audios provided should be at least 2 minutes!")
    
    print("Dataset Processed!")
    return train_meta, eval_meta

def train_model(custom_model, version, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    clear_gpu_cache()

    run_dir = Path(output_path) / "run"

    if run_dir.exists():
        shutil.rmtree(run_dir)
    
    lang_file_path = Path(output_path) / "dataset" / "lang.txt"
    current_language = None
    if lang_file_path.exists():
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()
            if current_language != language:
                print("The language specified does not match the dataset language. Using dataset language.")
                language = current_language
    
    if not train_csv or not eval_csv:
        raise ValueError("Train CSV and Eval CSV must be provided.")
    
    try:
        max_audio_length = int(max_audio_length * 22050)
        speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
            custom_model,
            version,
            language,
            num_epochs,
            batch_size,
            grad_acumm,
            train_csv,
            eval_csv,
            output_path=output_path,
            max_audio_length=max_audio_length
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Training was interrupted due to an error: {e}")
    
    ready_dir = Path(output_path) / "ready"
    ready_dir.mkdir(exist_ok=True)

    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")

    speaker_reference_new_path = ready_dir / "reference.wav"
    shutil.copy(speaker_wav, speaker_reference_new_path)

    print("Model training done!")
    return config_path, vocab_file, ready_dir / "unoptimize_model.pth", speaker_xtts_path, speaker_reference_new_path

def optimize_model(out_path, clear_train_data):
    out_path = Path(out_path)
    ready_dir = out_path / "ready"
    run_dir = out_path / "run"
    dataset_dir = out_path / "dataset"

    # Clear specified training data directories
    if clear_train_data in {"run", "all"} and run_dir.exists():
        shutil.rmtree(run_dir)
    if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    model_path = ready_dir / "unoptimize_model.pth"

    if not model_path.is_file():
        raise FileNotFoundError("Unoptimized model not found in ready folder.")

    # Load the checkpoint and remove unnecessary parts
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    if "optimizer" in checkpoint:
        del checkpoint["optimizer"]
    model_keys = list(checkpoint["model"].keys())
    for key in model_keys:
        if "dvae" in key:
            del checkpoint["model"][key]

    os.remove(model_path)

    # Save the optimized model
    optimized_model = ready_dir / "model.pth"
    torch.save(checkpoint, optimized_model)
    ft_xtts_checkpoint = str(optimized_model)

    clear_gpu_cache()
    print(f"Model optimized and saved at {ft_xtts_checkpoint}!")
    return ft_xtts_checkpoint

def main():
    parser = argparse.ArgumentParser(
        description="XTTS Fine-Tuning Script"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess dataset')
    preprocess_parser.add_argument('--audio_files', nargs='*', help='List of audio file paths', default=[])
    preprocess_parser.add_argument('--audio_folder_path', type=str, help='Path to folder with audio files', default="")
    preprocess_parser.add_argument('--language', type=str, help='Dataset language (e.g., en, es)', required=True)
    preprocess_parser.add_argument('--whisper_model', type=str, help='Whisper model to use', default="large-v3")
    preprocess_parser.add_argument('--out_path', type=str, help='Output path', default=str(Path.cwd() / "finetune_models"))

    # Train command
    train_parser = subparsers.add_parser('train', help='Train XTTS model')
    train_parser.add_argument('--custom_model', type=str, help='Path to custom model.pth file', default="")
    train_parser.add_argument('--version', type=str, help='XTTS base version', default="v2.0.2")
    train_parser.add_argument('--language', type=str, help='Dataset language (e.g., en, es)', required=True)
    train_parser.add_argument('--train_csv', type=str, help='Path to train CSV', required=True)
    train_parser.add_argument('--eval_csv', type=str, help='Path to eval CSV', required=True)
    train_parser.add_argument('--num_epochs', type=int, help='Number of epochs', default=32)
    train_parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
    train_parser.add_argument('--grad_acumm', type=int, help='Gradient accumulation steps', default=8)
    train_parser.add_argument('--output_path', type=str, help='Output path', default=str(Path.cwd() / "finetune_models"))
    train_parser.add_argument('--max_audio_length', type=int, help='Max audio length in seconds', default=20)

    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize trained XTTS model')
    optimize_parser.add_argument('--out_path', type=str, help='Output path', default=str(Path.cwd() / "finetune_models"))
    optimize_parser.add_argument('--clear_train_data', type=str, help='Clear training data (none, run, dataset, all)', default="none")

    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference with XTTS model')
    inference_parser.add_argument('--xtts_checkpoint', type=str, help='XTTS checkpoint path', required=True)
    inference_parser.add_argument('--xtts_config', type=str, help='XTTS config path', required=True)
    inference_parser.add_argument('--xtts_vocab', type=str, help='XTTS vocab path', required=True)
    inference_parser.add_argument('--xtts_speaker', type=str, help='XTTS speaker path', required=True)
    inference_parser.add_argument('--speaker_audio_file', type=str, help='Speaker reference audio file path', required=True)
    inference_parser.add_argument('--language', type=str, help='Language', default="en")
    inference_parser.add_argument('--tts_text', type=str, help='Input text for TTS', required=True)
    inference_parser.add_argument('--temperature', type=float, help='Temperature', default=0.75)
    inference_parser.add_argument('--length_penalty', type=float, help='Length penalty', default=1.0)
    inference_parser.add_argument('--repetition_penalty', type=float, help='Repetition penalty', default=5.0)
    inference_parser.add_argument('--top_k', type=int, help='Top K', default=50)
    inference_parser.add_argument('--top_p', type=float, help='Top P', default=0.85)
    inference_parser.add_argument('--sentence_split', action='store_true', help='Enable text splitting')
    inference_parser.add_argument('--use_config', action='store_true', help='Use inference settings from config')

    args = parser.parse_args()

    try:
        if args.command == 'preprocess':
            train_meta, eval_meta = preprocess_dataset(
                audio_files=args.audio_files,
                audio_folder_path=args.audio_folder_path,
                language=args.language,
                whisper_model=args.whisper_model,
                out_path=args.out_path
            )
            print("Preprocessing completed successfully.")
            print("Train CSV:", train_meta)
            print("Eval CSV:", eval_meta)

        elif args.command == 'train':
            config_path, vocab_file, xtts_checkpoint, xtts_speaker, speaker_reference_audio = train_model(
                custom_model=args.custom_model,
                version=args.version,
                language=args.language,
                train_csv=args.train_csv,
                eval_csv=args.eval_csv,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                grad_acumm=args.grad_acumm,
                output_path=args.output_path,
                max_audio_length=args.max_audio_length
            )
            print("Training completed successfully.")
            print("Config Path:", config_path)
            print("Vocab File:", vocab_file)
            print("XTTS Checkpoint:", xtts_checkpoint)
            print("XTTS Speaker Path:", xtts_speaker)
            print("Speaker Reference Audio:", speaker_reference_audio)

        elif args.command == 'optimize':
            optimized_model_path = optimize_model(
                out_path=args.out_path,
                clear_train_data=args.clear_train_data
            )
            print("Optimization completed successfully.")
            print("Optimized Model Path:", optimized_model_path)

        elif args.command == 'inference':
            load_model(
                xtts_checkpoint=args.xtts_checkpoint,
                xtts_config=args.xtts_config,
                xtts_vocab=args.xtts_vocab,
                xtts_speaker=args.xtts_speaker
            )
            generated_audio = run_tts(
                lang=args.language,
                tts_text=args.tts_text,
                speaker_audio_file=args.speaker_audio_file,
                temperature=args.temperature,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                top_k=args.top_k,
                top_p=args.top_p,
                sentence_split=args.sentence_split,
                use_config=args.use_config
            )
            print("Inference completed successfully.")
            print("Generated Audio Path:", generated_audio)

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
