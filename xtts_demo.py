import argparse
import os
import sys
from pathlib import Path
import shutil
import torch
import torchaudio
import traceback
from utils.formatter import format_audio_list, list_audios
from utils.gpt_train import train_gpt
from faster_whisper import WhisperModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def preprocess_dataset(audio_path, language, whisper_model, out_path):
    clear_gpu_cache()
    
    out_path = os.path.join(out_path, "dataset")
    os.makedirs(out_path, exist_ok=True)
    
    audio_files = list(list_audios(audio_path))
    
    if not audio_files:
        return "No audio files found!", "", ""
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
        train_meta, eval_meta, audio_total_size = format_audio_list(audio_files, asr_model=asr_model, target_language=language, out_path=out_path)
    except Exception as e:
        traceback.print_exc()
        return f"Error during data processing: {str(e)}", "", ""
    
    if audio_total_size < 120:
        return "The total duration of audio files should be at least 2 minutes!", "", ""
    
    print("Dataset Processed!")
    return "Dataset Processed!", train_meta, eval_meta

def train_model(custom_model, version, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    clear_gpu_cache()

    run_dir = Path(output_path) / "run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    
    lang_file_path = Path(output_path) / "dataset" / "lang.txt"
    if lang_file_path.exists():
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()
            if current_language != language:
                print(f"Warning: Dataset language ({current_language}) does not match specified language ({language}). Using dataset language.")
                language = current_language
    
    if not train_csv or not eval_csv:
        return "Error: Train CSV and Eval CSV are required!", "", "", "", ""
    
    try:
        max_audio_length = int(max_audio_length * 22050)
        speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
            custom_model, version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, 
            output_path=output_path, max_audio_length=max_audio_length
        )
    except Exception as e:
        traceback.print_exc()
        return f"Error during training: {str(e)}", "", "", "", ""

    ready_dir = Path(output_path) / "ready"
    ready_dir.mkdir(exist_ok=True)

    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")
    ft_xtts_checkpoint = os.path.join(ready_dir, "unoptimize_model.pth")

    speaker_reference_new_path = ready_dir / "reference.wav"
    shutil.copy(speaker_wav, speaker_reference_new_path)

    print("Model training completed!")
    return "Model training completed!", config_path, vocab_file, ft_xtts_checkpoint, speaker_xtts_path, speaker_reference_new_path

def optimize_model(out_path):
    out_path = Path(out_path)
    ready_dir = out_path / "ready"
    model_path = ready_dir / "unoptimize_model.pth"

    if not model_path.is_file():
        return "Unoptimized model not found in ready folder", ""

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    del checkpoint["optimizer"]

    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    os.remove(model_path)

    optimized_model = ready_dir / "model.pth"
    torch.save(checkpoint, optimized_model)

    clear_gpu_cache()
    return f"Model optimized and saved at {optimized_model}!", str(optimized_model)

def main():
    parser = argparse.ArgumentParser(description="XTTS fine-tuning script")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the folder with audio files")
    parser.add_argument("--language", type=str, required=True, help="Dataset language")
    parser.add_argument("--out_path", type=str, default="xtts-finetune-webui/finetune_models", help="Output path")
    parser.add_argument("--num_epochs", type=int, default=32, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_acumm", type=int, default=8, help="Grad accumulation steps")
    parser.add_argument("--max_audio_length", type=int, default=20, help="Max permitted audio size in seconds")
    args = parser.parse_args()

    print("Step 1: Creating dataset")
    result, train_csv, eval_csv = preprocess_dataset(args.audio_path, args.language, "large-v3", args.out_path)
    print(result)

    if "Error" in result or "!" in result:
        return

    print("\nStep 2: Training the model")
    result, config_path, vocab_file, checkpoint, speaker_path, reference_audio = train_model(
        "", "v2.0.2", args.language, train_csv, eval_csv, args.num_epochs, args.batch_size, 
        args.grad_acumm, args.out_path, args.max_audio_length
    )
    print(result)

    if "Error" in result:
        return

    print("\nStep 2.5: Optimizing the model")
    result, optimized_checkpoint = optimize_model(args.out_path)
    print(result)

    print("\nTraining and optimization completed. Model files are saved in the 'ready' folder within the output directory.")

if __name__ == "__main__":
    main()
