# app/run_funsd_parser.py

import argparse
import glob
import json
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# --- CONFIGURATION ---
DEFAULT_DONUT_MODEL_NAME = "naver-clova-ix/donut-base"
DEFAULT_TASK_PROMPT = "<s_iitcdip>"


def setup_argument_parser():
    parser = argparse.ArgumentParser(description="Parse FUNSD documents using Donut model and track with MLflow.")
    parser.add_argument(
        "--funsd_image_input_dir",
        type=str,
        default="../funsd_dataset/testing_data/images", # Đường dẫn tương đối đến thư mục ảnh FUNSD
        help="Path to the directory containing FUNSD .png images (testing_data/images)."
    )
    parser.add_argument(
        "--parsed_output_dir",
        type=str,
        default="../parsed_outputs",
        help="Directory to save the parsed JSON outputs from Donut."
    )
    parser.add_argument(
        "--donut_model_hf_name",
        type=str,
        default=DEFAULT_DONUT_MODEL_NAME,
        help="Name or path of the pre-trained Donut model from Hugging Face Hub."
    )
    parser.add_argument(
        "--donut_task_prompt",
        type=str,
        default=DEFAULT_TASK_PROMPT,
        help="Task-specific prompt for the Donut model (e.g., '<s_iitcdip>', '<s_cord-v2>', etc.)."
    )
    parser.add_argument(
        "--max_images_to_process",
        type=int,
        default=5, # Xử lý ít ảnh để test nhanh, đặc biệt cho dự án 1 ngày
        help="Maximum number of images to process. Set to 0 or negative for all images."
    )
    parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        default="FUNSD_Donut_Document_Parsing",
        help="Name for the MLflow experiment."
    )
    return parser


def ensure_directory_exists(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)


def load_donut_model_and_processor(model_hf_name: str, device: torch.device):
    print(f"Loading Donut model and processor: {model_hf_name}...")
    try:
        processor = DonutProcessor.from_pretrained(model_hf_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_hf_name)
        model.to(device)
        model.eval()
        print("Donut model and processor loaded successfully.")
        return processor, model
    except Exception as e:
        print(f"ERROR: Could not load model or processor '{model_hf_name}'. Exception: {e}")
        raise

def process_single_image_with_donut(
    image_path: Path,
    donut_processor: DonutProcessor,
    donut_model: VisionEncoderDecoderModel,
    task_prompt: str,
    device: torch.device
):
    try:
        image = Image.open(image_path).convert("RGB")

        pixel_values = donut_processor(image, return_tensors="pt").pixel_values.to(device)

        decoder_input_ids = donut_processor.tokenizer(
            task_prompt,
            add_special_tokens=False, # Donut thường tự thêm token đặc biệt qua processor khi generate
            return_tensors="pt"
        ).input_ids.to(device)

        # Sinh output từ model
        with torch.no_grad(): # Không cần tính gradient khi inference
            generated_outputs = donut_model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=donut_model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=donut_processor.tokenizer.pad_token_id,
                eos_token_id=donut_processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[donut_processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        raw_sequence = donut_processor.batch_decode(generated_outputs.sequences)[0]

        cleaned_sequence = raw_sequence.replace(donut_processor.tokenizer.eos_token, "").replace(donut_processor.tokenizer.pad_token, "")

        if task_prompt and cleaned_sequence.strip().startswith(task_prompt.strip()):
             cleaned_sequence = cleaned_sequence.strip().replace(task_prompt.strip(), "", 1).strip()

        print(f"  Successfully processed: {image_path.name}")
        print(f"  Raw model output (cleaned): {cleaned_sequence[:200]}...") # In ra 200 ký tự đầu
        return {"image_name": image_path.name, "raw_donut_output": cleaned_sequence.strip()}

    except Exception as e:
        print(f"  ERROR processing {image_path.name}: {e}")
        return {"image_name": image_path.name, "error": str(e), "raw_donut_output": ""}


def run_parsing_pipeline(args):

    input_dir = Path(args.funsd_image_input_dir)
    output_dir = Path(args.parsed_output_dir)
    ensure_directory_exists(output_dir)

    if not input_dir.is_dir():
        print(f"ERROR: Input directory '{input_dir}' not found or is not a directory.")
        return

    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Pytorch device: {selected_device}")

    # Load model and processor
    try:
        donut_processor, donut_model = load_donut_model_and_processor(args.donut_model_hf_name, selected_device)
    except Exception:
        print("Exiting due to model loading failure.")
        return

    # MLflow experiment
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"Donut_{Path(args.donut_model_hf_name).name}_FUNSD_Run") as mlf_run:
        print(f"MLflow Run ID: {mlf_run.info.run_id}")
        print(f"MLflow Experiment ID: {mlf_run.info.experiment_id}")
        print(f"MLflow artifacts will be logged to: {mlf_run.info.artifact_uri}")

        # Log các tham số của lần chạy này
        mlflow.log_param("donut_model_name", args.donut_model_hf_name)
        mlflow.log_param("task_prompt_used", args.donut_task_prompt if args.donut_task_prompt else "None (default model behavior)")
        mlflow.log_param("funsd_image_source_dir", str(input_dir.resolve())) # Lưu đường dẫn tuyệt đối
        mlflow.log_param("parsed_json_output_dir", str(output_dir.resolve()))
        mlflow.log_param("pytorch_device", str(selected_device))

        # Log model
        try:
            mlflow.pytorch.log_model(
                pytorch_model=donut_model,
                artifact_path="donut-model-hf-snapshot", # Tên thư mục trong artifacts
                # signature= # Bỏ qua signature cho đơn giản trong dự án 1 ngày, nếu không sẽ rất phức tạp
                # input_example= # Tương tự, bỏ qua input_example
                # registered_model_name= # Sẽ đăng ký model nếu muốn dùng Model Registry
            )
            print("Donut pre-trained model snapshot logged to MLflow artifacts.")
            mlflow.log_param("mlflow_model_snapshot_logged", True)
        except Exception as e_log_model:
            print(f"Warning: Could not log PyTorch model snapshot to MLflow: {e_log_model}")
            mlflow.log_param("mlflow_model_snapshot_logged", False)
            mlflow.log_param("mlflow_model_snapshot_error", str(e_log_model))


        # Tìm tất cả file ảnh .png trong thư mục input
        image_file_paths = sorted(list(input_dir.glob("*.png")))

        if not image_file_paths:
            print(f"No .png images found in '{input_dir}'. Please check the path.")
            mlflow.log_metric("images_found_in_dir", 0)
            return

        mlflow.log_metric("images_found_in_dir", len(image_file_paths))

        # Giới hạn số lượng ảnh xử lý nếu được chỉ định
        num_to_process = args.max_images_to_process
        if num_to_process <= 0 or num_to_process > len(image_file_paths):
            num_to_process = len(image_file_paths)

        print(f"Will process {num_to_process} out of {len(image_file_paths)} found images.")
        mlflow.log_param("actual_images_to_process", num_to_process)

        processed_successfully_count = 0
        error_count = 0

        for i, img_path in enumerate(image_file_paths[:num_to_process]):
            print(f"\n--- Processing image {i+1}/{num_to_process}: {img_path.name} ---")

            # Thực hiện parsing
            parsing_result_dict = process_single_image_with_donut(
                img_path, donut_processor, donut_model, args.donut_task_prompt, selected_device
            )

            # Lưu kết quả parsing vào file JSON
            output_json_filename = f"{img_path.stem}_parsed_output.json"
            output_json_filepath = output_dir / output_json_filename

            try:
                with open(output_json_filepath, "w", encoding="utf-8") as f_out:
                    json.dump(parsing_result_dict, f_out, ensure_ascii=False, indent=4)

                # Log file JSON kết quả và ảnh gốc vào MLflow artifacts
                mlflow.log_artifact(str(output_json_filepath), artifact_path="individual_parsed_outputs")
                mlflow.log_artifact(str(img_path), artifact_path="original_input_images")

                if "error" not in parsing_result_dict:
                    processed_successfully_count += 1
                else:
                    error_count += 1
                    mlflow.log_param(f"error_image_{img_path.name}", parsing_result_dict.get("error"))

            except Exception as e_save_json:
                print(f"  ERROR saving JSON output for {img_path.name}: {e_save_json}")
                error_count += 1 # Đếm cả lỗi này
                mlflow.log_param(f"error_saving_json_{img_path.name}", str(e_save_json))

        # Log metrics tổng kết
        mlflow.log_metric("images_processed_successfully", processed_successfully_count)
        mlflow.log_metric("images_with_processing_errors", error_count)
        mlflow.set_tag("Overall_Run_Status", "Completed" if error_count == 0 else "Completed_With_Errors")

        print("\n--- Document Parsing Pipeline Finished ---")
        print(f"Successfully processed: {processed_successfully_count}/{num_to_process} images.")
        print(f"Errors encountered: {error_count}/{num_to_process} images.")
        print(f"Parsed outputs (JSON files) are saved in: {output_dir.resolve()}")
        print("To view MLflow experiment tracking, run 'mlflow ui' in your project's root directory and open http://localhost:5000")


if __name__ == "__main__":
    cli_args = setup_argument_parser().parse_args()
    run_parsing_pipeline(cli_args)