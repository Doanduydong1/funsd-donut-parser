## 👨‍💻 Me

- **Trần Mạnh Khôi**
- GitHub: [Doanduyduong1](https://github.com/Doanduyduong1)
- LinkedIn: [khoitm11](https://www.linkedin.com/in/khoitm11)

# FUNSD Document Parser 📄➡️💻

Hi! This project uses the Donut model to try and understand scanned documents from the FUNSD dataset. It tracks experiments with MLflow and can be run with Docker.

**Quick Note:** I'm using the basic pre-trained Donut model (`naver-clova-ix/donut-base`). The text it pulls out is pretty raw – not a perfect JSON. For better results, fine-tuning the model on FUNSD would be needed, but the main goal here was to set up the A-to-Z pipeline!

## 🚀 What I Built

*   Parses FUNSD document images with Donut.
*   Saves the raw text output as JSON files.
*   Uses MLflow to keep track of runs, parameters, and outputs.
*   Includes a Dockerfile to package it all up for Windows.

## 🛠️ Tech Used

*   Python 3.8+
*   PyTorch
*   Hugging Face Transformers (Donut)
*   MLflow
*   Docker
*   Pillow, OpenCV

## 📊 Dataset

*   **FUNSD:** [https://guillaumejaume.github.io/FUNSD/](https://guillaumejaume.github.io/FUNSD/)
    (I used the `testing_data/images` part. You'll need to download it yourself.)
*   Please cite the original paper if you use FUNSD:
    ```
    @inproceedings{Jaume2019FUNSDAD,
        title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
        author={Guillaume Jaume and Hazim Kemal Ekenel and Jean-Philippe Thiran},
        booktitle={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
        year={2019},
        volume={2},
        pages={1-6}
    }
    ```

## ⚙️ You'll Need (For Windows)

*   Python (3.8 or higher recommended)
*   Pip (Python package installer)
*   Git
*   Docker Desktop (running)

## 🏁 How to Get Started

1.  **Clone this repo:**
    ```bash
    git clone <YOUR_GITHUB_REPOSITORY_URL_HERE>
    cd funsd-donut-document-parser
    ```
    *(Replace `<YOUR_GITHUB_REPOSITORY_URL_HERE>` after you create your repo!)*

2.  **Add FUNSD Images:**
    *   Download FUNSD from its [official page](https://guillaumejaume.github.io/FUNSD/).
    *   Extract the downloaded `dataset.zip`.
    *   Copy images from the `dataset/testing_data/images/` directory (from the extracted FUNSD data) into this project's `funsd_dataset/testing_data/images/` folder.

3.  **Set up a Python Virtual Environment (good idea!):**
    ```bash
    python -m venv venv
    ```
    Activate it (on Windows):
    ```powershell
    .\venv\Scripts\activate
    ```

4.  **Install libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: PyTorch CPU is installed via the Dockerfile. For local runs, ensure you have PyTorch installed fitting your system. The `requirements.txt` is set up for the Docker environment.)*

## 🏃‍♀️ Running the Code

### 1. Locally (Python Script on Windows)

1.  Navigate to the `app` directory:
    ```powershell
    cd app
    ```
2.  Run the script:
    ```powershell
    # Example: Process the first 5 images
    python run_funsd_parser.py --max_images_to_process 5
    ```
    (Default input is `../funsd_dataset/testing_data/images`, default output is `../parsed_outputs`)
3.  Go back to the project root after running:
    ```powershell
    cd ..
    ```
4.  Parsed JSON outputs will be in the `parsed_outputs/` directory.
5.  **See MLflow tracking:**
    (From the project root directory: `funsd_donut_document_parser/`)
    ```powershell
    mlflow ui
    ```
    Then open your web browser to `http://localhost:5000`. Look for the "FUNSD\_Donut\_Document\_Parsing" experiment.

### 2. With Docker 🐳 (on Windows)

1.  **Build the Docker image:**
    (Make sure Docker Desktop is running. From the project root directory.)
    ```powershell
    docker build -t funsd-parser:latest .
    ```

2.  **Prepare input/output folders on your computer (Windows):**
    *   Example for input: `D:\DockerTest\my_inputs` (put your FUNSD .png images here)
    *   Example for output: `D:\DockerTest\my_outputs` (this folder will get the JSON results)

3.  **Run the Docker container (Windows - PowerShell or Command Prompt):**
    (Replace `D:\DockerTest\...` paths with your actual folder paths!)
    *(Use `^` for CMD line breaks, or ``` ` ``` for PowerShell, or write it all on one line)*
    ```powershell
    docker run --rm `
      -v "D:\DockerTest\my_inputs:/app/mounted_funsd_input_images" `
      -v "D:\DockerTest\my_outputs:/app/mounted_parsed_outputs" `
      funsd-parser:latest
    ```
    JSON results will appear in your `my_outputs` folder on your computer.

    *Heads-up on MLflow with Docker:* When running with Docker, MLflow logs will be created *inside* the container (and deleted if you use `--rm`). For persistent MLflow logs with Docker, you'd need to mount the `mlruns` directory or use a remote MLflow server (a bit more advanced for this project!).

## 📂 Project Structure

- `funsd_donut_document_parser/`
  - `.git/`             *(Git repository data)*
  - `.idea/`            *(PyCharm project files - ignored by Git)*
  - `venv/`             *(Python virtual environment - ignored by Git)*
  - `app/`              *(Main application code)*
    - `mlruns/`         *(MLflow data for local runs - should be in .gitignore)*
    - `run_funsd_parser.py` *(The main Python script)*
  - `funsd_dataset/`    *(Placeholder for FUNSD data)*
    - `testing_data/`
      - `images/`       *(Place FUNSD .png images here)*
      - `annotations/`  *(FUNSD .json annotations - if used)*
  - `parsed_outputs/`   *(Local script outputs - should be in .gitignore)*
  - `.dockerignore`     *(Files to ignore for Docker build context)*
  - `.gitignore`        *(Files Git should ignore)*
  - `Dockerfile`        *(Instructions to build the Docker image)*
  - `requirements.txt`  *(Python dependencies)*
  - `README.md`         *(This file!)*

## ✨ Example Output Snippet

From an image like `82092117.png`, the `raw_donut_output` in the JSON might look something like this:
```json
{
    "image_name": "82092117.png",
    "raw_donut_output": "\\ Attorney General Betty D. Montgomery . CONFIDENTIAL FACSIMILE TRANSMISSION ... (actual output is longer and will vary per image)",
    "parsed_status": "raw_text"
}

