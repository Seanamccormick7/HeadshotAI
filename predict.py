# predict.py
from cog import BasePredictor, Input, Path
import os
import subprocess

class Predictor(BasePredictor):
    def predict(
        self,
        instance_data: Path = Input(
            description="Zip or directory with instance images",
        ),
        instance_prompt: str = Input(
            description="Prompt for the instance, e.g., 'photo of sks person'"
        ),
        steps: int = Input(
            description="Number of training steps",
            default=800
        ),
        output_dir: str = Input(
            description="Where to save model weights",
            default="trained_model"
        ),
    ) -> str:
        """
        Train DreamBooth. Return path to final model directory (or a sample image).
        """

        # 1. Unzip or handle directory
        instance_data_dir = "/src/instance_images"
        if instance_data.is_dir():
            # If the user provided a directory, use it directly
            instance_data_dir = str(instance_data)
        else:
            # If the user uploaded a zip, unzip into /src/instance_images
            os.system(f"unzip -o {instance_data} -d /src/instance_images")
            # Flatten any subfolder structure: move all files up to /src/instance_images
            # ignoring subfolders. Then remove any empty subfolders.
            os.system("find /src/instance_images -mindepth 2 -type f -exec mv -t /src/instance_images {} +")
            os.system("find /src/instance_images -mindepth 1 -type d -empty -delete")

        # 2. Build the command for train_dreambooth.py
        cmd = [
            "python", "dreambooth/train_dreambooth.py",
            "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
            f"--instance_data_dir={instance_data_dir}",
            f"--instance_prompt={instance_prompt}",
            "--resolution=512",
            "--train_batch_size=1",
            "--gradient_accumulation_steps=1",
            "--learning_rate=5e-6",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            f"--max_train_steps={steps}",
            f"--output_dir={output_dir}",
            "--mixed_precision=fp16"
        ]

        # 3. Run the training
        subprocess.run(cmd, check=True)

        # 4. Optionally generate a sample image from the newly trained model
        # Or just return the path to the model directory
        return f"Training complete. Model saved at {output_dir}/"

