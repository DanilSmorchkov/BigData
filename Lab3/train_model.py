from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

wandb.init(project="yolov8n", job_type="training")

model = YOLO("yolov8n.pt")
add_wandb_callback(model, enable_model_checkpointing=True)

model.train(data='data.yaml', epochs=10)

wandb.finish()
