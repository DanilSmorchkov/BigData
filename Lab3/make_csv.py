import os
from ultralytics import YOLO
import pandas as pd
import pathlib
from tqdm import tqdm
import re

path = os.path.join(pathlib.Path(__file__).parent, 'all_frames')
frames = sorted(os.listdir(path))

data = pd.DataFrame(columns=['date', 'file_name', 'time', 'target_is_broken', 'location'])

model = YOLO('best-2.pt')

for frame in tqdm(frames):
    path_to_frame = os.path.join(path, frame)
    results = model(path_to_frame)
    boxes = results[0].boxes
    split = re.split("[_.]", frame)
    date = split[0]
    file_name = split[1]
    time = int(split[4]) * 0.5
    is_broken = 1 in boxes.cls
    location = 0
    if boxes.cls.size()[0]:
        arg = boxes.conf.argmax().item()
        location = [((boxes[arg].xyxy[0][0] + boxes[arg].xyxy[0][2]).item() / 2,
                    (boxes[arg].xyxy[0][1] + boxes[arg].xyxy[0][3]).item() / 2)]
    row_data = pd.DataFrame({
        'date': date,
        'file_name': file_name,
        'time': time,
        'target_is_broken': is_broken,
        'location': location
    })
    data = pd.concat([data, row_data], ignore_index=True)

    data.to_csv()