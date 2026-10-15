# BagCount-AI

Real-time bag counting on conveyor-belt CCTV footage using YOLOv11 and ByteTrack.

A virtual line is drawn across the belt. Each bag is detected, tracked with a persistent ID, and counted once when it crosses the line. Counted bags turn red, and the annotated video is saved automatically.

## How it works

1. **Detection:** YOLOv11s trained on 654 frames I labeled myself on Roboflow (single class: `bag`).
2. **Tracking:** ByteTrack with a tuned config (`bag_tracker.yaml`) to survive fast belt motion and blurry CCTV frames.
3. **Counting:** the first frame opens so you click two points to draw the counting line. A track is counted when its center crosses the line and it has been seen in 3+ frames (filters out one-frame ghost detections).
4. **Output:** live dashboard (count, bags in view, FPS) and a saved `output.mp4`.

## Model results

Validation set of 131 images, trained for up to 150 epochs at 640px on a Kaggle T4 GPU.

| Metric | Value |
|---|---|
| mAP50 | 0.990 |
| mAP50-95 | 0.919 |
| Precision | 0.976 |
| Recall | 0.999 |

On my test clip the system counted 28 bags, which matched the manual count.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python counter.py path/to/video.mp4
```

1. A window opens with the first frame. Click two points to draw the counting line across the belt.
2. Counting starts. Press `Esc` to stop early.
3. The annotated video is saved as `output.mp4`.

To play the result on LinkedIn or in a browser, convert it to H.264:

```bash
ffmpeg -i output.mp4 -vcodec libx264 -crf 23 final.mp4
```

## Tracker settings (`bag_tracker.yaml`)

| Setting | Value | Why |
|---|---|---|
| `track_buffer` | 90 | keep lost tracks alive longer |
| `match_thresh` | 0.9 | accept less box overlap between frames (fast belt) |
| `new_track_thresh` | 0.5 | only confident detections start a new ID |

## Limitations

- The model was trained on one camera view. A different camera or angle needs new labeled frames and retraining.
- Tested on a small number of clips. Check the count against a manual count before relying on it.
- Draw the line where bags are fully visible. A line near the edge of the frame can miss bags.

## Tech stack

Python, Ultralytics YOLOv11, ByteTrack, OpenCV, Roboflow, Kaggle.
