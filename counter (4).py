import cv2, time, sys
from ultralytics import YOLO

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
OUT = "output.mp4"
model = YOLO("best.pt")
MAXW = 1280

def prep(f):
    h0, w0 = f.shape[:2]
    if w0 > MAXW:
        f = cv2.resize(f, (MAXW, int(h0 * MAXW / w0)))
    return f

cap = cv2.VideoCapture(VIDEO)
ok, first = cap.read()
if not ok:
    raise SystemExit(f"Could not read {VIDEO}. Check the file name and folder.")
first = prep(first)

# draw counting line: click 2 points on the first frame
pts = []
def click(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
        pts.append((x, y))

cv2.namedWindow("Draw line")
cv2.setMouseCallback("Draw line", click)
while len(pts) < 2:
    img = first.copy()
    for p in pts:
        cv2.circle(img, p, 6, (0, 0, 255), -1)
    cv2.putText(img, "Click 2 points for the line", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Draw line", img)
    if cv2.waitKey(20) == 27:
        raise SystemExit
cv2.destroyWindow("Draw line")
A, B = pts

def side(p):
    return (B[0]-A[0])*(p[1]-A[1]) - (B[1]-A[1])*(p[0]-A[0])

h, w = first.shape[:2]
fps_in = cap.get(5) or 30
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
if not writer.isOpened():
    raise SystemExit("VideoWriter failed to open. Try OUT = 'output.avi' with fourcc 'XVID'.")
import atexit
atexit.register(writer.release)  # always finalize the file, even if you stop early
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

MIN_HITS = 3   # a track must be seen in 3+ frames before it can be counted
last_side, counted = {}, set()
hits, crossed = {}, set()
order = {}   # tracker id -> count number shown on screen
while True:
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        break
    frame = prep(frame)
    r = model.track(frame, tracker="bag_tracker.yaml", conf=0.15,
                    persist=True, verbose=False)[0]
    live = 0
    if r.boxes.id is not None:
        live = len(r.boxes.id)
        for box, tid, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                r.boxes.id.int().cpu().tolist(),
                                r.boxes.conf.cpu().tolist()):
            x1, y1, x2, y2 = map(int, box)
            c = ((x1 + x2) // 2, (y1 + y2) // 2)
            s = side(c)
            hits[tid] = hits.get(tid, 0) + 1
            if tid in last_side and last_side[tid] * s < 0:
                crossed.add(tid)          # crossed the line
            last_side[tid] = s
            if tid in crossed and hits[tid] >= MIN_HITS and tid not in counted:
                counted.add(tid)          # real track, not a flicker
                order[tid] = len(counted)
            col = (0, 0, 255) if tid in counted else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            name = f"Bag #{order[tid]}" if tid in order else "Bag"
            cv2.putText(frame, f"{name} {cf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.circle(frame, c, 4, col, -1)

    cv2.line(frame, A, B, (0, 0, 255), 3)
    fps = 1 / max(time.time() - t0, 1e-6)
    cv2.rectangle(frame, (10, 10), (330, 130), (0, 0, 0), -1)
    cv2.putText(frame, f"BAG COUNT: {len(counted)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Live: {live}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    writer.write(frame)
    cv2.imshow("Bag counter", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print("Total bags:", len(counted))
