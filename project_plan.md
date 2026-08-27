Yes. I’d build it in **layers**, starting with a narrow MVP rather than trying to detect all 13 UCF-Crime categories immediately.

UCF-Crime itself contains 1,900 long surveillance videos, about 128 hours, with 13 anomaly types, and was designed for anomaly detection and activity recognition. ([Visual Culture Research Center][1])

## Target architecture

```text
                     CCTV / RTSP
                         │
                         ▼
                  Video ingestion
                         │
                         ▼
               Sample/decode frames
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Object detection         Clip buffer
     person/car/object         last ~5-10 sec
             │                       │
             ▼                       │
          Tracking                   │
      person #17, #21                │
             │                       │
             └──────────┬────────────┘
                        ▼
                 Event / action model
                        │
             ┌──────────┴───────────┐
             │                      │
         Normal 0.03            Fight 0.91
                                    │
                                    ▼
                              Event engine
                                    │
                         confidence > threshold
                                    │
                                    ▼
                       Save evidence clip
                                    │
                                    ▼
                           Human operator
```

Detection + tracking is a standard real-time pattern: detections are linked across frames to persistent IDs rather than treating every frame independently. ([Ultralytics Academy][2])

## Phase 1 — Don't start with "crime detection"

Start with **one visually clear event**.

I'd choose:

**Fight / physical violence detection**

Why? It is much easier to define than things like:

```text
burglary
robbery
stealing
abuse
```

For example, seeing this:

```text
person picks up laptop
```

doesn't tell you whether they are stealing it.

But:

```text
two people
  ↓
close proximity
  ↓
rapid repeated aggressive movements
  ↓
possible fighting
```

is considerably more visually observable.

Your first classifier should therefore simply be:

```text
5-second CCTV clip
        ↓
      model
        ↓

Normal   0.07
Fight    0.93
```

### Dataset

Start with UCF-Crime:

```text
Fighting/
Normal_Videos_event/
```

But I would **not rely solely on UCF-Crime** eventually.

Its value is getting your first pipeline running. The original dataset was specifically built around weakly labelled long surveillance videos, where anomalous training videos may contain large stretches of normal activity. ([Visual Culture Research Center][1])

---

# Phase 2 — Build an offline video classifier

Before CCTV streaming, make this work:

```bash
python detect_event.py video.mp4
```

Output:

```text
00:00-00:05  normal    0.97
00:02-00:07  normal    0.94
00:04-00:09  normal    0.89
00:06-00:11  fighting  0.73
00:08-00:13  fighting  0.91
00:10-00:15  fighting  0.96
...
```

Take a video and create overlapping windows.

For example:

```text
Video
──────────────────────────────────────────────

[------5 sec------]
     [------5 sec------]
          [------5 sec------]
               [------5 sec------]
```

Maybe:

```text
clip length = 4 sec
stride      = 1 sec
```

Each clip goes into your video model.

---

# Phase 3 — Train the video-understanding model

You have several choices, but conceptually:

```text
16 / 32 / 64 frames
       ↓
Video encoder
       ↓
temporal representation
       ↓
classifier
       ↓
fight / normal
```

For a first research prototype, I would use a pretrained **video encoder** and fine-tune it rather than training from scratch.

Conceptually:

```text
frames

t1 t2 t3 t4 ... t32
│  │  │  │       │
└──┴──┴──┴───────┘
        ↓
 Video Transformer
        ↓
     embedding
        ↓
  Linear classifier
        ↓
Fight / Normal
```

The important difference from a CNN image classifier is:

```text
Image model:
one frame → features

Video model:
multiple frames → spatial + temporal features
```

---

# Phase 4 — Add real-time CCTV input

Once:

```text
video.mp4 → fight detection
```

works, replace the file with:

```text
rtsp://camera-ip/stream
```

Your program continuously reads frames.

Something conceptually like:

```python
buffer = []

while True:
    frame = camera.read()

    buffer.append(frame)

    if enough_frames(buffer):
        clip = make_clip(buffer)

        prediction = model(clip)

        if prediction["fight"] > threshold:
            create_alert()
```

But don't let the buffer grow indefinitely.

You maintain a **rolling window**:

```text
t=0

[1 2 3 ... 32]

t=1

[9 10 11 ... 40]

t=2

[17 18 19 ... 48]
```

So you are continuously asking:

> What is happening during the last few seconds?

---

# Phase 5 — Add person detection

Now add something like a YOLO-style detector.

```text
frame
   ↓
detector
   ↓

person  bbox=(120,80,250,470)
person  bbox=(300,100,420,475)
car     bbox=(...)
```

You probably don't want the video model analyzing huge areas of irrelevant scene.

Instead:

```text
┌────────────────────────────────────┐
│                                    │
│       Person A       Person B      │
│          □              □          │
│         /|\            /|\         │
│         / \            / \         │
│                                    │
│       parked cars                  │
└────────────────────────────────────┘
```

Crop around the relevant people.

Then:

```text
person interaction region
          ↓
      clip model
          ↓
      fight = 0.94
```

---

# Phase 6 — Add tracking

Now use ByteTrack / BoT-SORT-like tracking.

Tracking turns:

```text
Frame 100:
person
person

Frame 101:
person
person
```

into:

```text
Frame 100:
Person #12
Person #19

Frame 101:
Person #12
Person #19
```

Persistent IDs are what let you reason about movements and trajectories across frames. ([Ultralytics Academy][2])

You can now ask interesting questions:

```text
Are #12 and #19 approaching each other?

How long have they been interacting?

Did #12 suddenly fall?

Did #19 run away?

Has #12 entered a restricted region?
```

---

# Phase 7 — Don't classify every frame

This matters a lot for performance.

Imagine:

```text
CCTV = 30 FPS

1 camera:
30 frames/sec

100 cameras:
3000 frames/sec
```

Running every expensive model on every frame becomes wasteful.

Instead, you can do something like:

```text
Video stream:        30 FPS

Detector:            10 FPS

Tracker:             10 FPS

Action model:        every ~0.5-1 sec

Event decision:      rolling
```

Tracking fills in the gaps.

---

# Phase 8 — Build an event engine

This part is extremely important.

Do NOT do:

```python
if fight_probability > 0.5:
    alert()
```

Otherwise you'll spam operators.

Instead:

```text
t=0    fight = 0.40
t=1    fight = 0.71
t=2    fight = 0.90
t=3    fight = 0.94
t=4    fight = 0.91
```

Then maybe:

```text
IF fight score > 0.80
FOR >= 2 seconds

→ generate alert
```

And implement cooldown:

```text
Person #17 + #21
already generated fight alert

don't generate another
for next 30 sec
```

So:

```text
raw model predictions
        ↓
temporal smoothing
        ↓
thresholding
        ↓
persistence rules
        ↓
deduplication
        ↓
EVENT
```

This logic will have a surprisingly large impact on the actual product.

---

# Phase 9 — Save evidence

When an event triggers, don't just save a screenshot.

Maintain a circular video buffer:

```text
             ALERT
               ↓
─────10 sec─────┼─────10 sec─────
    BEFORE      │      AFTER
```

Save:

```json
{
  "event_id": "CAM12-20260827-142304",
  "camera_id": "CAM12",
  "event": "possible_fight",
  "confidence": 0.93,
  "start_time": "14:22:58",
  "alert_time": "14:23:04",
  "people": [17, 21],
  "clip": "event_23981.mp4"
}
```

An operator opens:

```text
⚠ Possible Fight
Camera 12
Confidence: 93%

[10 sec before + event + 10 sec after]

         PLAY VIDEO

[Confirm] [False Alarm]
```

That human feedback becomes extremely valuable later.

---

# Phase 10 — Expand beyond fighting

Once the infrastructure works, don't build a single 14-way classifier immediately.

Add modules.

For example:

```text
                    CCTV
                      │
       ┌──────────────┼───────────────┐
       ▼              ▼               ▼
    people         vehicles        objects
       │              │               │
       ▼              ▼               ▼
    tracker         tracker         tracker
       │
       ├── Fight detector
       ├── Fall detector
       ├── Running detector
       ├── Loitering detector
       ├── Restricted-zone detector
       ├── Crowd detector
       └── Abandoned-object detector
```

Then combine them.

---

# Phase 11 — Build crimes from lower-level events

This is where it gets interesting.

Instead of directly predicting:

```text
BURGLARY
```

you can represent:

```text
Person #47
   ↓
enters restricted property
   ↓
at unusual time
   ↓
interacts with door/window
   ↓
enters building
   ↓
possible intrusion
```

Similarly:

### Possible robbery

```text
Person A approaches B
       ↓
aggressive interaction
       ↓
object transferred
       ↓
Person B falls / retreats
       ↓
Person A runs away
       ↓
possible robbery
```

### Shoplifting

```text
Person enters store
       ↓
interacts with product
       ↓
product disappears
       ↓
person exits
       ↓
no corresponding checkout event
       ↓
possible shoplifting
```

Notice how much richer this is than:

```text
video → shoplifting
```

---

# Phase 12 — Add general anomaly detection

Eventually you want the system to detect things you didn't explicitly train.

You can have two branches:

```text
                   CCTV
                     │
             Video encoder
                     │
             ┌───────┴────────┐
             ▼                ▼
     Known-event model    Anomaly model
             │                │
         fighting          unusual?
         accident             │
         vandalism            │
             │                │
             └───────┬────────┘
                     ▼
                  Alerts
```

This is closer to the original UCF-Crime philosophy: assign high anomaly scores to suspicious segments rather than requiring every anomaly to fit one exact class. ([Visual Culture Research Center][1])

---

# Phase 13 — Train with your own CCTV footage

This is probably the most important long-term step.

UCF-Crime footage might look very different from your deployment:

```text
UCF-Crime
    ↓
different country
different cameras
different resolution
different camera angle
different lighting
different crowd density
different compression
```

Your actual cameras:

```text
1920×1080
camera mounted 7m high
India
nighttime lighting
rain
crowded market
specific building
etc.
```

That is **domain shift**.

Eventually build:

```text
Your_CCTV_Dataset/

normal/
fight/
fall/
intrusion/
vandalism/
accident/
...
```

Especially collect a lot of **hard negatives**:

```text
hugging
dancing
children playing
people arguing
people running
workers carrying objects
sports
crowded movement
```

Otherwise your fight detector will fire constantly.

---

# Phase 14 — Evaluation

Do not primarily optimize:

```text
accuracy = 94%
```

For CCTV, more useful metrics include:

### Event recall

Of 100 actual fights:

```text
how many did we detect?
```

### False alarms

```text
false alerts / camera / hour
```

This is critical.

Suppose:

```text
99% accuracy
```

sounds great.

But if you have:

```text
500 cameras × 24 hours
```

even a tiny false-positive rate can create thousands of useless alerts.

I would track:

```text
Event Recall
Event Precision
F1
False alarms/hour/camera
Detection latency
Average confidence
Missed-event rate
GPU utilization
FPS
```

---

# Phase 15 — Production architecture

Eventually:

```text
        Camera 1 ──┐
        Camera 2 ──┤
        Camera 3 ──┤
        Camera 4 ──┤
                   ▼
             Stream Gateway
                   │
                   ▼
             GPU Inference
             ┌─────┴─────┐
             │           │
        detection     video model
             │           │
             └─────┬─────┘
                   ▼
             Event Engine
                   │
       ┌───────────┼────────────┐
       ▼           ▼            ▼
    Database   Clip Storage   Alert Queue
                                 │
                                 ▼
                          Control Room UI
```

You can put inference:

```text
Camera → Edge GPU → events → server
```

or:

```text
Camera → central GPU cluster → events
```

RTSP ingestion and frame-by-frame streaming are common approaches; live/long streams should be processed incrementally rather than loading the whole video into memory. ([Ultralytics Academy][2])

---

# What I would actually build first

Forget the full architecture initially.

### MVP 1

```text
UCF-Crime
    ↓
extract Fight + Normal clips
    ↓
pretrained video model
    ↓
fine-tune
    ↓
video.mp4
    ↓
Fight / Normal + timestamps
```

**Goal:**

```text
$ python inference.py fight_video.mp4

00:00-00:04 NORMAL 0.98
00:02-00:06 NORMAL 0.91
00:04-00:08 FIGHT  0.79
00:06-00:10 FIGHT  0.94
00:08-00:12 FIGHT  0.96
```

### MVP 2

Replace:

```text
video.mp4
```

with:

```text
webcam / RTSP
```

and show:

```text
LIVE CAMERA

┌─────────────────────────────┐
│                             │
│    □ Person 12              │
│            □ Person 17      │
│                             │
│                             │
│  POSSIBLE FIGHT — 91%       │
└─────────────────────────────┘
```

### MVP 3

Add:

```text
YOLO
+
ByteTrack/BoT-SORT
+
video classifier
+
event persistence
+
save 20-sec clip
+
simple alert dashboard
```

At that point you have an **actual end-to-end CCTV intelligence prototype**, rather than just a trained ML model.

---

## Then expand in this order

| Stage | Capability                 | Difficulty |
| ----- | -------------------------- | ---------: |
| 1     | Fight detection            |         ★★ |
| 2     | Fall detection             |         ★★ |
| 3     | Restricted-area intrusion  |         ★★ |
| 4     | Loitering                  |         ★★ |
| 5     | Road accident              |        ★★★ |
| 6     | Vandalism                  |        ★★★ |
| 7     | Abandoned object           |        ★★★ |
| 8     | Weapon-visible alert       |        ★★★ |
| 9     | Stealing/shoplifting       |       ★★★★ |
| 10    | Robbery                    |       ★★★★ |
| 11    | Burglary                   |       ★★★★ |
| 12    | General anomalous behavior |      ★★★★★ |

I would **not start with UCF-Crime 13-class classification**. Start with one event, solve streaming + temporal inference + false alarms + alerts, and only then broaden the semantic understanding. That way most of what you build—RTSP ingestion, buffers, tracking, inference service, event engine, clip storage and dashboard—remains reusable as you add new behaviors.

[1]: https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/?utm_source=chatgpt.com "Real-world Anomaly Detection in Surveillance Videos – Center for Research in Computer Vision"
[2]: https://academy.ultralytics.com/courses/train-your-first-yolo/video-inference?utm_source=chatgpt.com "Run Inference on Video | Train your first YOLO model | Ultralytics Academy"
