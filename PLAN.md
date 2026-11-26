# Home Tower implementation plan

Start with one fixed camera in one room. The first question we want it to answer is simple: “Is anyone visible here?” Once we trust that answer, we can build up to questions like “How long did Dad watch TV?” and “When did the cat last eat?”

## How we’ll work

- [ ] means the stage is open. Tick its box after its branch has passed the checks and merged. Start the next branch from that merge.
- Before coding a stage, choose example clips, label the expected events and times, and agree on what counts as a pass. Set aside separate clips for the final check. Use household footage with permission.
- Test event rules with timestamped observations first. Then replay the reserved clips with the real models and check the full path through tracking, event storage, and the preview. Compare live and replay decisions on the same scene.
- For each branch, record missed events, false events, wrong-person assignments, and timing errors where relevant. Run earlier stages’ regression tests and JEV on the branch diff before merging.

The browser preview and saved-video processor should use the same event rules. For recordings, measure time from the video timestamps. For live input, use elapsed time. Keep local event history in SQLite so daily totals can be traced back to individual intervals. Store times and labels by default. Keep footage for testing only with the household’s agreement.

## Stage 0 — Get a dependable starting point

- [x] `fix/video-baseline`

Confirm we have usable Git history, install the model dependencies, and locate the weights. Fix the four issues from the project review so they don’t interfere with later tests.

**Ready to merge when** the native camera preview runs past frame 100, a batch with an input/output collision is rejected before any file is written, the unclear-posture count matches labeled frames, and the browser server starts directly and as a module. Run a real-camera smoke test and record the command and result.

**Branch verification.** On Python 3.10.16 with CPU inference, all 13 regression tests passed. A simulated camera ran through frame 101. A five-frame excerpt from `videos/v1.mp4` produced five matched poses with five unclear labels during the posture hold, and the JSON report counted five unclear frames. Both server launch modes served the page and returned an annotated JPEG using the real detector and pose weights.

To repeat the checks on Linux, install FFmpeg with its H.264 encoder, then run these commands from the `t1` checkout. The tested runtime uses the official Ultralytics v8.4.0 release weights in the sibling `weights` directory.

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python --torch-backend cpu ultralytics==8.4.216 scipy==1.15.3 lap==0.5.14 pytest
mkdir -p ../weights
for model in yolo26s yolo11n-pose; do
  curl --fail --location --output "../weights/$model.pt" "https://github.com/ultralytics/assets/releases/download/v8.4.0/$model.pt"
done
.venv/bin/python -m pytest -q tests/test_video_baseline.py
```

**Accepted development check.** The household tested the live laptop webcam through the browser. The final baseline suite passed 17 tests, including 101 simulated native-camera frames. We agreed to defer the physical native-camera check to production camera commissioning. At that point, verify the real feed continues updating past frame 100.

## Stage 1 — Notice when the room is empty

- [x] `feature/room-presence`

Show whether someone is visible, no one is visible, or the camera is unavailable. Start with a five-second grace period after the last person disappears. Record one vacancy event when that period expires. A disconnected camera starts a coverage gap. The status applies only to the camera’s view.

**Ready to merge when** a person entering clears the vacancy, one of two people leaving keeps the room occupied, and the last person leaving produces exactly one vacancy event within two seconds after the grace period. Require zero false vacancy notices on the reserved short-occlusion and disconnect clips. Include a pet alone in the room.

**Development verification.** The live preview reports presence for the camera’s field of view. It holds the occupied status through brief occlusions and records one vacancy after five continuous seconds of empty observations. The server writes UTC state changes to local SQLite. Use `--history PATH` to select the file and `--vacancy-grace SECONDS` to change the delay. Saved-video reports include the transitions in `room_presence.events` using seconds from the recording.

The real-model replay used a two-person image, a single-person video frame, and a pet-only image. With one-second samples, the last person left at second 5, the vacancy started at second 10, and the person returned at second 12. There were zero false or missed vacancies in this sequence. The camera backend agreed with the replay and reported a coverage gap on disconnect. All 24 automated tests passed. The household confirmed that room presence worked in the live laptop preview and accepted Stage 1.

## Stage 2 — Measure sitting time

- [ ] `feature/sitting-time`

Use the existing posture estimator to record sitting intervals for each visible track. Keep unclear intervals separate so hidden legs don’t inflate sitting time.

**Ready to merge when** the intervals agree with hand-labeled sit-down and stand-up times within the tolerance agreed for the clips. Include two people in different postures and a person partly hidden by furniture.

**Clip selection in progress.** Use `videos/v6.mp4` for development. At half-second sampling, the person lowers onto the sofa around 7.5–8.0 seconds and is visibly seated by 8.0 seconds. The cushion partly obscures the lower body. Reserve `videos/v4.mp4` as a separate check for someone already sitting at the start. An independent clip with a visible stand-up and two people in different postures is still needed for the final check.

**Proposed timing tolerance.** Allow one second on either side of a hand-labeled transition. Report missed or extra intervals separately. Exclude time labeled unclear from sitting totals.

**Posture diagnosis.** In the full-frame-rate model replay of `v4.mp4`, 83 of 229 frames counted as sitting (2.77 of 7.64 seconds). During the 1.90–4.40 second gap both predicted knees passed the confidence cutoff on all 75 frames. At 3.00 seconds the knee markers landed near the elbows, with the cushion obscuring the lap. Lowering the cutoff would not fix this error. In `v6.mp4` the first sitting label appeared at 7.68 seconds, and the model counted 12.04 seconds seated before the 21.00-second ending.

**Stand-up check.** Reprocessing every third frame of the existing laptop preview recording at 10 samples per second yielded 2.20 seconds labeled sitting. The last sitting interval ended at 20.30 seconds, during the visible rise (about 20.2–21.0 seconds). This recording includes the original preview overlays. The same sampling on the walking clip `v3.mp4` produced zero sitting labels. A matched recording of the cushion scene from a wider, slightly side-on view with the legs visible is still needed to measure the effect of camera placement.

**False sitting while lying down.** In `videos/v10.mp4` two people lie under a duvet for the whole 20.56 seconds. The pipeline labeled 162 of 514 frames sitting (159 on one track ID, 3 on another) and none lying down. It tracked the two people as one in 430 frames, but the merged box does not cause the label: sitting persisted in 62 of the 83 frames where they had separate boxes. The pose model places one knee beside the hip under the duvet, and the flat-thigh rule in `posture.py` accepts that knee as sitting with no lower leg in view. All 352 raw sitting calls came from that rule. The lying rule cannot fire because the torso points up the image. `v4.mp4` earns its sitting labels from the same evidence, a guessed knee with the legs out of frame. Requiring a visible lower leg (a confident ankle below the knee) removed all false sitting in `v10.mp4`, but it also removed all 83 sitting frames in `v4.mp4` and moved the first sitting label in `v6.mp4` from 7.68 to 12.0 seconds. The code is unchanged.

**Pose-confidence rule rejected (25 November 2026).** On the development clips, counting a flat thigh as sitting only when the matched pose box confidence is at least 0.70 removed all 162 false frames in `v10.mp4` and kept every sitting frame in `v4.mp4` and `v6.mp4`. A crop of `v10.mp4` about 2.7 times closer still left 13 false frames. We then tested the fixed 0.70 threshold on 18 clips it had never been run on: 16 from Mixkit and the two 5-second UP-Fall views in `videos/up-fall/S1-A8-T1`. Clips were chosen by title and thumbnail. Lying, seated, sit-down and stand-up times were hand labeled from timestamped frames before either version ran. Both versions ran through the saved-video pipeline at `ed33d4a` on identical model outputs.

The rule made no useful difference, so do not adopt it. False sitting while lying under covers stayed at 4.75 of 93.25 seconds. Sitting kept went from 38.13 to 37.96 of 105.52 seated seconds. Sit-down timing was identical in every clip. The poses behind the false labels were confident, with a median pose confidence of 0.84 and 0.87 in the two night-time couple clips.

| Lying under covers | Scene | Lying (s) | False sitting, current code (s) | With 0.70 rule (s) |
|---|---|---|---|---|
| Mixkit 3207 | Couple, bright, close overhead view | 10.83 | 0 | 0 |
| Mixkit 42981 | Couple at night, overhead | 9.59 | 1.54 | 1.54 |
| Mixkit 42972 | Couple on their sides at night | 7.55 | 1.92 | 1.92 |
| Mixkit 31414 | Man, low side view, tosses the duvet | 13.18 | 0 | 0 |
| Mixkit 50754 | Man, before he sits up at 7.2 s | 7.20 | 0 | 0 |
| Mixkit 31399 | Woman under a duvet, before she sits up at 8.3 s | 8.30 | 1.29 | 1.29 |
| Mixkit 48733 | Man on his back at night | 36.60 | 0 | 0 |

| Seated | Scene | Seated (s) | Sitting kept, current code (s) | With 0.70 rule (s) | Sit-down: labeled → first sitting label (s) |
|---|---|---|---|---|---|
| UP-Fall camera 1 | Frontal, full body on a chair | 5.00 | 1.68 | 1.68 | none in clip |
| UP-Fall camera 2 | Side view, second seated person behind | 5.00 | 4.68 | 4.68 | none in clip |
| Mixkit 15774 | Armchair, legs crossed | 7.57 | 7.27 | 7.27 | none in clip |
| Mixkit 12837 | Park bench, frontal, shins in view | 10.60 | 0 | 0 | none in clip |
| Mixkit 250 | Couch, lap and legs out of frame | 14.18 | 0 | 0 | none in clip |
| Mixkit 48751 | Armchair with a blanket, camera pushes in | 23.00 | 9.72 | 9.72 | none in clip |
| Mixkit 12866 | Rises from an armchair and sits again | 3.46 | 1.98 | 1.98 | 6.4 → 7.12 (+0.72) |
| Mixkit 50764 | Sits on a sofa with legs stretched out | 7.14 | 6.92 | 6.92 | 2.2 → 2.42 (+0.22) |
| Mixkit 26083 | Sits at a table, legs hidden | 4.18 | 0 | 0 | 7.0 → never |
| Mixkit 29328 | Sits behind a pizza box, head out of frame | 8.45 | 5.88 | 5.71 | 5.4 → 5.71 (+0.31) |
| Mixkit 16156 | Sits on the ground, seen from behind | 16.94 | 0 | 0 | 11.1 → 10.62, three frames during the squat, then never |

Other results from the same runs, identical in both versions. In Mixkit 12866 the first sitting interval ended at 1.12 seconds against a labeled stand-up at 1.1 seconds. No frame was labeled sitting during the labeled standing intervals in Mixkit 12866, 26083 and 16156. In Mixkit 48751 the hips leave the frame at about 9.6 seconds, and 9.28 of those 9.6 seconds were labeled sitting. The frontal bench clip Mixkit 12837 had 22 frames labeled standing while the man sat. Lying down was labeled in Mixkit 31414, 50754 and 48733, and for five frames in Mixkit 31399 after the woman had sat up.

How the totals are counted. They add the 18 clip views, and every labeled second is covered by processed frames: 93.25 lying seconds and 105.52 seated seconds, matching `stage2_checks/labels.json`. A frame counts as sitting when any track is labeled sitting, so it is counted once however many people are in view. The two UP-Fall files are simultaneous camera views of one participant for the same 5 seconds. They are scored as two clips and add 10.00 seated seconds and 6.36 kept seconds for one 5-second performance. With the pair counted once, the seated total is 100.52 seconds, with 33.45 kept using camera 1 or 36.45 using camera 2 (33.28 or 36.28 with the rule). In camera 2 the participant accounts for all 82 sitting frames. The second seated person was labeled sitting on 67 of those same frames and adds no time.

Scored afterwards on the same recordings, and not part of the comparison planned before the runs, the visible-lower-leg rule removed all 114 false sitting frames in the lying clips. It cut Mixkit 29328 from 141 to 5 sitting frames and Mixkit 48751 from 243 to 186, and left the other seated clips unchanged. These figures came from full joint traces that were not kept, so `stage2_checks` does not reproduce them. The rule still needs its own test on fresh labeled clips before any decision.

To reproduce the tables, run `python stage2_checks/score.py --check`. The folder holds the hand labels, the per-frame labels from both versions, the clip checksums and a script that reruns both versions. The Mixkit clips are in `videos/stage2-unseen/` and download from `https://assets.mixkit.co/videos/ID/ID-720.mp4`. Videos stay on this machine and out of the repository. Stage 2 stays open: the false sitting under covers, the missed sitting behind tables and in frontal views, and the two-person check remain unresolved.

**Per-track interval recording (25 November 2026).** Saved-video processing and the browser preview now share one recorder, `PostureIntervals` in `posture.py`. It turns each track's final posture labels into sitting and unclear intervals, using the recording's timestamps for videos and elapsed time for live input. The README explains the counting rules, the `sitting_time_by_track_id` report section and the `posture_intervals` history table. The posture labels themselves are unchanged: `run_versions.py --check upfall_cam1` reproduced the saved labels exactly. Timestamped tests with hand-worked totals cover two simultaneous tracks in different postures, sitting → unclear → sitting, irregular frame spacing, a disappearance, a disconnect, a new session, a reused display ID and the end of a video. The same observations fed through the saved-video pipeline and the preview server produce identical intervals.

Real-model replays at full frame rate. Each total equals the sum of its intervals.

| Clip | Hand label | Sitting intervals (s) | Sitting | Unclear |
|---|---|---|---|---|
| `v6.mp4`, 21.00 s, one track | Lowers onto the sofa 7.5–8.0 s, seated to the end | 7.68–8.24, 9.08–10.08, 10.52–21.00 | 12.04 s | 4.52 s: 4.44–7.68, 8.24–9.08, 10.08–10.52 |
| `v4.mp4`, 7.64 s, one track | Seated throughout | 0.30–0.83, 1.17–1.50, 1.83–1.90, 4.40–6.11, 6.44–6.57 | 2.77 s | 4.87 s, longest 1.90–4.40 |
| `v10.mp4`, 20.56 s, five tracks | Two people lying under a duvet throughout | Track 1: 1.36–1.84, 3.84–4.28, 5.04–5.16, 5.80–6.00, 6.36–8.52, 10.36–11.44, 13.20–13.72, 15.32–16.68. Track 8: 20.44–20.56 | 6.48 s, all false | 17.12 s |

In `v6.mp4` the first sitting label falls inside the one-second tolerance of the sit-down. Of the 13.0 seconds seated after 8.0 seconds, 11.72 count as sitting. The two unclear spans while seated (1.28 seconds) split one hand-labeled interval into three, which counts as two extra intervals under the proposed scoring. `v4.mp4` keeps 2.77 of 7.64 seated seconds in five pieces. In `v10.mp4` the known false labels become 6.48 seconds of sitting in nine intervals: 6.36 seconds on track 1, which holds both people for most of the clip, and 0.12 seconds on track 8 in the last frames. The longest false interval is 2.16 seconds. Six of the nine are shorter than 0.6 seconds, but `v4.mp4` also has a genuine 0.07-second interval, so a minimum duration would discard real sitting as well.

Live check. `v6.mp4` was sent to the real preview server as JPEG frames at the video's own pace, as a browser camera would send them. The server processed 100 of 525 frames (4.7 per second) and recorded 11.72 seconds of sitting in three intervals and 3.94 seconds unclear, against 12.04 and 4.52 seconds in the replay. When the frames stopped, the server's idle check closed the intervals at the last processed frame and saved them to SQLite, where the totals matched. The presence history also logged a coverage gap after the first frame. This is existing behavior: the first inference most likely took more than 3 seconds to warm up. No physical webcam or browser was available for this check. The independent final-check clip with a visible stand-up and two people in different postures is still needed.

**Two-person check (26 November 2026).** No local clip had two seated people with one standing up: `v9.mp4` has three people seated throughout, and the rest show one person or two people lying down. The test used Pexels video 4008311, a continuous 24.84-second shot. Woman A, in a yellow T-shirt on the left, stays seated throughout. Woman B, in a white hoodie on the right, leaves the seat at 6.7 seconds, walks past the camera, is out of view from 7.5 to 9.8 seconds, and then stands in the kitchen. The labels and the rule assigning boxes to people were written before any model ran. The files are in `stage2_checks/two_person*`, and `python stage2_checks/timer_check.py --labels two_person_labels.json --check` reproduces the result.

The check fails at posture labeling, so it does not yet test the timers.

| Person | Track IDs | Seated (labeled) | Counted as sitting | Where the seated time went |
|---|---|---|---|---|
| A | 1, all 621 frames | 0–24.84 s | 0 s | Unclear 24.80 s, no matched pose 0.04 s |
| B | 2 until 7.52 s, then 5 from 9.84 s | 0–6.70 s | 0 s | Unclear 6.72 s; the frame at 6.68 s lasts until 6.72 s |

Detection and tracking held. Neither track ever moved to the other person, a box covering both women at 7.44 seconds stayed untracked for that one frame, and no sitting time went to the wrong person or to a merged box. B was absent from 7.60 to 9.76 seconds, matching her labeled time out of view, and track 5 labeled her standing on 303 of its 375 frames. The recorded intervals match the per-frame labels exactly. B's timer never started, so its stop against the ±1-second tolerance could not be measured. A's timer did not keep counting through the stand-up because it never ran.

The cause is camera framing. The bottom of the frame cuts through both women's laps, and every seated box reaches it. No seated frame has a knee at the 0.30 joint cutoff: the median best knee confidence is 0.03 for A and 0.02 for B, and the median best hip is 0.16 for both. Both sitting rules need a knee, so the estimator returns unclear by design. This is the same pattern as Mixkit 250, where the lap and legs are out of frame. The two-person timer question still needs a clip that shows both people's knees, framed like the proposed wider camera placement.

**One-person timer checks (26 November 2026).** No public clip qualified for a two-person check with hips and knees in view. Of Pexels 7969387, 8484797 and 9354153, none keeps a seated person's hips and knees visible through the other person's sit-down. The user then supplied three local clips, kept on this machine only. Each shows one person, so together they test one timer at a time. vv2 covers A's role of sitting without a break, vv1 the stop at a stand-up, and vv3 the start at a sit-down. They do not answer the two-person questions of A continuing to count while B moves, or time going to the wrong person. Both two-person checks stay open. The labels were written before any model ran, and `python stage2_checks/timer_check.py --labels one_person_vv1_labels.json --check` (and likewise for `vv2` and `vv3`) reproduces each result.

| Clip | Labeled | Timer | Error | Sitting counted | Lost | Gained | Track IDs |
|---|---|---|---|---|---|---|---|
| vv1, 36.32 s, man in silhouette with a cane | Seated 0–11.5 s, stand-up at 11.5 s | Sitting 0.32–12.36 s | Stops +0.86 s | 11.20 of 11.50 s | 0.32 s, the 0.3-second hold at the start | 0.84 s while rising | 1 throughout; no box for two frames at 12.84–12.88 s, bent over the cane |
| vv2, 10.92 s, man seated with a white cane | Seated throughout | Sitting 0.32–10.92 s | No transition | 10.60 of 10.92 s | 0.32 s, the hold at the start | 0 s | 1 on all 273 frames |
| vv3, 15.70 s, man sits astride a chair | Walks in at 4.0 s, sit-down at 8.9 s | Sitting 8.63–8.68 s, then 9.22–12.97 s | First sitting frame −0.27 s; sustained sitting +0.32 s | 3.75 of 6.80 s | 3.00 s: unclear 2.67 s from 12.97 s to the end, hold 0.33 s | 0.04 s, one frame during the descent | 1 from 4.42 s; the first detected frame was untracked |

Every timer moved within the one-second tolerance, and no clip had an ID change. The recorded intervals match the per-frame labels in all three. Hips and knees were confident in every seated frame: the median best confidence was 0.99 or higher.

Two estimator behaviors account for the timing. In `vv1.mp4` the label stays sitting for 0.86 seconds after his hips leave the seat, because he rises slowly while bent forward with his thighs still level. That is within the tolerance but close to its edge. In `vv3.mp4`, seen from the front while he sits astride the chair, each knee drops 0.48–0.79 of the thigh's length below the hip in the frames sampled after he sits. The flat-thigh rule calls sitting only up to 0.60, and the shortened-thigh rule needs at least 0.75 on both sides. From 12.97 seconds he leans on the chair back, the left thigh measures 0.61–0.64 and the right 0.73–0.79, so neither rule applies and 2.67 seated seconds are labeled unclear. The same rule fired once during his descent at 8.63 seconds, when both thighs measured 0.81 with his hips still above the seat. The estimator is unchanged; these are findings for the accuracy review.

## Stage 3 — Assign activity to household members

- [ ] `feature/household-profiles`

Let an adult create named profiles for parents, grandparents, and children, assign a visible track, and correct a mistaken assignment. A temporary track ID is valid only for that track. Leave uncertain activity unassigned and exclude it from individual totals.

**Ready to merge when** two people crossing paths, someone leaving and returning, and a restart all preserve correct attribution or leave it open for review. Correct an assignment and verify the affected person’s activity history.

**Optional follow-up branch** `feature/returning-member-recognition`. Take this up separately if the household needs hands-free identification. Test unfamiliar visitors, similar-looking relatives, poor lighting, and children with guardian approval. Require zero false family assignments on the agreed holdout clips and report how often recognition leaves someone unassigned.

## Stage 4 — Notice a new arrival

- [ ] `feature/person-arrivals`

Define a doorway in the camera view. Record a person entering through it. If the arrival has no household assignment, show it for review and let the household mark a visitor.

**Ready to merge when** a returning resident, a visitor, and two people entering together produce the expected arrivals. Walking past the doorway or picking up a fresh track ID after an occlusion must produce zero arrivals.

## Stage 5 — Estimate TV watch time

- [ ] `feature/tv-watch-time`

Count viewing time for each assigned person while the TV is on and the person remains oriented toward it. Mark intervals uncertain when the camera cannot establish the TV state or viewing direction.

**Ready to merge when** the TV being off, an empty room, and a person facing away all contribute zero watch time on labeled clips. Check two viewers watching together, then one leaving while the other stays. Compare each person’s total with the labels.

## Stage 6 — Estimate sleep and wake times

- [ ] `feature/sleep-wake`

For an assigned person in an agreed sleeping area, estimate sleep onset, wake time, overnight duration, and naps. Keep uncertainty visible when quiet wakefulness looks like sleep.

**Ready to merge when** estimates meet the agreed timing tolerance against a household sleep log. Include an empty bed, resting awake, a short trip out of bed, a nap, a night spanning midnight, and loss of camera coverage.

## Stage 7 — Record meal routines

- [ ] `feature/meal-routines`

Record when a person appears to eat, how long the meal lasts, and how often meals occur. Allow confirmation of an ambiguous event. Food type and portion size would need additional evidence.

**Ready to merge when** meal times and counts agree with the household meal log within the agreed tolerance. Include two diners, cooking, sitting at the table without eating, and a camera gap during a meal. For meals, sleep, and TV viewing, compare two people’s independent labels before setting timing tolerances.

## Stage 8 — Track pet activity

- [ ] `feature/pet-activity`

Support the household’s pet species and record visible movement and rest. Keep human occupancy independent of pet activity. Attribute activity to an individual pet only when its assignment is reliable.

**Ready to merge when** the system handles a pet alone, two pets crossing, a person carrying a pet, and a pet leaving the view. Check the activity intervals, pet assignments, and human occupancy for each clip.

## Stage 9 — Record pet feeding times

- [ ] `feature/pet-feeding`

Mark the bowl area and look for sustained feeding behavior. Record the visit time and duration so the household can see when feeding usually happens.

**Ready to merge when** visit counts and times agree with hand labels. Include walking past the bowl, a person refilling it, drinking nearby, two pets sharing the area, and one pet returning for another visit.

## Stage 10 — Estimate pet sleep time

- [ ] `feature/pet-sleep`

Record rest intervals and estimate sleep where the camera view supports it. End observed coverage when the pet disappears from view.

**Ready to merge when** labeled examples cover quiet wakefulness, sleep, changing position, leaving the bed, and disappearing behind furniture. Verify that time out of view adds zero observed sleep.

## Stage 11 — Flag activity for review

- [ ] `feature/unclassified-activity`

When visible activity continues and none of the enabled features can explain it, create one review item with the room, time, and reason. An unfamiliar activity and an unassigned person are separate observations.

**Ready to merge when** sustained unfamiliar activity creates one item per incident. Check ordinary movement, a lighting change, and camera movement for false alerts.

## Stage 12 — Put the day together

- [ ] `feature/daily-summary`

Show sitting, TV viewing, sleep, and meal routines by household member, along with pet activity, feeding, and sleep. Show gaps in coverage alongside the totals so the household can tell how much of the day was observed.

**Ready to merge when** every total can be reconciled with the underlying events. Check a night crossing midnight, overlapping activities such as sitting while watching TV, a corrected profile, a restart, and replaying the same recording twice. A replay must leave the totals unchanged.
