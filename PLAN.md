# Home Tower implementation plan

Start with one fixed camera in one room. The first question we want it to answer is simple: “Is anyone visible here?” Once we trust that answer, we can build up to questions like “How long did Dad watch TV?” and “When did the cat last eat?”

## How we’ll work

- [ ] means the stage is open. Tick its box after its branch has passed the checks and merged. Start the next branch from that merge.
- Before coding a stage, choose example clips, label the expected events and times, and agree on what counts as a pass. Set aside separate clips for the final check. Use household footage with permission.
- Test event rules with timestamped observations first. Then replay the reserved clips with the real models and check the full path through tracking, event storage, and the preview. Compare live and replay decisions on the same scene.
- For each branch, record missed events, false events, wrong-person assignments, and timing errors where relevant. Run earlier stages’ regression tests and JEV on the branch diff before merging.

The browser preview and saved-video processor should use the same event rules. For recordings, measure time from the video timestamps. For live input, use elapsed time. Keep local event history in SQLite so daily totals can be traced back to individual intervals. Store times and labels by default. Keep footage for testing only with the household’s agreement.

## Stage 0 — Get a dependable starting point

- [ ] `fix/video-baseline`

Confirm we have usable Git history, install the model dependencies, and locate the weights. Fix the four issues from the project review so they don’t interfere with later tests.

**Ready to merge when** the native camera preview runs past frame 100, a batch with an input/output collision is rejected before any file is written, the unclear-posture count matches labeled frames, and the browser server starts directly and as a module. Run a real-camera smoke test and record the command and result.

## Stage 1 — Notice when the room is empty

- [ ] `feature/room-presence`

Show whether someone is visible, no one is visible, or the camera is unavailable. Start with a five-second grace period after the last person disappears. Record one vacancy event when that period expires. A disconnected camera starts a coverage gap. The status applies only to the camera’s view.

**Ready to merge when** a person entering clears the vacancy, one of two people leaving keeps the room occupied, and the last person leaving produces exactly one vacancy event within two seconds after the grace period. Require zero false vacancy notices on the reserved short-occlusion and disconnect clips. Include a pet alone in the room.

## Stage 2 — Measure sitting time

- [ ] `feature/sitting-time`

Use the existing posture estimator to record sitting intervals for each visible track. Keep unclear intervals separate so hidden legs don’t inflate sitting time.

**Ready to merge when** the intervals agree with hand-labeled sit-down and stand-up times within the tolerance agreed for the clips. Include two people in different postures and a person partly hidden by furniture.

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
