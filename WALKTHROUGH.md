# Read the code in execution order

0. `try_my_data.py` asks for a file and the settings, then calls `run.run()`
   with them and reads the outputs back. Everything below is the same whether
   you start there or at `run.py`.

1. `run.py: main()` parses the command; `run()` opens the recording, picks the
   preset (`PRESETS`), and walks the selected time range in back-to-back
   windows (`windows_of`). Without `--input` it builds two synthetic insects
   with known wingbeats (`synthetic_events`).

2. `event_input.py: open_recording()` chooses the decoder (faery for `.raw` and
   `.dat`) and reads the sensor size from the header. `canonical()` checks every
   packet: integer coordinates and microsecond timestamps, polarity mapped to
   -1/+1. `select_packets()` applies the time range and optional region.

3. `census.py: run_census()` is the algorithm, one window at a time:
   - `background_activity_mask` (`insect_evs/denoise.py`) keeps events that have
     a neighbour within 1 px and 1 ms: isolated sensor noise goes, insects stay.
   - `detect_blobs` (`insect_evs/detect.py`) accumulates events over the preset
     window, blurs, thresholds and finds connected blobs on a fixed time grid.
   - `link_tracks` (`insect_evs/track.py`) links detections into tracks with a
     constant-velocity prediction, a 45 px gate and the preset's coast time.
   - For each track, `PeriodicityGate.apply` (`insect_evs/gate.py`) runs
     `analyse_periodicity` (`insect_evs/periodicity.py`) on the track's events:
     the ON-minus-OFF event rate's strongest line (`f0_signed_hz`), its
     strength (`snr_signed_db`), and the period estimator (`f0_yin_hz`).
   - The octave check: candidate wingbeats are the signed line, half and a
     third of it, and the period estimate, inside the preset's band.
     `pose_autocorr` measures how strongly the animal's spatial ON/OFF event
     pattern repeats after each candidate period (`descriptor.py`), and
     `pick_fundamental` takes the shortest period that repeats about as well
     as the best one. A wing comes back to the same pose once per stroke; a
     rate count sees the two half-strokes alike and reads double.
   - `same_period_value` then picks the number for that period. When the
     signed line and the period estimate name the same period and the pose
     test cannot tell them apart, the signed line's number is reported, not
     whichever of the two reads higher.
   - `lamp_lock` measures each track's lock to the scene's mains flicker.
   - `_verdict` calls a track `wingbeat` when its line is strong (10 dB) and it
     lasts at least ten strokes.

4. Back in `run.py`, `stroke_times` finds each wing stroke from the track's ON
   events: band-passed at 0.5-1.5 times the wingbeat, peaks at least 0.75 of a
   period apart, each peak placed between the 0.5 ms bins by a parabola
   through its neighbours. `running_stroke_rate` turns those strokes into the
   changing wingbeat number, the median rate of the last five. Then the tables
   and `census_figure` are written.

Events in, tracks and frequencies out; nothing is trained and nothing is
downloaded. A box is the algorithm's hypothesis, not a label.
