# How each algorithm works, step by step

An event camera does not record frames. Each pixel reports, on its own, the
moment its brightness changes: an **event** is x, y, a time in microseconds
(us), and a polarity, ON for brighter or OFF for darker. A flapping wing makes a
burst of events every stroke. Everything below works from those events alone.

The chain runs once per processing window (default 4 s of recording):

1. noise filter
2. detection: insect-sized blobs in short time slices
3. tracking: blobs joined into one track per insect
4. wingbeat frequency of each track
5. which multiple of the stroke rate is the true wingbeat
6. the number reported
7. verdict: wingbeat or no line
8. wing strokes
9. the 5-stroke number
10. lamp lock

Values in brackets are the moth / bee / wide presets where they differ.
Code: `census.run_census` calls every step in this order.

## 1. Noise filter

`insect_evs/denoise.py`, `background_activity_mask`

Camera noise fires single pixels at random. An insect fires many neighbouring
pixels at once. The filter keeps an event only if a neighbour fired just before.

1. Keep, for every pixel, the time of its most recent event.
2. For each new event, look at the 8 pixels around it (not the pixel itself).
3. If any of them fired in the last 1 ms, keep the event. Otherwise drop it.
4. Record this event's time at its pixel, kept or not.

The pixel itself is skipped so that one faulty pixel firing again and again
cannot keep itself alive.

## 2. Detection, and the blur

`insect_evs/detect.py`, `detect_blobs`

The events are cut into short overlapping time slices and each slice is turned
into an image of where events landed.

1. Slice time into windows of 12 / 5 / 8 ms, a new one every 4 / 2.5 / 4 ms.
   The window is about one wing stroke long, so a blob holds a whole stroke.
   The window grid is fixed to multiples of the step in absolute time, so the
   same recording always gives the same windows.
2. Count the events on each pixel in the window, ON and OFF together.
3. **Blur** the count image with a Gaussian of 1 px width. Each event is
   spread over its neighbours, so events from one insect that land on adjacent
   pixels add up into one solid patch. A lone event stays faint: after the blur
   it leaves only 0.16 at its own pixel.
4. Keep the pixels where the blurred count is at least 0.5. That needs about
   three events piled on one pixel, or more spread over neighbouring pixels.
5. Join touching kept pixels, diagonals included, into blobs.
6. Throw away blobs smaller than 3 px, larger than 5% of the image, wider or
   taller than 90 px, or holding fewer than 10 events.
7. Each surviving blob is a **detection**: its time is the window's centre, its
   position is the average position of its events, and it has a box and an
   event count.

Nearby blobs are not merged: the detector cannot tell one insect seen as two
blobs from two insects flying close together.

## 3. Tracking

`insect_evs/track.py`, `link_tracks`

1. Go through the window times in order.
2. End any track that has gone unmatched for longer than the coast time
   (100 / 10 / 50 ms).
3. Predict where each live track is now: its last position plus its velocity
   times the time since.
4. List every pairing of a track with a detection within 45 px of the
   prediction. Take the closest pair first, then the next closest, using each
   track and each detection once.
5. A matched track gains the detection, and its velocity becomes half the old
   velocity plus half the latest step.
6. A detection that matched no track starts a new track with a new ID.
7. At the end, keep tracks with at least 2 detections spanning at least 5 ms.

## 4. Wingbeat frequency of a track

`insect_evs/periodicity.py`, `analyse_periodicity`, run on every track by
`insect_evs/gate.py`

1. Take the track's events: for each of its detections, the noise-filtered
   events inside that detection's box, widened by 3 px, during that
   detection's window. Windows overlap, so each event is counted once.
   Steps 5, 8 and 10 use the same boxes on the unfiltered events, because the
   noise filter can thin out parts of the wing stroke.
2. Count them in 0.2 ms bins twice: all events, and ON minus OFF. This gives
   two rate signals sampled 5,000 times a second.
3. Remove slow changes, such as the insect growing brighter as it approaches,
   with a high-pass filter.
4. **Spectral line** (`f0_signed_hz`): take the frequency spectrum of the ON
   minus OFF signal, padded 8 times for fine resolution, find the strongest
   peak in the search range, and refine it between frequency bins by fitting a
   parabola to the peak.
5. **Line strength** (`snr_signed_db`): power at that line against the median
   power elsewhere in the band, leaving out the line's own harmonics, in dB.
6. **Period estimate** (`f0_yin_hz`): the YIN method. Compare the signal with
   itself shifted by each possible delay. The first delay where the two match
   closely is one period.

## 5. Which multiple is the true wingbeat

`census.py`, `pose_autocorr` and `pick_fundamental`

A wing stroke fires a burst on the way down and another on the way back, so
the event rate often repeats twice per stroke, and step 4 reads double the
wingbeat. A mains lamp lighting the insect can add a line at three times it.
The rate alone cannot tell these apart. Where the events land can: the wing is
in the same place only once per full stroke.

1. Candidates: the spectral line, its half, its third, and the YIN estimate,
   kept only if inside the wingbeat range (18-80 / 120-320 / 15-500 Hz).
2. Draw a 6 x 6 grid over the insect, centred on its smoothed path, and count
   ON and OFF events in each cell every 0.5 ms. Remove slow trends.
3. For each candidate, measure how closely this pattern matches itself one
   candidate period later. A true period matches well.
4. Among the candidates that match within 0.05 of the best, take the shortest
   period. Twice the true period also matches, so the shortest stops a
   multiple from winning.
5. The period counts as **settled** only if it matches clearly better, by 0.05,
   than at 1.37 times that period, a delay where no wing period should repeat.

## 6. The number reported

`census.py`, `same_period_value`

The spectral line and YIN often name the same period a few percent apart.
Among candidates the step-5 test cannot separate, the spectral line's number is
reported, because it is the more precise estimate. If that line is the lamp's
own frequency, YIN's number is used instead. `fundamental_from` in
`tracks.csv` says which one supplied `f0_wingbeat_hz`.

## 7. Verdict

`census.py`, `_verdict`

A track gets `wingbeat` when all of these hold, and `no line` otherwise:

- its spectral line is at least 10 dB above the background;
- its period settled in step 5;
- the wingbeat is not within 1 Hz of 1, 2 or 3 times the lamp frequency;
- the track lasts at least 10 wing strokes.

## 8. Wing strokes

`census.py`, `stroke_times`

1. Count the track's ON events in 0.5 ms bins.
2. Remove the slow trend over two wingbeat periods.
3. Keep only frequencies between 0.5 and 1.5 times the wingbeat. Cutting
   below twice the wingbeat stops each stroke showing two peaks.
4. Every peak at least 0.75 of a period after the previous one is a stroke.
5. Place each peak between the 0.5 ms bins by fitting a parabola through it
   and its two neighbours.
6. `stroke_rate_hz` is 1 / the time since the previous stroke.

## 9. The 5-stroke number

`census.py`, `running_stroke_rate`

`rate_median5_hz` is the median of the last five stroke rates. One stroke
counted twice or missed cannot move a median of five, so the number stays
steady through those errors. It follows real changes in the wingbeat that are
slower than about four swings a second on moths, and shows them about 60 ms
late.

## 10. Lamp lock

`census.py`, `lamp_frequency` and `lamp_lock`

Mains lighting flickers at twice the mains frequency, 100 Hz in Australia, and
every insect it lights flickers with it.

1. Sum the ON minus OFF rate of the whole scene and find its strongest line
   between 95 and 105 Hz: the lamp frequency.
2. For each track, add up its events as arrows on a clock turning at the lamp
   frequency, ON events forward and OFF events backward. Events in step with
   the lamp point the same way and add up; random events cancel.
3. `lamp_R` is the length of that sum, scaled by the square root of the number
   of events. Above 2.63 the track is locked to the lamp: by chance alone that
   happens once in a thousand tracks.
