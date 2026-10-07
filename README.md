# Insect tracking and wingbeat starter

Give it an event-camera recording; it finds the flying insects, tracks them,
and measures each one's wingbeat, as a per-track frequency and as the time of
every wing stroke.

This is the code that produced the Ecology project's Bogong moth results and
that its bee settings use. It is copied from the research code by
`analysis/export_starter.py`, not rewritten. `PROVENANCE.json` names the
source commit (this build: `aff11f0`, 8 October 2026) and the hash of every
copied file. On the moth recording and on bee data it gives the research
code's own answers, track for track and stroke for stroke. That check
(`analysis/starter_equivalence.py`) lives in the research repository, which
is private because the data cannot be shipped.

## Install once

Unzip the folder and open a terminal inside it. Windows, Python 3.9 or newer:

```powershell
py -3.9 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-raw.txt
```

On macOS or Linux use `python3 -m venv .venv` and `.venv/bin/python`. For
NPZ/CSV input only, `requirements.txt` is enough; for HDF5 add `h5py`.

`numba` is required, not optional: without it the noise filter falls back to
an approximation and the results stop matching the research code.

## Try it on your own recording

```powershell
.venv\Scripts\python.exe try_my_data.py
```

It opens a file dialog (or pass the path: `try_my_data.py recording.raw`;
`--no-dialog` to type it), asks which settings to use (moth or bee), where to
start and how many seconds to process, then runs everything and prints the
wingbeat of the longest tracks. Every question has a default in brackets;
press Enter to take it. Outputs go to `outputs/NAME_SETTINGS_STARTs/`. It runs
exactly the code below; `run.py` is the same thing driven by flags.

## Run

```powershell
.venv\Scripts\python.exe run.py --input "recording.raw" --preset moth --start 10 --duration 8 --out moth_out
.venv\Scripts\python.exe run.py --input "bees.raw" --preset bee --duration 4 --out bee_out
.venv\Scripts\python.exe run.py --out demo          # synthetic insects, known wingbeats
```

`--start` is seconds after the recording's first event; timestamps in the
outputs keep the recording's own clock. The recording is processed in
back-to-back windows of `--window` seconds (default 4). A 4 s window of a busy
1280x720 recording (about 5 million events) takes one to two minutes.

### Presets: one algorithm, two physical scales

| preset | detector window / step | coast | wingbeat band | used for |
|---|---|---|---|---|
| `moth` | 12 ms / 4 ms | 100 ms | candidates 18-80 Hz, line search 15-500 Hz | Bogong moths at a lamp, Oct 2026 |
| `bee` | 5 ms / 2.5 ms | 10 ms | 120-320 Hz | honey bees (the project's frozen detection recipe) |

The detector window is about one wing stroke of the animal: 5 ms is a whole
stroke of a 230 Hz bee but a fifth of a 40 Hz moth's, and a window shorter
than a stroke splits one moth into wing fragments. Everything else is shared.

## Outputs

- `tracks.csv`: one row per track. `f0_wingbeat_hz` is the track's wingbeat
  (empty if none could be settled), `fundamental_from` says which estimate
  supplied it, and `verdict` is `wingbeat` or `no line`. `f0_signed_hz` is the
  strongest line in the ON-minus-OFF event rate; on moths it is often a
  harmonic, which is why the octave check exists. `lamp_R` measures lock to
  the scene's mains flicker. Every other column the research code computes
  follows.
- `strokes.csv`: one row per detected wing stroke: track, time, interval to
  the previous stroke, `stroke_rate_hz` = 1 / interval, and
  `rate_median5_hz`, the median of the last five `stroke_rate_hz` values,
  which is the changing wingbeat number the research clips show.
- `detections.csv`: every detection in every track: time, position, size.
- `census_<start>s.png`: tracks drawn on the window's event count image,
  coloured by verdict, with the wingbeat histogram.
- `summary.json`: settings, windows, counts.

What the numbers are, and how well they were measured:
- Track counts are not animal counts: one animal can be several tracks.
- `f0_wingbeat_hz` is an average over the track. Which multiple of the
  stroke rate is the wingbeat is decided by where the animal's events repeat
  in space; the number then comes from the spectral line
  (`fundamental_from` = signed, signed/2 or signed/3), and from the period
  estimator (`yin`) only when no spectral candidate names that period. On
  this folder's synthetic insects (`run.py` with no input) every track lands
  within 0.10% (moths) and 0.32% (bees) of the true wingbeat.
- `stroke_rate_hz` times each stroke from the ON-event rate, between the
  0.5 ms bins. On synthetic insects with known rates the typical stroke is
  within 0.8% (moths) and 1.5% (bees) of the truth. On real recordings there
  is no truth, but intervals reading over 1.4x or under 0.7x the track's own
  wingbeat, which are stroke-clock errors rather than wings, were 7% on a bee
  test window and 11% on the moth recording. A second clock (the event centroid)
  disagrees with this one stroke by stroke, so the per-stroke rate shows that
  the rate changes more than it measures each stroke exactly.
- `rate_median5_hz` ignores a single stroke counted twice or missed: those
  gross errors fall to 0.9% on the bee window and 3.3% on the moth recording.
  The cost is time. On synthetic insects whose wingbeat swings by 15%, it
  shows 95% of a moth's swing at 2 swings per second, 70% at 4 and 32% at 8,
  about 60 ms late; for bees, 94% at 10 swings per second, 11 ms late.
- No detection accuracy is claimed. These recordings have no labels.

## Known limits of this version

- A track that crosses a window edge is cut there and appears in both
  windows.
- A moth whose wings form separate blobs in one detector window can carry
  two IDs at once. The research project is testing a fix (grouping
  overlapping boxes); it is not in this version.
- The bee results in the research project also used a calibrated front end
  (Conv1 or a frozen threshold map) that needs per-site calibration. It is
  not included, so bee runs use the same 1 ms noise filter as moth runs.
- The lamp tests always use the strongest scene line between 95 and 105 Hz,
  even in a recording with no lamp, and a wingbeat within 1 Hz of 1, 2 or 3
  times that line is not reported. Near 200 Hz this can drop a bee.

## Supported formats

| Format | Layout |
|---|---|
| `.raw` | Prophesee/Metavision EVT2, EVT2.1 or EVT3, decoded by faery. Encoding and size come from the header; headerless files need `--encoding`, `--width`, `--height`. |
| `.dat` | Prophesee DAT, decoded by faery. |
| `.npz` | 1D arrays `x`, `y`, `t` (microseconds), `p`; optional `width`, `height`. |
| `.csv` | Headers `x,y,t_us,polarity` or `x,y,t,p`; pass `--width`, `--height`. |
| `.h5` | `x`, `y`, `t`, `p` datasets at root or under `/events`; `width`/`height` attributes. |

Polarity may be -1/+1, 0/1 or boolean. Events must be in time order.

Do not swap faery for expelliarmus. On an IDS (Sony IMX636) EVT3 recording
expelliarmus returned the right events with timestamps running at half speed
(the lamp's 100 Hz flicker read 49.99 Hz), which halves every frequency.

## Check it

```powershell
.venv\Scripts\python.exe verify.py
```

Synthetic insects with known wingbeats come back within 1% (both presets);
NPZ, CSV, HDF5 and EVT3 RAW copies of the same events give identical tracks;
`try_my_data.py` with typed answers gives the same tracks as `run.py`;
malformed input is rejected with a message.

## Read the code

Start with `WALKTHROUGH.md`, then `run.py`, then `census.run_census`.

## Licence

MIT, from the Ecology project (`PROJECT_LICENSE_DECLARATION.toml`, `LICENSE`).
`insect_evs/descriptor.py` is from evfilt (MIT, same author). faery is
LGPL-3.0, installed separately and not bundled. No recordings, derived media
or dataset files are included.
