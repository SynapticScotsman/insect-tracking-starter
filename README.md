# Insect tracking and wingbeat starter

Give it an event-camera recording of flying insects, honey bees by default;
it finds them, tracks them, and measures each one's wingbeat, as a per-track
frequency and as the time of every wing stroke. It reads Prophesee `.raw`,
Kairos `.raw.kai` and most other event formats (see Input files).

The algorithm files (`census.py` and `insect_evs/`) are copied from a private
research repository; `PROVENANCE.json` names the source commit and
fingerprints every copied file. Their comments were rewritten for a first
read, and the code itself is unchanged. Run on a Bogong moth recording and a
honey bee recording, this folder gives the research code's answers exactly,
track for track and stroke for stroke.

## Install once

Unzip the folder and open a terminal inside it. Windows, Python 3.9 or newer:

```powershell
py -3.9 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux use `python3 -m venv .venv` and `.venv/bin/python`.

Check the install:

```powershell
.venv\Scripts\python.exe check_install.py
```

It lists every package, what it is for and the version found, and prints the
command that installs anything missing or too old. `numba` is required, not
optional: without it the noise filter falls back to a slower approximation
that gives slightly different results.

## Try it on your own recording

```powershell
.venv\Scripts\python.exe try_my_data.py
```

It opens a file dialog (or pass the path: `try_my_data.py bees.raw`;
`--no-dialog` to type it), asks which settings to use (bee, moth or wide), where to
start and how many seconds to process, then runs everything and prints the
wingbeat of the longest tracks. Every question has a default in brackets;
press Enter to take it. Outputs go to `outputs/NAME_SETTINGS_STARTs/`. It runs
exactly the code below; `run.py` is the same thing driven by flags.

## Use from Python

```python
import run, make_clip

folder = run.track("bees.raw", start=10, duration=4, out="bee_out")   # bee settings by default
make_clip.make_clip(folder, start=1.0)                       # 0.5 s of bees at 1/40 speed
make_clip.make_clip(folder, follow=134)                      # zoom on track 134

moths = run.track("moths.raw", preset="moth", out="moth_out")
make_clip.make_clip(moths)                                   # 2 s of moths at 1/10 speed
```

`run.track` takes the same options as `run.py` below and returns the output
folder. Run it from this folder, or add this folder to `sys.path`.

## Videos

`make_clip.py` turns a finished run into a short slow-motion MP4, without
running the tracking again:

```powershell
.venv\Scripts\python.exe make_clip.py bee_out --start 1
.venv\Scripts\python.exe make_clip.py bee_out --follow 134
.venv\Scripts\python.exe make_clip.py moth_out --seconds 3
```

Each frame shows the last few milliseconds of events: blue for ON (brighter),
orange for OFF (darker), cyan for events no track is near, which is where an
insect was missed. Every live track has a circle, its ID and a short trail.
Magenta tracks have a measured wingbeat and show `rate_median5_hz`, grey ones
do not. Bees beat about 230 times a second, so a bee run's clip defaults to
0.5 s at 1/40 speed, and a moth run's to 2 s at 1/10; `--seconds` and
`--slow` change either. `--follow ID` keeps one track
centred and zoomed in. `try_my_data.py` offers a clip at the end of each run.
The clip is written into the run folder.

## Run

```powershell
.venv\Scripts\python.exe run.py --input "bees.raw" --duration 4 --out bee_out          # bee settings
.venv\Scripts\python.exe run.py --input "moths.raw" --preset moth --start 10 --duration 8 --out moth_out
.venv\Scripts\python.exe run.py --out demo          # synthetic bees, known wingbeats
```

`--start` is seconds after the recording's first event; timestamps in the
outputs keep the recording's own clock. The recording is processed in
back-to-back windows of `--window` seconds (default 4). A 4 s window of a busy
1280x720 recording (about 5 million events) takes one to two minutes.

### Presets: one algorithm, three physical scales

| preset | detector window / step | coast | wingbeat range | for |
|---|---|---|---|---|
| `bee` (default) | 5 ms / 2.5 ms | 10 ms | 120-320 Hz | honey bees |
| `moth` | 12 ms / 4 ms | 100 ms | 18-80 Hz | moths and other large, slow flyers (tested on Bogong moths) |
| `wide` | 8 ms / 4 ms | 50 ms | 15-500 Hz | a recording where you do not know what is flying |

For bumble bees and other large bees, whose stronger second harmonic can make
the frequency read double, check `f0_yin_hz` against `f0_signed_hz` before
quoting a number. For hoverflies, which reach about 330 Hz, widen the range
with `--band 110 350`; the 10 dB line test then drops a few borderline tracks.

The detector window is about one wing stroke of the animal: 5 ms is a whole
stroke of a 230 Hz bee but a fifth of a 40 Hz moth's, and a window shorter
than a stroke splits one moth into wing fragments. The coast is how long a
track may go unseen before it ends. Everything else is shared.

Any preset can be adjusted with `run.py`:
- `--band LO HI`: the wingbeat range in Hz. A wingbeat outside it is not
  reported.
- `--window-ms MS`: the detector window; the step is half of it.
- `--max-size PX`: the largest blob kept, default 90 px. Raise it for insects
  that fill more of the image.

For an unknown recording, run `wide` first, look at where the wingbeats fall,
then narrow `--band` and set `--window-ms` to about 1000 / wingbeat.

## Outputs

- `tracks.csv`: one row per track. `f0_wingbeat_hz` is the track's wingbeat
  (empty if none could be settled), `fundamental_from` says which estimate
  supplied it, and `verdict` is `wingbeat` or `no line`. `f0_signed_hz` is the
  strongest line in the ON-minus-OFF event rate; on moths it is often a
  multiple of the true wingbeat, which is why the multiple is checked
  separately. `lamp_R` measures how strongly the track flickers with the
  scene's mains lighting. Every other column the census computes follows.
- `strokes.csv`: one row per detected wing stroke: track, time, interval to
  the previous stroke, `stroke_rate_hz` = 1 / interval, and
  `rate_median5_hz`, the median of the last five `stroke_rate_hz` values,
  which follows the wingbeat as it changes.
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
  wingbeat, which are errors of the stroke timing rather than real wing
  changes, were 7% on a honey bee recording and 11% on a moth recording. Timing
  strokes a second way, from the event centroid, disagrees with this one
  stroke by stroke, so treat single strokes as approximate.
- `rate_median5_hz` ignores a single stroke counted twice or missed: those
  gross errors fall to 0.9% on the bee recording and 3.3% on the moth one.
  The cost is time. On synthetic insects whose wingbeat swings by 15%, it
  shows 95% of a moth's swing at 2 swings per second, 70% at 4 and 32% at 8,
  about 60 ms late; for bees, 94% at 10 swings per second, 11 ms late.
- With the `wide` preset, wingbeats are as accurate as with the others on
  synthetic insects, but per-stroke timing on slow wings is weaker: two of
  four synthetic 38 Hz tracks read their median stroke 18% low. Once the
  wingbeats are known, rerun with a narrower `--band`.
- No detection accuracy is claimed: the test recordings have no hand labels.

## Known limits

- A track that crosses a window edge is cut there and appears in both
  windows.
- A moth whose wings form separate blobs in one detector window can carry
  two IDs at once.
- Two insects whose boxes overlap in one window become one detection, and
  insects crossing within about 45 px can swap IDs.
- The lamp tests always use the strongest scene line between 95 and 105 Hz,
  even in a recording with no lamp, and a wingbeat within 1 Hz of 1, 2 or 3
  times that line is not reported. Near 200 Hz this can drop a bee.

## Input files

| File | From | How it is read |
|---|---|---|
| `.raw` | Prophesee/Metavision software, and cameras on Prophesee sensors such as the IDS uEye EVS | EVT2, EVT2.1 or EVT3, named with the sensor size in the file's text header. A file with no header needs `--encoding`, `--width` and `--height`. |
| `.raw.kai` | the Kairos recorder (a Prophesee EVK4) | EVT3 behind a 16-byte binary header that holds the sensor size. The header is swapped for a Prophesee one in a temporary copy, which is deleted after reading. The `.index.kai`, `.samples.kai` and `.toml` files beside it are not needed. |
| `.dat` | Prophesee DAT | faery; sensor size from the file |
| `.es` | Event Stream | faery; sensor size from the file |
| `.aedat`, `.aedat4` | iniVation cameras (DV software) | faery; sensor size from the file |
| `.npz` | numpy | arrays `x`, `y`, `t` (microseconds), `p`, plus `width` and `height` |
| `.csv` | any tool | a header row naming `x`, `y`, `t` or `t_us`, `p` or `polarity`; integers only; pass `--width` and `--height` |
| `.h5`, `.hdf5` | h5py and others | datasets `x`, `y`, `t`, `p` at the root or under `/events`; `width` and `height` attributes |

A Kairos file is recognised from its first bytes, whatever it is called;
everything else by its extension. Polarity may be -1/+1, 0/1 or boolean, and
events must be in time order. `verify.py` writes the same synthetic events in
every format above and requires identical tracks from each.

Do not swap faery for the expelliarmus decoder. On EVT3 recordings from an IDS
camera (Sony IMX636 sensor), expelliarmus returns the right events with
timestamps running at half speed, which halves every frequency.

## Check it

```powershell
.venv\Scripts\python.exe verify.py
```

It runs `check_install.py` first, then: synthetic insects with known
wingbeats come back within 1% (all three presets); the same events written in
all 11 file layouts above give identical tracks;
`try_my_data.py` with typed answers gives the same tracks as `run.py`;
malformed input is rejected with a message; `run.track` works from Python and
`make_clip.py` writes a playable clip.

## Read the code

`HOW_IT_WORKS.md` explains each algorithm step by step in plain language.
Then follow `WALKTHROUGH.md` through the code: `run.py`, then
`census.run_census`.

## Licence

MIT (`LICENSE`; the source project's own declaration is in
`PROJECT_LICENSE_DECLARATION.toml`).
`insect_evs/descriptor.py` is from evfilt (MIT, same author). faery is
LGPL-3.0, installed separately and not bundled. No recordings, derived media
or dataset files are included.
