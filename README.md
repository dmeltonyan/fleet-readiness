# Fleet Readiness

Predictive maintenance on NASA's [C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) jet engine dataset. The model predicts how many flight cycles each engine has left, and a maintenance policy uses that prediction to decide which engines to pull in for service when only a couple of maintenance slots are available.

Live results page: https://meltonyan.dev/fleet-readiness/

## The question

Most maintenance programs replace parts on a schedule: every N hours or cycles, regardless of condition. That's safe, but it pulls a lot of healthy engines early. Predicting remaining life sounds like the fix, but a prediction by itself doesn't tell a maintainer what to do this week, especially when the hangar can only take two engines at a time.

So I compared four ways of making that call on a simulated fleet:

1. Run to failure: never pull an engine, fix it when it breaks.
2. Time-based: replace at a fixed age (the scheduled approach).
3. Predicted RUL: pull when the model's estimate of remaining life drops below a threshold.
4. Risk-based: pull when the pessimistic end of the model's uncertainty range (roughly the 10th percentile) drops below a threshold, and when slots are short, take the most at-risk engines first.

## Result

30 engines, 3,000 flight cycles, 2 maintenance slots every 10 cycles, averaged over 10 simulated fleets. Costs are relative: a planned removal is 1 unit, an engine failing in service is 6.

| Policy | Cost per 1k cycles | Failures per 1k | Planned removals | Life thrown away | Availability |
| --- | --- | --- | --- | --- | --- |
| Run to failure | 22.82 | 3.80 | 0 | 0 | 90.5% |
| Time-based (replace at 110 cycles) | 6.67 | 0.00 | 600 | 90.6 cycles | 96.7% |
| Predicted RUL (pull at ≤ 40) | 4.99 | 0.04 | 425 | 33.1 cycles | 97.5% |
| Risk-based (pull at 10th pct ≤ 28) | 4.67 | 0.02 | 411 | 24.8 cycles | 97.7% |

The risk-based policy came out about 30% cheaper than the time-based schedule. Most of that comes from not throwing away good engine life: the scheduled approach pulls engines with 90 cycles left on average, the risk-based one with about 25. It was the cheapest policy at every capacity I tried (1 to unlimited slots) and when I changed the failure cost to 3x or 12x a planned removal.

![Cost and failures as maintenance capacity changes](reports/figures/policy_vs_capacity.png)

## How it works

**Features.** The engines fly in six different operating conditions, and the condition moves the sensor readings much more than wear does. I cluster the three operating settings into six regimes and normalize each sensor within its regime. Then for every cycle I compute rolling means, a recent slope, and how far each sensor has drifted from that engine's own first 20 cycles. The drift features ended up carrying most of the signal (removing them takes the error from 10.9 to 15.5 cycles). Everything only looks backwards in time, and there's a test that checks that.

**Model.** LightGBM regression on remaining life, capped at 125 cycles, since early in an engine's life there's nothing in the sensors that says how far away failure is. Two extra quantile models give a 10th and 90th percentile, and I calibrate that range on a separate set of engines (split conformal) so it actually covers what it claims to.

**Simulation.** Each fleet position holds an engine that starts at a random age. Every 10 cycles the policy picks which engines to pull. A planned removal takes 5 cycles of downtime; a failure takes 25. Failed or removed engines get replaced from a pool of engines the model never trained on. Each policy's threshold is tuned on one group of engines and then scored on a different group, so the tuning can't fit the test.

## Model accuracy

RMSE in cycles on NASA's official test sets (true RUL capped at 125, which is the usual convention):

| Subset | Conditions / fault modes | Fleet-average life | Ridge | LightGBM | LightGBM, random cut-off |
| --- | --- | --- | --- | --- | --- |
| FD001 | 1 / 1 | 36.3 | 16.2 | 11.1 | 9.2 |
| FD002 | 6 / 1 | 35.1 | 15.1 | 10.7 | 13.9 |
| FD003 | 1 / 2 | 49.5 | 15.7 | 9.7 | 9.0 |
| FD004 | 6 / 2 | 54.4 | 16.4 | 10.6 | 14.1 |

The official FD004 number looked too good to me, so I added the last column: hold out 20% of the training engines, cut each one off at a random point in its life, and score there. The official test set has a lot of engines cut off early (67 of 248 in FD004 have more than 125 cycles left), and those are easy once the target is capped. I'd quote about 14 cycles for FD004.

![Predicted remaining life for a typical held-out engine](reports/figures/engine_trace.png)

## Things worth knowing

- The scheduled baseline is a fair fight. Fitting a Weibull distribution to the training engines' lifetimes and solving the standard age-replacement problem gives an optimal replacement age of 135 cycles; tuning it in the simulation gave 140.
- The uncertainty range is weakest where it matters most. It covers 89% of held-out cycles overall (target 80%) but only 79% in the last 50 cycles before failure.
- C-MAPSS is simulated data. Real maintenance data has missing sensors, messy logs and parts shortages, none of which are modeled here.
- The costs and downtimes are my assumptions, all in `config.yaml`.

More detail on each decision is in [DECISIONS.md](DECISIONS.md).

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.fetch_data   # download C-MAPSS
.venv/bin/python -m src.run          # benchmark + fleet simulation, about 3 minutes
.venv/bin/python -m pytest           # tests
```

Results are written to `reports/`.

```
src/data.py        loading data, splitting engines into groups
src/features.py    regime normalization and rolling features
src/models.py      LightGBM model, uncertainty range, baselines, NASA score
src/policies.py    the four maintenance policies
src/simulate.py    fleet simulation
src/run.py         runs everything and makes the charts
tests/             leakage, split and simulator tests
```

## Next

- Add spare parts and lead times. A lot of real downtime is waiting on parts.
- Plan removals a few checks ahead instead of one check at a time (an integer program over a rolling window).
- Calibrate the uncertainty range separately for engines close to failure.
- Try N-CMAPSS, NASA's newer version built from real flight profiles.
