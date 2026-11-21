# Decisions

Notes on the choices I made and why.

## Splitting by engine

Rows from the same engine are almost identical cycle to cycle, so splitting by row would leak: cycle 150 in training, cycle 151 in test. I split FD004's 249 training engines into four groups instead: 137 to train the model, 37 to calibrate the uncertainty range, 38 to tune the policy thresholds, and 37 that are only used for the final simulation results.

The extra groups matter for the policies. If I tuned the thresholds on the same engines I report results on, the model-based policies would look better than they really are.

I used FD004 for the simulation because it's the hardest subset (six operating conditions, two fault modes).

## Features

The operating condition (altitude, speed, throttle) changes the raw sensor values far more than wear does. KMeans on the three operating settings finds the six conditions, and I normalize each sensor within its condition. This is fit on the training engines only.

Every engine starts with slightly different manufacturing variation, so I also measure how far each sensor has drifted from that engine's own first 20 cycles. This turned out to be the most important feature group by a lot: 83% of the model's split gain, and dropping it raises held-out error from 10.9 to 15.5 cycles. Engine age is second (dropping it costs about 3 cycles). The slope and exponential average features barely help.

To make sure nothing looks ahead, `tests/test_features.py` adds future cycles to an engine and checks that earlier features don't change, and changes one engine's data to check that no other engine's features move. As another sanity check, a model trained on shuffled labels gets 40+ cycles of error, which is what you'd expect from a model that learned nothing.

## The model

I cap remaining life at 125 cycles. Early on, the sensors don't contain any information about whether failure is 200 or 300 cycles away, so asking the model to tell the difference just adds noise. This is the standard way people handle this dataset.

I went with LightGBM instead of an LSTM or transformer because it trains in seconds, is easy to inspect, and was accurate enough. The simulation results suggest the ranking and the uncertainty estimate matter more for the decisions than squeezing out another cycle of error.

The FD004 official test score (10.6) looked suspiciously good, so I checked it a second way: hold out training engines, cut each at a random point, and score there. That gives 14.1. The gap is because a quarter of the official test engines are cut off early, where the capped answer is just 125. I'd quote 14.

For the uncertainty range, raw quantile regression tends to be overconfident. Split conformal calibration widens it just enough to hit 80% coverage on the calibration engines. On new engines it covers 89% overall but 79% in the last 50 cycles. Conformal guarantees assume the data points are exchangeable, and rows from the same engine aren't really independent, so that near-failure gap is probably where that shows up.

## Policies and the simulation

RMSE treats every error the same, but in maintenance a 20-cycle mistake when an engine has 120 cycles left is nothing, and the same mistake at 20 cycles left is a failure in service. That's why the comparison is done in a simulation that counts costs and failures.

The risk-based policy uses the 10th percentile of predicted remaining life instead of the point estimate. The idea is "pull it if there's a real chance it fails before the next couple of checks", which seems closer to how a maintainer would actually think.

I wanted the time-based baseline to be as strong as a fixed schedule can be, otherwise beating it doesn't mean much. Its replacement age is tuned the same way as the other policies, and the result (140 cycles) is close to what the textbook Weibull age-replacement calculation gives (135).

With only one slot per check, the tuned time-based age ends up at the bottom of its search range (60). Every policy uses every slot at that capacity, so the setting barely changes anything.

Every policy sees exactly the same engines, in the same positions, starting at the same ages. That way differences between policies come from their decisions and not from luck.

The costs (planned removal = 1, failure = 6) and downtimes (5 vs 25 cycles) are assumptions, so I also re-ran with failures costing 3 and 12. The risk-based policy was cheapest every time.
