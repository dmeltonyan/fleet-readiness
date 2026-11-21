# Decision log

What I decided, what I considered instead, and why.

## Data and splits

**Split by engine, never by row.** Rows from the same engine are near-duplicates of each other. A row-level split would put cycle 150 of an engine in training and cycle 151 in test and report a score no new engine would ever get. FD004's 249 training engines are shuffled once (seed 7) into train 137 / calibrate 37 / tune 38 / test 37.

**Four disjoint groups, each with one job.** The model learns on *train*. The uncertainty band is calibrated on *calibrate*. Policy thresholds are tuned on *tune*. Only *test* engines appear in the reported fleet results. If thresholds were tuned on the engines they are scored on, the risk-based policy would look better than it is.

**FD004 for the fleet study.** It is the hardest subset: six operating conditions and two fault modes. If the approach works there, it works on the easier ones.

## Features

**Normalise within operating regime.** In FD004 the operating condition explains most of the variance in raw sensor readings; degradation is a small signal on top. KMeans on the three settings finds the six regimes, fitted on training engines only.

**Drift from the engine's own baseline.** Engines start with different manufacturing variation. Comparing each engine to its first 20 cycles removes that offset. It is by far the most useful feature family: 83% of the model's split gain, and removing it raises held-out RMSE from 10.9 to 15.5 cycles. Engine age (cycle count) is second; removing it costs 3.1 cycles of RMSE. Slopes and the exponentially weighted mean add almost nothing on top.

**Leakage is tested, not assumed.** `tests/test_features.py` appends future cycles to an engine and checks that no earlier feature changes, and alters one engine to check no other engine's features move. A model trained on shuffled labels scores RMSE 40+ on held-out engines, as it should.

## Model

**Cap RUL at 125.** Early in life the sensors carry no information about when an engine will fail, so asking the model to distinguish "300 cycles left" from "200 cycles left" just adds noise. Capping is the standard treatment for this dataset and matches how a maintainer thinks: healthy, or counting down.

**LightGBM over a deep sequence model.** Gradient-boosted trees on good trailing-window features reach 10.6 RMSE on FD004's official test set (14.1 on harder held-out random cuts), train in seconds, and are easy to inspect. A sequence model might gain a little accuracy; the decision results show that ranking and uncertainty matter more than the last cycle of RMSE.

**Report a second, harder score.** The official FD004 test set has 67 of 248 engines with true RUL above 125, which the cap turns into an easy target. Holding out training engines and cutting them at random points gives 14.1 RMSE, and that is the number I would quote.

**Conformal calibration of the band.** Raw quantile regression is often over-confident. Split-conformal calibration on separate engines widens the band by exactly enough to hit the target coverage on those engines. On test engines it covers 89% overall (target 80%) but 79% within 50 cycles of failure. Rows from the same engine are correlated, so the exchangeability assumption behind conformal guarantees only holds approximately; the near-failure shortfall is the visible cost.

## Policies and simulation

**Score decisions, not just predictions.** RMSE treats a 20-cycle error at 120 cycles of remaining life the same as one at 20 cycles. In maintenance the second one is the in-flight failure. The simulation measures what each policy actually costs.

**The risk-based policy uses the pessimistic bound.** "Pull when there is a real chance this engine fails before the next couple of checks" is closer to how maintainers reason than "pull when the average guess crosses a line". Ranking by the same bound sends the engines most likely to fail through first when slots are short.

**A strong time-based baseline.** The comparison is only meaningful if the calendar schedule is as good as a calendar schedule can be. Its replacement age is tuned on the same engines as the other policies, and it agrees with the classic Weibull age-replacement optimum (135 vs 140 cycles).

**At one slot per check the time-based age hits its search floor (60).** With one slot every policy saturates it (300 removals = one per check), so the replacement age barely changes the outcome. Lowering the floor would not change the conclusion.

**Common random numbers.** Every policy sees exactly the same engines in the same fleet positions with the same starting ages. Differences in the results come from decisions, not luck of the draw.

**Costs are relative.** A planned removal is 1 unit and an in-service failure 6 units, with 5 vs 25 cycles of downtime. These are assumptions, so the study is re-run at failure costs of 3 and 12 units. The risk-based policy is cheapest in every case.
