Metrics
=======

The evaluators treat each level as a binary decision: anomalous (the
positive class) or normal. They do not need a threshold. Instead, they
measure how well the scores *rank* anomalous data above normal data, over all
possible thresholds. ``compute()`` returns the result as a dict:

.. code-block:: python

   >>> timestep_eval.compute()
   {'auroc': 0.81, 'aupr': 0.46, 'ap': 0.45, 'fpr95': 0.62,
    'n_timesteps': 61234,
    'by_type': {'change-weather': {'auroc': 0.93, ..., 'n_timesteps': 2910},
                'spawn-props': {...}, ...}}

All metrics are fractions between 0 and 1. The paper and the
`results tables <https://carlanomaly.de/tasks/>`_ on the website give them as
percentages.

The metrics
-----------

``auroc``
   Area under the ROC curve: the probability that a randomly chosen anomalous
   sample gets a higher score than a randomly chosen normal one. 1 is a
   perfect ranking, 0.5 is chance. It does not depend on how rare anomalies
   are, which makes it easy to compare across levels and types.
``aupr``, ``ap``
   Area under the precision-recall curve. Precision is the share of flagged
   samples that are really anomalous, so unlike AUROC, these metrics drop
   when anomalies are rare and false alarms are many. Chance level equals the
   share of anomalous samples, which is small at the sample level. ``aupr``
   integrates the curve with the trapezoidal rule, ``ap`` as a step function,
   like ``sklearn.metrics.average_precision_score``. They agree for
   continuous scores. For scores with many ties, such as binary flags,
   ``aupr`` is too optimistic; report ``ap`` then.
``fpr95``
   False positive rate at 95% true positive rate: set the threshold so that
   95% of the anomalous samples are detected, and report the share of normal
   samples above it. Lower is better. It describes the operating point a
   safety system would need, where AUROC averages over all of them.

.. _time-to-detection:

Time to detection
-----------------

The metrics above say how well a detector ranks anomalies, not how soon it
reports them. :meth:`TimestepEvaluator.time_to_detection
<carlanomaly.TimestepEvaluator.time_to_detection>` (also on
:class:`~carlanomaly.SensorEvaluator`) measures that, at one operating point:

1. Set the threshold on the normal scenarios of the test split: the lowest
   value that at most ``max_false_alarm_rate`` of them (default 1%) exceed
   at any timestep. A score counts as an alarm only if it is strictly above
   the threshold, so scores tied with it never raise one.
2. In each anomaly scenario, find the first timestep that is labelled
   anomalous and the first alarm from then on. Alarms before the anomaly
   starts are ignored.
3. The delay is the time between the two, in seconds. A scenario without an
   alarm counts with the time to its last timestep.

The result:

``ttd``
   The mean delay over all anomaly scenarios, misses included. Lower is
   better. Since misses count with the rest of the scenario, it rewards both
   detecting more anomalies and detecting them sooner.
``detection_rate``
   The share of anomaly scenarios with an alarm after the anomaly started.
``false_alarm_rate``
   The share of normal scenarios with an alarm at the chosen threshold. It
   is at most ``max_false_alarm_rate``, and below it when several normal
   scenarios share the deciding score. Report it with ``ttd``.
``threshold``, ``n_scenarios``, ``n_normal_scenarios``, ``by_type``
   The threshold, the number of anomaly and normal scenarios used, and
   ``ttd``, ``detection_rate`` and ``n_scenarios`` per anomaly type.

Like ``fpr95``, this fixes an operating point on the test split itself. That
is not tuning the detector: the threshold only decides which point of its
behaviour is reported. A detector deployed with a threshold chosen on other
data would show a different false-alarm rate.

Breakdowns and counts
---------------------

``by_type``
   The metrics for each anomaly type on its own, keyed by the type names in
   :ref:`the dataset <evaluation-levels>`. At the sensor, timestep and
   scenario levels, each type's scenarios are evaluated together with all
   normal scenarios. At the sample level, only the pixels or points of the
   type's scenarios are used.
``n_samples``, ``n_timesteps``, ``n_scenarios``
   How many scores went into the result. Compare them with the size of the
   test split to make sure nothing was skipped.
``n_positive``
   (Sample level, also per type.) The number of anomalous pixels or points.

A metric is ``nan`` when the evaluated data contains only one class, for
example on the train split, which has no anomalies, or in ``by_type`` for a
type without masks at the sample level.

Sample level
------------

The sample-level results have two more entries:

``scenario_macro``
   The metrics computed for each scenario separately and then averaged,
   over the scenarios that contain anomalous pixels, plus their count
   ``n_scenarios``. The main metrics are dominated by scenarios with large
   anomalies, because they contribute the most anomalous pixels; here every
   scenario counts the same.
``clamped_fraction``
   The share of scores outside the range of the evaluator's histogram. It
   should be close to 0.

The test split has billions of pixels, too many to keep every score. The
sample-level evaluators count scores in a fine histogram instead, which keeps
memory constant and is accurate as long as the scores fall inside its range.
The default range suits unbounded scores, such as negative logits or
reconstruction errors, up to a magnitude of about one million. If
``clamped_fraction`` is not close to 0, pass ``score_range``, and for scores
of an unusual distribution ``score_transform``, to
:class:`~carlanomaly.PixelEvaluator` or :class:`~carlanomaly.PointEvaluator`.

The sensor, timestep and scenario evaluators keep every score and compute the
metrics exactly; for them, only the order of the scores matters.
