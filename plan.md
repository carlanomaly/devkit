The central problem is not AUROC itself, but the **conditional sampling distribution induced by discarding anomaly-free frames**.

## 1. What your current metric actually measures

Let (S) denote the pixel anomaly score, (Y\in{0,1}) the pixel label, and (A_f) indicate that frame (f) contains at least one anomalous pixel.

Your current procedure estimates

[
\frac{1}{|\mathcal F_+|}
\sum_{f\in\mathcal F_+}
\Pr!\left(S_f^+>S_f^-\mid f\right),
\qquad
\mathcal F_+={f:A_f=1}.
]

This is a **macro-average of within-positive-frame ranking probabilities**. It does not estimate the probability that a randomly selected anomalous pixel receives a higher score than a randomly selected normal pixel from the complete test distribution:

[
\operatorname{AUROC}_{\mathrm{global}}
======================================

\Pr(S^+>S^-).
]

The conditioning on (A_f=1) changes the negative distribution from

[
p(S\mid Y=0)
]

to

[
p(S\mid Y=0,A_f=1).
]

That distinction creates exactly the traffic-light pathology. If malfunctioning traffic lights occur only in traffic-light-heavy frames, the negative pixels against which anomalous traffic-light pixels are compared are drawn mostly from roads, buildings, sky, and vehicles—not from ordinary traffic lights. A detector can therefore rank

[
\text{all traffic lights} > \text{most other pixels}
]

without ranking

[
\text{malfunctioning traffic lights} >
\text{ordinary traffic lights}.
]

Yet it receives a high AUROC.

This problem is especially pertinent for CarlAnomaly because several annotated anomalies are object- and context-specific, including turned-off and flickering traffic lights. 

## 2. AUPR does not solve this problem

Replacing AUROC with AUPR or average precision while retaining only positive frames does **not** remove the selection bias.

AUPR answers a different question:

[
\operatorname{Precision}(t)
===========================

\frac{\Pr(S\ge t,Y=1)}
{\Pr(S\ge t)}.
]

It is sensitive to anomaly prevalence, which is often desirable for highly imbalanced pixel segmentation. However, after restricting evaluation to anomalous frames, both the score distribution and anomaly prevalence are conditional on (A_f=1). The model is still not penalized for scoring ordinary traffic lights highly in normal frames.

Moreover:

* AP is undefined in a frame without positive pixels.
* Assigning AP (=0) to such frames is arbitrary: a perfectly normal frame with no false positives would still receive zero.
* Ignoring such frames recreates the original bias.
* Macro-averaging frame AP gives a tiny anomaly and a huge anomaly equal weight.

AUPR/AP is nevertheless a useful **dataset-level complement** to AUROC because anomaly pixels are rare. Current anomaly-segmentation benchmarks commonly report pixel-level AP/AUPR and FPR95; some also report component-level localization measures. ([arXiv][1])

## 3. Recommended primary metric: pooled pixel evaluation

The cleanest solution is to treat all evaluated pixels as one binary classification problem:

[
\mathcal D_{\mathrm{pixel}}
===========================

\bigcup_{f\in\mathcal F}
{(s_{fi},y_{fi})}_{i=1}^{H W}.
]

Compute a single:

* pooled pixel AUROC;
* pooled pixel AUPR/AP;
* pooled FPR95.

This includes:

* all pixels from anomalous frames;
* all pixels from non-anomalous frames;
* ideally all normal scenarios as well.

This metric asks the operationally relevant question:

> Across the complete benchmark distribution, does the detector rank anomalous pixels above normal pixels?

Normal traffic lights then enter the negative population and expose the shortcut.

Pixel-wise uncertainty evaluation over complete datasets is conceptually consistent with established anomaly-segmentation benchmarks such as Fishyscapes, which evaluate pixel-wise uncertainty against anomalous-object masks. ([arXiv][2])

### Important qualification

A global pixel AUROC will be dominated by abundant easy negatives such as sky, road, and buildings. Thus it should be the primary broad metric, but not the only diagnostic metric.

## 4. You do not need to retain or sort all pixels

The computational objection is valid for a naïve implementation, but not fundamental.

### 4.1 Streaming histogram approximation

Maintain two histograms over anomaly scores:

[
h_b^+
=====

#{i:y_i=1,\ s_i\in I_b},
\qquad
h_b^-
=====

#{i:y_i=0,\ s_i\in I_b},
]

where (I_1,\ldots,I_B) are score intervals.

For every batch or frame:

1. compute scores and labels;
2. bin positive scores;
3. bin negative scores;
4. add the counts to the two global histograms;
5. discard the score maps.

After traversing the dataset, cumulative counts from high to low thresholds yield

[
\operatorname{TPR}_b
====================

\frac{\sum_{j\ge b}h_j^+}{N_+},
\qquad
\operatorname{FPR}_b
====================

\frac{\sum_{j\ge b}h_j^-}{N_-}.
]

The ROC integral is then approximated by trapezoidal integration. Precision–recall is obtained analogously:

[
\operatorname{Precision}_b
==========================

\frac{\sum_{j\ge b}h_j^+}
{\sum_{j\ge b}(h_j^++h_j^-)}.
]

Complexity becomes

[
\text{memory}=O(B),\qquad
\text{time}=O(N+B),
]

rather than storing (O(N)) scores and performing an (O(N\log N)) global sort.

With (B=65,536) bins and 64-bit counters, the two count arrays require roughly

[
2\cdot 65,536\cdot8
\approx 1\text{ MiB}.
]

Histogram and sketch methods are an established approach for bounded-memory approximation of AUC and streaming distributions. ([Hal Science][3])

For bounded scores such as MSP, fixed bins are immediate. For unbounded MaxLogit or energy scores, use one of:

* a monotonic bounded transform, such as (\sigma(s/T));
* score limits determined from a calibration split;
* a first pass to determine robust quantiles;
* adaptive or quantile-based bins.

Because AUROC is invariant under strictly monotonic transformations, a bounded monotonic transform does not change the exact ranking; it only controls histogram resolution.

### 4.2 Exact external-memory computation

An exact result is also possible without large RAM:

1. write sorted score-label chunks to disk;
2. externally merge the chunks;
3. accumulate ROC statistics during the merge.

This retains exact ordering but requires substantial disk I/O. For a benchmark implementation, a validated histogram approximation is likely the better engineering trade-off.

### 4.3 Pixel subsampling

Another practical option is:

* retain all anomalous pixels;
* randomly sample normal pixels from every frame;
* compute AUROC on the resulting sample.

Uniform negative sampling produces a consistent estimate of AUROC because AUROC is a pairwise probability. However, naïve subsampling changes class prevalence and therefore changes precision and AUPR. For AUPR, either retain known sampling weights or compute it from streaming counts over the complete dataset.

## 5. Scenario-pooled AUROC as a useful secondary metric

Your suggestion to pool all pixels within one scenario is sound:

[
\operatorname{AUROC}_k
======================

\operatorname{AUROC}
\left(
\bigcup_{f\in\text{scenario }k}
{(s_{fi},y_{fi})}
\right).
]

Then report

[
\frac{1}{K_+}\sum_{k:y_k^+=1}\operatorname{AUROC}_k.
]

This is much better than averaging frame AUROCs because it includes anomaly-free frames before and after the event within each anomalous scenario. Consequently, normal instances of difficult classes appearing elsewhere in the clip become negatives.

However, it remains conditional on anomalous scenarios. It therefore does not fully penalize false positives occurring only in normal scenarios. I would treat it as a **scenario-macro diagnostic**, not as the principal pixel-level metric.

It also has a useful weighting property:

* global pooled AUROC weights scenarios roughly by their numbers of pixels and positives;
* macro scenario AUROC gives every anomalous scenario equal weight.

Reporting both distinguishes performance on frequent/easy anomalies from consistency across scenarios.

## 6. Add an explicit normal-data false-positive metric

AUROC and AUPR describe ranking over thresholds, but your traffic-light example is fundamentally also a false-alarm problem. Therefore, separately evaluate normal frames or normal scenarios.

Choose a threshold (\tau) using only training or validation data, for example such that the normal validation pixel FPR is (1%). Then report on normal test scenarios:

[
\operatorname{PixelFPR}(\tau)
=============================

\frac{#{Y=0,S\ge\tau}}
{#{Y=0}},
]

and perhaps

[
\operatorname{FrameFPR}(\tau)
=============================

\frac{#{\text{normal frames containing a predicted anomaly}}}
{#{\text{normal frames}}}.
]

For segmentation, “containing a predicted anomaly” should usually require a minimum connected-component size rather than a single pixel. Otherwise, one isolated noisy pixel makes an entire frame positive.

This metric directly penalizes the model for repeatedly flagging ordinary traffic lights, even though all pixels in those frames are negative.

## 7. Context-matched or class-conditional evaluation

The strongest diagnostic for your concrete failure mode is to compare anomalous objects against semantically matched normal objects.

For traffic-light anomalies:

[
\operatorname{AUROC}_{\mathrm{TL}}
==================================

\Pr\left(
S_{\text{anomalous traffic light}}

>

S_{\text{normal traffic light}}
\right).
]

You already possess semantic and instance labels, so this should be feasible. Generalize it as:

[
\operatorname{AUROC}_c
======================

\Pr(S^+>S^-\mid C=c),
]

where (C) is the underlying semantic class or object category.

Examples:

* malfunctioning traffic-light pixels versus normal traffic-light pixels;
* anomalous vehicle pixels versus normal vehicle pixels;
* anomalous streetlight pixels versus normal streetlight pixels.

This disentangles two abilities:

1. **object novelty or uncertainty:** “this looks like a difficult traffic light”;
2. **anomaly discrimination:** “this traffic light is abnormal relative to other traffic lights.”

For contextual anomalies, the negative matching can additionally condition on location or context:

[
\Pr(S^+>S^-\mid
\text{same class, similar scale, similar region}).
]

This should not replace the global metric because matched controls may be unavailable for arbitrary unknown objects. It is nevertheless an excellent category-specific diagnostic and would make your discussion substantially stronger.

## 8. Component-level metrics

Pixel metrics can reward diffuse score maps and are dominated by large regions. Since CarlAnomaly supplies instance identifiers, you can additionally evaluate anomalous components or objects.

Useful quantities include:

* fraction of anomalous instances detected at threshold (\tau);
* mean overlap with anomalous components;
* number of false-positive components per frame;
* component-level precision, recall, and (F_1);
* detection rate as a function of anomaly size.

Segment-Me-If-You-Can supplements per-pixel AP and FPR95 with component-level metrics such as component-wise IoU, positive predictive value, and component-level (F_1). ([arXiv][1])

These metrics address a different pathology: a model should not receive nearly perfect credit merely because it marks part of a very large anomalous region, nor should a small object disappear statistically among millions of background pixels.

## 9. A defensible metric suite for CarlAnomaly

I would recommend the following hierarchy.

### Primary sample-level results

1. **Global pooled pixel AUPR**
2. **Global pooled pixel AUROC**
3. **Global pooled pixel FPR95**

Computed over all test pixels using streaming histograms.

AUPR should probably be listed first because anomaly pixels are rare and because precision captures the practical cost of false alarms. AUROC remains useful because it is prevalence-invariant and easy to interpret as a ranking probability.

### Secondary robustness results

4. **Macro scenario-pooled AUROC/AUPR**, over anomalous scenarios
5. **Normal-scenario false-positive rate** at validation-calibrated thresholds
6. **Component-level detection metrics**

### Per-anomaly diagnostics

7. **Class- or context-matched AUROC**, where meaningful
8. Results stratified by anomaly type and anomaly size

This combination separates three questions:

[
\begin{array}{ll}
\text{Global discrimination:}
& \text{Do anomaly pixels outrank ordinary test pixels?}[2mm]
\text{False-alarm control:}
& \text{Does the method remain quiet on normal data?}[2mm]
\text{Semantic discrimination:}
& \text{Can it distinguish abnormal objects from normal}\
& \text{instances of the same difficult category?}
\end{array}
]

## 10. What I would avoid

I would not use any of the following as the headline metric:

* mean AUROC over only anomaly-containing frames;
* mean AP over only anomaly-containing frames;
* treating AP/AUROC on all-negative frames as zero;
* reporting only global pixel AUROC without AUPR or false-positive analysis;
* comparing each anomaly solely against negatives from its own frame.

These are computable, but they answer distorted or incomplete questions.

## Bottom line

The best correction is **not merely replacing AUROC with AUPR**. The essential change is to remove the conditioning on anomaly-containing frames.

Use a streaming, dataset-pooled evaluation over all pixels; report AUROC, AUPR, and FPR95; complement it with normal-scenario false-positive rates and class-matched diagnostics. Scenario-level pooling is a good interim improvement, but global streaming aggregation is both statistically cleaner and computationally practical.

[1]: https://arxiv.org/pdf/2404.17961?utm_source=chatgpt.com "arXiv:2404.17961v2 [cs.CV] 15 Sep 2024"
[2]: https://arxiv.org/abs/1904.03215?utm_source=chatgpt.com "The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation"
[3]: https://hal.science/hal-03321231v1/document?utm_source=chatgpt.com "Fast and memory ecient AUC-ROC approximation in ..."


