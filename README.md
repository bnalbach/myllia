# Myllia| Echoes of Silenced Genes: A Cell Challenge

The Challenge can be found [here](https://www.kaggle.com/competitions/echoes-of-silenced-genes/). <br>
Citation :
```
@misc{echoes-of-silenced-genes,
    author = {Myllia Biotechnology},
    title = {Myllia| Echoes of Silenced Genes: A Cell Challenge},
    year = {2026},
    howpublished = {\url{https://kaggle.com/competitions/echoes-of-silenced-genes}},
    note = {Kaggle}
}
```

## Problemset
**Predicting responses of a human cancer cell line to CRISPR pertubations**

## Data
Single-cell RNA-seq of nearly most human cell types.

## Evaluation
*Only slightly adapted from the Competition page.*

### Weighted Mean Absolute Error (WMAE)

The WMAE for a single pertubation is given by:

$$
WMAE(true, pred) = \frac{1}{n} \sum_{i=1}^{n} w_i \, |true_i - pred_i|.
$$

with *true* and *predicted* being delta expression values across $n$ genes and a weight vector $w = (w_1, \dots, w_n) \in \mathbb{R}^n$. 

$$
w_i\ge0 \text{ for } i = (1,\dots,n) \qquad \sum_{i=1}^{n} w_i = n.
$$

The weights vector is calculated using t values of the moderated t-statistic (limma package),
$$
t_i=\frac{\text{estimated effect for gene i}}{\text{moderated standard error for gene i (from eBayes)}}
$$

with $t=
(t_1,\dots, t_n) \in\mathbb{R}^n$ for a sinlge pertubation:
$$
c_i = \min{(|t_i|+0.1, 10)}, \qquad  i=1,...,n. \\
$$

If the perturbation targets gene with index $g \in 1,\dots,n$, set $c_g = 0$. Let
$$
M = \max_{1\le\text{j}\le\text{n}}{c_i}
$$

The resulting weight vector is defined as:
$$
w_i = n\frac{(\frac{c_i}{M}²)}{\sum_{k=1}^{n}(\frac{c_k}{M})²}, \qquad i = 1,...,n.
$$

Over L pertubations the WMAE ratio score W is then calculated as
$$
W = \sum_{l=1}^{L}min{(5, \log_{2}{(\frac{WMAE_{l}^{base}}{WMAE_{l}^{pred}})})} 
$$
where $WMAE_{l}^{base}$ and $WMAE_{l}^{pred}$ are the baseline and model predictions for the l-th pertubation. The threshold 5 is only to protect against outliers.

The baseline prediction is hereby defined as the arithmetic mean of the log fold-change vectors across the 80 training perturbations. Let $x^{i} \in \mathbb{R}^n$ denote the log fold-change vector for training perturbation $j$, for $j = 1, \dots,80$. The baseline prediction vector is

$$
\bar{x} = \frac{1}{80}\sum_{j=1}^{80}x^{(j)}
$$

### Weighted Cosine Similarity (Wcos)

We define the weighted cosine similarity score $Wcos$ between two vectors $a, b \in \mathbb{R}^m$. In our setting, $a$ is the single concatenated ground-truth delta vector across all genes in all perturbations of length $m = L \times n$, and $b$ is the corresponding concatenated predictions vector.

Let the smoothstep function be defined as:

$$
s(t) = t^2 (3 - 2t).
$$

We fix the gating constants $left = 0$ and $right = 0.3$. For each coordinate in the concatenated vectors $i = 1, \dots, m$, we define

$$
x_i = \max(|a_i|, |b_i|), \quad
t_i = \frac{x_i - left}{right - left} = \frac{x_i}{0.3}, \quad
\tilde{t}_i = \min(1, \max(0, t_i)),
$$

and the (smooth) gate weight

$$
w_i = s(\tilde{t}_i).
$$

Let $w_i^2$ denote the squared weights. Then the weighted cosine similarity is

$$
Wcos(a, b) = \frac{\sum_{i=1}^{m} w_i^2 a_i b_i}{\big(\sum_{i=1}^{m} w_i^2 a_i^2\big)^{1/2} \, \big(\sum_{i=1}^{m} w_i^2 b_i^2\big)^{1/2}}.
$$

(If the denominator is zero, define $Wcos(a, b) = 0$.)


### Final score
$$
W\times\max(0, Wcos)
$$

## Approach