<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# what have people done for challenge data ens 22 Volatility prediction in financial markets? what techniques are commonly used and could help get a great benchmark score

The CFM × ENS Challenge Data \#22 asks you to predict end-of-day volatility of US equities using past (beginning-of-day) returns and historical volatilities across multiple instruments. The metric is typically R² (coefficient of determination). Here's a comprehensive breakdown of what people have done and what works best.

## Challenge Setup

The dataset from **Capital Fund Management (CFM)** provides historical beginning-of-day returns and past volatilities for a universe of financial instruments. The goal is to regress future volatility from cross-sectional and time-series features — a multi-asset, panel regression problem.[^1][^2]

## Baseline: HAR-RV Model

The standard starting point for any realized volatility challenge is the **Heterogeneous Autoregressive model of Realized Volatility (HAR-RV)**, proposed by Corsi (2004). It decomposes volatility into three components:[^3]

- **Daily lag** (RV at t-1)
- **Weekly lag** (average RV over t-1 to t-5)
- **Monthly lag** (average RV over t-1 to t-22)

This captures the well-known "long memory" of volatility and is a strong linear baseline that is hard to beat without careful feature engineering.[^3]

## Feature Engineering (Key to High Scores)

This is where most top solutions differentiate themselves:

- **Multi-lag rolling windows** — beyond the standard 1/5/22 lags, adding 10-day, 63-day (quarterly), and 252-day (annual) rolling averages of past volatility significantly improves fit
- **Cross-asset features** — including volatilities of correlated instruments (sector peers, index) as predictors, since the dataset spans many assets[^1]
- **Return-based features** — signed and unsigned lagged returns, squared returns as proxy for realized variance, and rolling return dispersion
- **Volatility ratios** — ratio of short-term to long-term volatility (captures regime shifts and mean-reversion signals)
- **Rank/normalization** — cross-sectional ranking of volatility features per time period, reducing the impact of outliers


## Tree-Based Models (Best Single Models)

Gradient boosting — particularly **XGBoost** and **LightGBM** — consistently outperform linear models when given well-engineered lag features:[^4][^5]

- They handle non-linearities in the volatility term structure naturally
- They are robust to outliers (large volatility spikes from market events)
- Feature importance outputs help prune irrelevant lags
- Hyperparameter tuning via early stopping on a temporal validation set is critical (never shuffle financial time series for validation)


## Deep Learning Approaches

For further gains, several architectures have been applied:[^6][^7]

- **LSTM / BiLSTM** — model the sequential autocorrelation structure of volatility directly; particularly effective when the input is formatted as a time series per asset
- **Attention + BiLSTM hybrids** — the attention mechanism assigns dynamic weights to different time lags, useful since volatility memory is not uniform[^8]
- **Transformer-based models** — effective for long-sequence volatility forecasting, though they tend to overfit on smaller panel datasets[^6]


## Ensemble Methods (Top Benchmark Scores)

The **Kairos package** was specifically developed by community participants for this CFM/ENS challenge and applies advanced ML ensemble techniques. General best practices for ensembling:[^9]

- Blend HAR-RV (linear, low-variance) with XGBoost/LightGBM (non-linear, higher-variance)
- Stack predictions from daily, weekly, and monthly horizon models
- Use **out-of-fold predictions** from time-series cross-validation (rolling window, not random) to train the meta-learner


## Practical Tips to Maximize R²

| Technique | Why it helps |
| :-- | :-- |
| Log-transform volatility | Normalizes the heavy-tailed distribution [^10] |
| Temporal cross-validation | Prevents data leakage across time [^4] |
| Cross-sectional normalization per day | Removes market-wide volatility regime from features |
| LASSO/Ridge for feature selection | Prunes redundant lags, reduces overfitting [^4] |
| HAR + XGBoost stack | Combines statistical robustness with ML flexibility [^7] |

LASSO-based forecasts have been shown to achieve Sharpe ratio improvements of **24–42%** over benchmark portfolios in similar volatility-prediction setups, suggesting strong out-of-sample generalization. The key insight across all top solutions is that **feature engineering around multi-scale volatility lags** matters more than model complexity alone — a well-tuned XGBoost on rich HAR-style features will consistently beat a vanilla deep learning model on tabular volatility data.[^4][^9]
<span style="display:none">[^11][^12][^13][^14][^15][^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30]</span>

<div align="center">⁂</div>

[^1]: https://challengedata.ens.fr/challenges/22

[^2]: https://www.college-de-france.fr/en/agenda/seminar/apprenticeship-and-the-curse-of-large-dimensions/2018-challenges-presentation-1

[^3]: https://github.com/deep-hedger-Peng/HAR-RV

[^4]: https://www.sciencedirect.com/science/article/abs/pii/S0275531924005166

[^5]: https://systems.enpress-publisher.com/index.php/jipd/article/view/4972

[^6]: http://gsconlinepress.com/journals/gscarr/sites/default/files/GSCARR-2025-0163.pdf

[^7]: https://onlinelibrary.wiley.com/doi/10.1155/2022/8952996

[^8]: https://arxiv.org/pdf/2401.11621.pdf

[^9]: https://github.com/tomespel/kairos-volatility-prediction

[^10]: https://www.longdom.org/open-access/forecasting-financial-market-volatility-methodologies-challenges-and-implications-101831.html

[^11]: https://challengedata2.ens.fr/en/challenge/34/volatility_prediction_in_financial_markets.html

[^12]: https://challengedata.ens.fr/challenges/year/2018

[^13]: https://arxiv.org/abs/2307.13422

[^14]: https://combinatorialpress.com/jcmcc-articles/volume-124/research-on-the-construction-of-financial-market-volatility-prediction-model-in-digital-economy-environment-based-on-machine-learning-algorithm/

[^15]: http://rama.cont.perso.math.cnrs.fr/pdf/clustering.pdf

[^16]: https://arxiv.org/html/2108.00480v4

[^17]: https://zenodo.org/records/17678443/files/9-20.pdf?download=1

[^18]: https://www.ensae.fr/en/education/ingenieur-ensae-program/third-year/finance-risks-and-data

[^19]: https://challengedata.ens.fr

[^20]: https://radar.inria.fr/rapportsactivite/intranet/PDF-2019/resultats-Digital.pdf

[^21]: https://aclanthology.org/W17-55.pdf

[^22]: https://github.com/ljmdeb/ChallengeData2019

[^23]: https://github.com/Hupperich-Manuel/LSTM-XGBoost-Hybrid-Forecasting

[^24]: https://abfer.org/media/abfer-events-2024/annual-conference/papers-investment/AC24P3033-The-Low-Frequency-Trading-Arms-Race-Machines-Versus-Delays.pdf

[^25]: https://www.imo.universite-paris-saclay.fr/~yannig.goude/Materials/ProjetMLF/intro2024.pdf

[^26]: https://arxiv.org/pdf/2510.23150.pdf

[^27]: https://ijaers.com/complete-issue/IJAERS-Sep2022-Issue-Complete-Issue(v9i9).pdf

[^28]: https://dachxiu.chicagobooth.edu/download/ML_supp.pdf

[^29]: https://www.dechert.com/content/dam/dechert files/knowledge/publication/2019/2/EuroHedge.pdf

[^30]: https://www.ngfs.net/system/files/2025-01/NGFS Climate Scenarios Technical Documentation.pdf

