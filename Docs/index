---
title: OrderFusion
description: Foundation Model for Probabilistic Intraday Electricity Price Forecasting
---

# OrderFusion

[![arXiv](https://img.shields.io/badge/arXiv-2502.06830-b31b1b.svg)](https://arxiv.org/abs/2502.06830)
[![Code](https://img.shields.io/badge/GitHub-Repository-181717.svg)](https://github.com/runyao-yu/OrderFusion)

**Authors:** Runyao Yu, Yuchen Tao, Fabian Leimgruber, Tara Esterl, Jochen L. Cremer

## Abstract
Accurate and reliable probabilistic forecasting of intraday electricity prices is essential to manage market uncertainties and support robust trading strategies. However, current methods rely heavily on domain feature extraction and fail to capture the dynamics between buy and sell orders, limiting the ability to form rich representations of the orderbook. Furthermore, these methods often require training separate models for different quantiles and face the quantile crossing issue, where predicted upper quantiles fall below lower ones. To address these challenges, we propose an end-to-end foundation model called OrderFusion, tailored for the intraday electricity market. OrderFusion encodes the orderbook into a 2.5D representation and employs a jump fusion backbone to model buy-sell dynamics without the need for domain feature extraction. The head anchors on the median quantile and hierarchically estimates other quantiles through constrained residuals, ensuring monotonicity. We conduct extensive experiments and ablation studies on three key price indices (ID1, ID2, and ID3) using three years of orderbook data from the German market. To assess the generalizability of the proposed foundation model, we further evaluate it on the Austrian market. The results confirm that OrderFusion remains accurate, reliable, and generalizable across different market settings.

![Teaser](assets/trading.gif)

<iframe width="100%" height="420" src="https://www.youtube.com/embed/XXXXXXXX" frameborder="0" allowfullscreen></iframe>

## Citation
```bibtex
@misc{yu2025orderfusion,
  title         = {OrderFusion: Encoding Orderbook for End-to-End Probabilistic Intraday Electricity Price Prediction},
  author        = {Yu, Runyao and Tao, Yuchen and Leimgruber, Fabian and Esterl, Tara and Cremer, Jochen L.},
  year          = {2025},
  eprint        = {2502.06830},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.CP},
  url           = {https://arxiv.org/abs/2502.06830}
}