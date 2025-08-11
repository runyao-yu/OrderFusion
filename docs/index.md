<p align="right" style="margin:0 0 8px 0;">
  <img src="assets/affiliations.PNG" alt="Affiliations" height="56" style="max-width:40vw;">
</p>

---
title: Foundation Model for Probabilistic Intraday Electricity Price Forecasting
description: Foundation Model for Probabilistic Intraday Electricity Price Forecasting
---
<p align="left">
  <img src="assets/Trade.gif" alt="Teaser" width="128">
</p>

# OrderFusion

[![arXiv](https://img.shields.io/badge/arXiv-2502.06830-b31b1b.svg)](https://arxiv.org/abs/2502.06830)
[![Code](https://img.shields.io/badge/GitHub-Repository-181717.svg)](https://github.com/runyao-yu/OrderFusion)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/runyao-yu/)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?logo=gmail&logoColor=white)](mailto:runyao.yu@tudelft.nl)

**Authors:** Runyao Yu, Yuchen Tao, Fabian Leimgruber, Tara Esterl, Jochen L. Cremer

## Abstract
Accurate and reliable probabilistic forecasting of intraday electricity prices is essential to manage market uncertainties and support robust trading strategies. However, current methods rely heavily on domain feature extraction and fail to capture the dynamics between buy and sell orders, limiting the ability to form rich representations of the orderbook. Furthermore, these methods often require training separate models for different quantiles and face the quantile crossing issue, where predicted upper quantiles fall below lower ones. To address these challenges, we propose an end-to-end foundation model called OrderFusion, tailored for the intraday electricity market. OrderFusion encodes the orderbook into a 2.5D representation and employs a jump fusion backbone to model buy-sell dynamics without the need for domain feature extraction. The head anchors on the median quantile and hierarchically estimates other quantiles through constrained residuals, ensuring monotonicity. We conduct extensive experiments and ablation studies on three key price indices (ID1, ID2, and ID3) using three years of orderbook data from the German market. To assess the generalizability of the proposed foundation model, we further evaluate it on the Austrian market. The results confirm that OrderFusion remains accurate, reliable, and generalizable across different market settings.

## Model Structure
![Model structure](assets/model_structure.PNG)

## Citation

```bibtex
@misc{yu2025OrderFusion,
  title         = {OrderFusion: Foundation Model for Probabilistic Intraday Electricity Price Forecasting Using Orderbook},
  author        = {Yu, Runyao and Tao, Yuchen and Leimgruber, Fabian and Esterl, Tara and Cremer, Jochen L.},
  year          = {2025},
  eprint        = {2502.06830},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.CP},
  url           = {https://arxiv.org/abs/2502.06830}
}
```