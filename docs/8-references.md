# References — what each source contributed

Chess architecture (the proven body and heads)
- Monroe, Eilender, Chalmers, Tang, Anderson. *Chessformer: A Unified Architecture
  for Chess Modeling.* ICLR 2026. arXiv:2605.19091. **Primary source** for the
  square-token encoder, GAB, source→destination policy head, configs, the
  Leela-CF vs Leela-CNN +112 ± 7 Elo result, and the GAB ablation (+83 Elo).
- Monroe, Chalmers. *Mastering Chess with a Transformer Model.* arXiv:2409.12272
  (superseded by the above). Training recipe: Nadam, grad clip 10, SWA,
  CF-6M LR schedule, 53M games of April 2024 Lc0 data.
- Lc0 blog, *Transformer Progress* (2024-02): https://lczero.org/blog/2024/02/transformer-progress/
  BT1–BT4 configs, smolgen description and cost, head dim 32, "FFN 4× gives
  little", new input embedding.

Chess + MoE
- *Checkmating One, by Using Many: Combining Mixture of Experts with MCTS to
  Improve in Chess* (M2CTS). arXiv:2401.16852. Phase experts inside MCTS
  (+122 ± 23 Elo), plus the limitations that motivate a learned router and a
  shared expert.
- *Mixture of Experts Made Intrinsically Interpretable* (MoE-X). arXiv:2503.07639.
  Sparse MoE transformer evaluated on chess move prediction; interpretability, not strength.

MoE methods
- Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer.* ICLR 2017.
- Fedus, Zoph, Shazeer. *Switch Transformers.* arXiv:2101.03961.
- Zoph et al. *ST-MoE: Designing Stable and Transferable Sparse Expert Models.*
  arXiv:2202.08906. Router z-loss.
- Riquelme et al. *Scaling Vision with Sparse MoE* (V-MoE). arXiv:2106.05974.
  MoE in alternate layers for token grids (ablation M4).
- Dai et al. *DeepSeekMoE.* arXiv:2401.06066. Fine-grained experts plus a shared
  expert (main design).
- Wang et al. *Auxiliary-Loss-Free Load Balancing Strategy for MoE.*
  arXiv:2408.15664. Bias-based balancing (main design).
- DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437. Production use
  of aux-loss-free balancing plus a tiny complementary balance loss.
- Puigcerver et al. *From Sparse to Soft Mixtures of Experts.* arXiv:2308.00951.
  Static-shape alternative (ablation M3).
- Zhou et al. *Mixture-of-Experts with Expert Choice Routing.* arXiv:2202.09368.
  Considered and rejected: expert-choice lets a square get zero experts, and
  per-position batch composition makes routing depend on other positions in
  the batch, which breaks determinism in search.

Data and formats
- Lc0 training data: https://storage.lczero.org/files/training_data/ (licence:
  `LICENSE.txt` → ODbL 1.0 / DbCL 1.0).
- Record format: `lc0/src/trainingdata/trainingdata_v6.h` (`V6TrainingData`,
  8,356 bytes, GPL-3.0 source).
- lczero-training: https://github.com/LeelaChessZero/lczero-training (reference
  loader and training configs).
