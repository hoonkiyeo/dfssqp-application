# References

- **Na, S. (2025).** [Derivative-Free Sequential Quadratic Programming for
  Equality-Constrained Stochastic Optimization](https://arxiv.org/abs/2510.22458v1).
  arXiv:2510.22458. The DF-SSQP algorithm used in this application.

- **Chayti, E. M., El Bakkali El Kadi, T., Saadi, O., and Jaggi, M. (2025).**
  [Stochastic Optimization with Random Search](https://arxiv.org/abs/2510.15610v2).
  arXiv:2510.15610, v2. Source for the stochastic random-search iteration
  (Algorithm 1 and Section 4, with one shared stochastic sample for the two candidates).
  Our application adds normalization of the selected candidate to enforce unit norm;
  exact loss ties select the positive candidate.

- **Spall, J. C. (1992).** Multivariate Stochastic Approximation Using a Simultaneous
  Perturbation Gradient Approximation. *IEEE Transactions on Automatic Control*,
  37, 332–341. [Author's reference list](https://www.jhuapl.edu/SPSA/Pages/References-List_Ref.htm).
  Background for the two-evaluation gradient estimator within DF-SSQP.

- **He, M., Wang, Y., Zeng, Q., and Zhang, Y. (2023).**
  [Forecasting aggregate stock market volatility with industry volatilities:
  The role of spillover index](https://doi.org/10.1016/j.ribaf.2023.101983).
  *Research in International Business and Finance*, 65, 101983.
  Application background and the use of Mean/PCA industry summaries.

- **Kingma, D. P., and Ba, J. (2014).**
  [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
  arXiv:1412.6980. The optimizer used for inner neural-model training.
