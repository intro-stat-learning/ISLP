r"""
We want to evaluate

$$ \int_{\mathbb{R}} \frac{1}{(2\pi\sigma^2)^{n/2}} \frac{1}{\sqrt{2
\pi \sigma^2_{\mu}}} \exp\left(-\frac{1}{2\sigma^2} \sum_{i=1}^n
(R_i-\mu)^2 - \frac{1}{2 \sigma^2_{\mu}}(\mu-\mu_0)^2 \right) \; d\mu
$$


Clearly depends on $R$ only through $\bar{R}$ and $\|R-\bar{R}\|^2_2$
$$ = \frac{\exp \left(-\frac{1}{2\sigma^2}
\|R-\bar{R}\|^2_2\right)}{(2\pi\sigma^2)^{n/2}} \int_{\mathbb{R}}
\frac{1}{\sqrt{2 \pi \sigma^2_{\mu}}} \exp\left(-\frac{n}{2\sigma^2}
(\bar{R}-\mu)^2 - \frac{1}{2 \sigma^2_{\mu}}(\mu-\mu_0)^2 \right) \;
d\mu $$


Expanding: $$ =\frac{\exp\left(-\frac{1}{2\sigma^2}
\|R-\bar{R}\|^2_2-\frac{n}{2\sigma^2}\bar{R}^2-\frac{\mu_0^2}{2\sigma^2_{\mu}}\right)}{(2\pi\sigma^2)^{n/2}\sqrt{2
\pi \sigma^2_{\mu}}} \int_{\mathbb{R}}
\exp\left(-\left(\frac{n}{2\sigma^2} + \frac{1}{2
\sigma^2_{\mu}}\right)\mu^2 + \mu \left(\frac{\bar{R} \cdot
n}{\sigma^2} + \frac{\mu_0}{\sigma^2_{\mu}}\right)\right) \; d\mu $$


Setting $$ \bar{\sigma}^{2} = \left(\frac{n}{2 \sigma^2} + \frac{1}{2
\sigma^2_{\mu}}\right)^{-1} $$ this is $$
=\frac{\exp\left(-\frac{1}{2\sigma^2}
\|R\|^2_2-\frac{\mu_0^2}{2\sigma^2_{\mu}}\right)}{(2\pi\sigma^2)^{n/2}\sqrt{2
\pi \sigma^2_{\mu}}} \cdot \sqrt{2 \pi \bar{\sigma}^2}
\int_{\mathbb{R}} \frac{1}{\sqrt{2\pi\bar{\sigma}^2}}
\exp\left(-\frac{1}{2 \bar{\sigma}^2} \mu^2 + \mu \left(\frac{\bar{R}
\cdot n}{\sigma^2} + \frac{\mu_0}{\sigma^2_{\mu}}\right)\right) \;
d\mu $$


Finally, this is $$ =\frac{\exp\left(-\frac{1}{2\sigma^2}
\|R\|^2_2-\frac{\mu_0^2}{2\sigma^2_{\mu}}\right)}{(2\pi\sigma^2)^{n/2}\sqrt{2
\pi \sigma^2_{\mu}}} \cdot \sqrt{2 \pi \bar{\sigma}^2}
\exp\left(\frac{\bar{\sigma}^2}{2} \left(\frac{\bar{R} \cdot
n}{\sigma^2} + \frac{\mu_0}{\sigma^2_{\mu}}\right)^2 \right) $$


Or, setting $$ \bar{\mu} = \frac{\frac{\bar{R} \cdot n}{\sigma^2} +
\frac{\mu_0}{\sigma^2_{\mu}}}{\frac{n}{\sigma^2} +
\frac{1}{\sigma^2_{\mu}}} $$ this is $$
=\frac{\exp\left(-\frac{1}{2\sigma^2} \|R\|^2_2
-\frac{\mu_0^2}{2\sigma^2_{\mu}}\right)}{(2\pi\sigma^2)^{n/2}\sqrt{2
\pi \sigma^2_{\mu}}} \cdot \sqrt{2 \pi \bar{\sigma}^2}
\exp\left(\frac{1}{2\bar{\sigma}^2} \bar{\mu}^2 \right) $$

"""

import numpy as np

def marginal_loglikelihood(response,
                           sigmasq,
                           mu_prior_mean,
                           mu_prior_var,
                           incremental=False,
                           response_moments=None):
    if response_moments is None:
        n = response.shape[0]
        response_sum = response.sum()
        if not incremental:
            responsesq_sum = (response**2).sum()
        else:
            responsesq_sum = None
        response_moments = (n, response_sum, responsesq_sum)
    else:
        n, response_sum, responsesq_sum = response_moments
    if response_sum is None:
        n, response_sum = response.shape[0], response.sum()

    sigmasq_bar = 1 / (n / sigmasq + 1 / mu_prior_var)
    mu_bar = (response_sum / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar

    logL = (0.5 * np.log(sigmasq_bar / mu_prior_var) +
            0.5 * (mu_bar**2 / sigmasq_bar))
    logL -= 0.5 * mu_prior_mean**2 / mu_prior_var

    if not incremental:
        if responsesq_sum is None:
            responsesq_sum = (response**2).sum()
            response_moments = (n, response_sum, responsesq_sum)
            
        logL -= n * 0.5 * np.log(sigmasq)
        logL -= 0.5 * responsesq_sum / sigmasq
                
    return logL, response_moments

def incremental_loglikelihood(response,
                              idx_L,
                              idx_R,
                              sigmasq,
                              mu_prior_mean,
                              mu_prior_var):
    r_L = response[idx_L]
    r_R = response[idx_R]
    n_L, n_R = r_L.shape[0], r_R.shape[0]
    sum_L, sum_R = r_L.sum(), r_R.sum()
    sumsq_L, sumsq_R = None, None 

    sum_f = sum_L + sum_R
    
    # for idx_L

    sigmasq_bar_L = 1 / (n_L / sigmasq + 1 / mu_prior_var)
    mu_bar_L = (sum_L / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar_L

    logL_L = (0.5 * np.log(sigmasq_bar_L / mu_prior_var) +
              0.5 * (mu_bar_L**2 / sigmasq_bar_L))
    logL_L -= 0.5 * mu_prior_mean**2 / mu_prior_var
                
    # for idx_R

    sigmasq_bar_R = 1 / (n_R / sigmasq + 1 / mu_prior_var)
    mu_bar_R = (sum_R / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar_R

    logL_R = (0.5 * np.log(sigmasq_bar_R / mu_prior_var) +
            0.5 * (mu_bar_R**2 / sigmasq_bar_R))
    logL_R -= 0.5 * mu_prior_mean**2 / mu_prior_var
                
    # for full: union of idx_L and idx_R

    sigmasq_bar_f = 1 / ((n_L + n_R) / sigmasq + 1 / mu_prior_var)
    mu_bar_f = (sum_f / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar_f

    logL_f = (0.5 * np.log(sigmasq_bar_f / mu_prior_var) +
            0.5 * (mu_bar_f**2 / sigmasq_bar_f))
    logL_f -= 0.5 * mu_prior_mean**2 / mu_prior_var

    return logL_L + logL_R - logL_f, (n_L, sum_L, sumsq_L), (n_R, sum_R, sumsq_R)


