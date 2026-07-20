"""Prior distributions used by GeneFlow."""

import torch


class PriorSampler:
    def __init__(self, prior_sample_type, **kwargs):
        self.prior_sample_type = prior_sample_type

        if prior_sample_type == "gaussian":
            self.prior_sampler = gaussian_prior
        elif prior_sample_type == "zero":
            self.prior_sampler = all_zeros
        elif prior_sample_type == "zinb":
            from scvi.distributions import ZeroInflatedNegativeBinomial

            distribution = ZeroInflatedNegativeBinomial(
                total_count=kwargs.get("total_count"),
                logits=kwargs.get("logits"),
                zi_logits=kwargs.get("zi_logits"),
            )
            self.prior_sampler = (
                lambda shape: distribution.sample(shape).squeeze(-1)
            )
        else:
            raise ValueError(
                "Invalid prior sample type. Choose gaussian, zero, or zinb."
            )

    def sample(self, shape):
        return self.prior_sampler(shape)


def gaussian_prior(shape):
    return torch.randn(shape)


def all_zeros(shape):
    return torch.zeros(shape)
