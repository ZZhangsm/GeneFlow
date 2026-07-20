"""Flow matching objective and Euler sampler."""

import torch
import torch.nn.functional as F

from .noise import PriorSampler


class Interpolant:
    def __init__(self, prior_sample_type, normalize=True, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prior_sampler = PriorSampler(prior_sample_type, **kwargs)
        self.normalize = normalize

    def sample_from_prior(self, shape):
        expression = self.prior_sampler.sample(shape).to(self.device)
        if self.normalize:
            expression = torch.log(expression + 1)
        return expression

    @staticmethod
    def sample_t(shape):
        return torch.rand(shape)

    def corrupt_exp_x0(self, expression):
        timestep = self.sample_t((expression.shape[0],)).to(self.device)
        prior_expression = self.sample_from_prior(expression.shape)
        interpolated = (
            prior_expression * (1 - timestep[:, None])
            + expression * timestep[:, None]
        )
        return interpolated, prior_expression, timestep

    def flow_matching_loss(self, model, expression, image_features):
        """Compute the x-prediction flow-matching loss."""

        expression_t, expression_0, timestep = self.corrupt_exp_x0(
            expression
        )
        expression_prediction = model(
            exp=expression_t,
            t=timestep,
            img_features=image_features,
        )
        target_velocity = expression - expression_0
        predicted_velocity = expression_prediction - expression_0
        valid_mask = image_features.sum(-1) != 0
        return F.mse_loss(
            predicted_velocity[valid_mask], target_velocity[valid_mask]
        )

    @staticmethod
    def x_pred_to_velocity(
        expression_prediction,
        expression_t,
        timestep,
        min_denominator=5e-2,
    ):
        return (
            expression_prediction - expression_t
        ) / (1 - timestep[:, None]).clamp(min_denominator)

    def euler_step(
        self,
        model,
        expression_t,
        image_features,
        start_time,
        end_time,
    ):
        batch_size = expression_t.shape[0]
        device = expression_t.device
        start_vector = torch.full(
            (batch_size,), float(start_time), device=device
        )
        delta_vector = torch.full(
            (batch_size,), float(end_time - start_time), device=device
        )
        prediction = model(
            expression_t, start_vector, image_features
        )
        velocity = self.x_pred_to_velocity(
            prediction, expression_t, start_vector
        )
        return expression_t + delta_vector[:, None] * velocity

    @torch.no_grad()
    def sample(
        self,
        model,
        image_features,
        input_gene_size,
        n_sample_steps,
        t_start=0.01,
        t_end=1.0,
    ):
        """Generate expression by integrating the learned flow."""

        batch_size = image_features.shape[0]
        device = image_features.device
        expression_t = self.sample_from_prior(
            (batch_size, input_gene_size)
        ).to(device)
        timesteps = torch.linspace(
            t_start, t_end, n_sample_steps, device=device
        )
        for start_time, end_time in zip(
            timesteps[:-1], timesteps[1:]
        ):
            expression_t = self.euler_step(
                model,
                expression_t,
                image_features,
                start_time,
                end_time,
            )
        return expression_t
