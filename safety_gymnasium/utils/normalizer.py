# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Running mean/standard-deviation normalization."""

from __future__ import annotations

from typing import Any

import numpy as np


__all__ = ['MeanStdNormalizer', 'RunningMeanStd']


class RunningMeanStd:
    """Track a running mean and variance using Chan's parallel-variance algorithm.

    This is a self-contained implementation on purpose. Gymnasium has moved
    ``RunningMeanStd`` between ``gymnasium.wrappers.normalize``,
    ``gymnasium.utils.running_mean_std`` and ``gymnasium.wrappers.utils`` across
    releases, and depending on it means carrying import fallbacks forever.

    Attributes:
        mean (np.ndarray): Running mean.
        var (np.ndarray): Running variance.
        count (float): Running sample count, seeded with ``epsilon`` to keep the first
            update well defined.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        """Initialize an instance of :class:`RunningMeanStd`.

        Args:
            shape (tuple): Shape of the tracked statistics.
            epsilon (float): Initial pseudo-count.
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray) -> None:
        """Update the statistics with a batch of samples stacked along axis 0."""
        batch = np.asarray(batch, dtype=np.float64)
        self.update_from_moments(
            batch_mean=np.mean(batch, axis=0),
            batch_var=np.var(batch, axis=0),
            batch_count=batch.shape[0],
        )

    def update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """Merge externally computed batch moments into the running statistics."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count


class MeanStdNormalizer:
    """Normalize batches of data by a running mean and standard deviation.

    The statistics are updated only while the normalizer is writable, which lets a single
    normalizer be shared between a training environment and an evaluation environment:
    evaluation reads the latest training statistics without contributing to them. Call
    :meth:`set_read_only` before evaluating and :meth:`unset_read_only` afterwards.

    The tracked shape is derived lazily from the first batch as ``(1,) + batch.shape[1:]``,
    i.e. statistics are per feature and pooled over the leading (batch) axis.

    Examples:
        >>> normalizer = MeanStdNormalizer()
        >>> normalizer.unset_read_only()
        >>> normalized = normalizer(np.stack([obs_env0, obs_env1]))
        >>> normalizer.set_read_only()
        >>> saved = normalizer.state_dict()
    """

    def __init__(
        self,
        clip: float = 50.0,
        epsilon: float = 1e-20,
        read_only: bool = True,
    ) -> None:
        """Initialize an instance of :class:`MeanStdNormalizer`.

        Args:
            clip (float): Normalized values are clipped to ``[-clip, clip]``.
            epsilon (float): Stability term added to the *variance* before taking the
                square root.
            read_only (bool): When ``True``, :meth:`__call__` normalizes without updating
                the running statistics.
        """
        self.clip = clip
        self.epsilon = epsilon
        self.read_only = read_only
        self.rms: RunningMeanStd | None = None

    def set_read_only(self) -> None:
        """Stop updating the running statistics."""
        self.read_only = True

    def unset_read_only(self) -> None:
        """Resume updating the running statistics."""
        self.read_only = False

    def __call__(self, batch: np.ndarray, update: bool | None = None) -> np.ndarray:
        """Normalize a batch, updating the statistics first unless read-only.

        Args:
            batch (np.ndarray): Samples stacked along axis 0.
            update (bool): Whether to fold this batch into the running statistics.
                Defaults to ``not self.read_only``. Pass it explicitly when several
                callers share one normalizer and each needs its own read/write policy,
                so that one caller freezing itself cannot freeze the others.
        """
        batch = np.asarray(batch)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1, *batch.shape[1:]))
        if update is None:
            update = not self.read_only
        if update:
            self.rms.update(batch)
        normalized = (batch - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip)

    def unnormalize(self, batch: np.ndarray) -> np.ndarray:
        """Invert :meth:`__call__`, ignoring the clipping."""
        if self.rms is None:
            raise RuntimeError('Cannot unnormalize before any statistics have been collected.')
        batch = np.asarray(batch)
        return batch * np.sqrt(self.rms.var + self.epsilon) + self.rms.mean

    def state_dict(self) -> dict[str, Any] | None:
        """Return the running statistics, or ``None`` if nothing has been seen yet."""
        if self.rms is None:
            return None
        return {'mean': self.rms.mean, 'var': self.rms.var, 'count': self.rms.count}

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        """Restore running statistics produced by :meth:`state_dict`."""
        if state is None:
            return
        if self.rms is None:
            self.rms = RunningMeanStd(shape=np.shape(state['mean']))
        self.rms.mean = np.asarray(state['mean'], dtype=np.float64)
        self.rms.var = np.asarray(state['var'], dtype=np.float64)
        self.rms.count = state['count']
