"""Neural and statistical models for point-in-time V26 predictions.

Importing this package does not require PyTorch.  Neural model constructors
raise :class:`TorchUnavailableError` with installation guidance when the
optional runtime is absent; NumPy-based stacking and calibration remain usable.
"""

from ._torch import TorchUnavailableError, torch_available
from .baseline import DEFAULT_CATBOOST_PARAMETERS, CatBoostClassifierBaseline
from .heads import (
    DirectPlayerGameHead,
    DirectRunDistributionHead,
    RunDistributionHead,
    WDLHead,
)
from .interaction import (
    DEFAULT_PA_OUTCOMES,
    PAInteractionDecoder,
    PlateAppearanceInteractionDecoder,
)
from .player_encoder import (
    PLAYER_ROLES,
    PlayerEncoder,
    PlayerRole,
    RoleAwarePlayerEncoder,
)
from .relgnn import CompositeRelGNNBackbone, RelGNNBackbone, RelGNNState
from .stacking import (
    BinaryIsotonicCalibrator,
    OOFPredictionSet,
    OOFProbabilityStacker,
    OOFStackingPipeline,
    StagewiseTemperatureCalibrator,
    TemperatureCalibrator,
    TemporalOOFStackingPipeline,
)

__all__ = [
    "BinaryIsotonicCalibrator",
    "CatBoostClassifierBaseline",
    "CompositeRelGNNBackbone",
    "DEFAULT_CATBOOST_PARAMETERS",
    "DEFAULT_PA_OUTCOMES",
    "DirectPlayerGameHead",
    "DirectRunDistributionHead",
    "OOFPredictionSet",
    "OOFProbabilityStacker",
    "OOFStackingPipeline",
    "PAInteractionDecoder",
    "PLAYER_ROLES",
    "PlateAppearanceInteractionDecoder",
    "PlayerEncoder",
    "PlayerRole",
    "RelGNNBackbone",
    "RelGNNState",
    "RoleAwarePlayerEncoder",
    "RunDistributionHead",
    "StagewiseTemperatureCalibrator",
    "TemperatureCalibrator",
    "TemporalOOFStackingPipeline",
    "TorchUnavailableError",
    "WDLHead",
    "torch_available",
]
