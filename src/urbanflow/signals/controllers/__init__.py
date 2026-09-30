"""Signal controllers: the protocol, context and registry, plus the built-ins (plan H.3, H.4).

Importing this package registers ``fixed_time`` and ``external``.
"""

from urbanflow.signals.controllers.base import (
    ControllerBase,
    ControllerContext,
    ControllerRef,
    ControllerSetup,
    EmptyParams,
    LaneData,
    SignalController,
    controller_name,
    controller_registry,
    create_controller,
    register_controller,
)
from urbanflow.signals.controllers.external import External
from urbanflow.signals.controllers.fixed_time import (
    FixedTime,
    FixedTimeParams,
    cycle_position,
    realised_cycle,
    stage_steps,
)

__all__ = [
    "ControllerBase",
    "ControllerContext",
    "ControllerRef",
    "ControllerSetup",
    "EmptyParams",
    "External",
    "FixedTime",
    "FixedTimeParams",
    "LaneData",
    "SignalController",
    "controller_name",
    "controller_registry",
    "create_controller",
    "cycle_position",
    "realised_cycle",
    "register_controller",
    "stage_steps",
]
