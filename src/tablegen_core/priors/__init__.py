"""Static allowlist: only the sixteen selected extension nodes are imported."""
from .nodes.piecewise_route_engine import NODE_CLASS as _node_0
from .nodes.nori_anchor_rho_copy import NODE_CLASS as _node_1
from .nodes.gev_tail_warp import NODE_CLASS as _node_2
from .nodes.ewens_sampling_formula import RandomEwensSamplingFormulaFunction as _node_3
from .nodes.markov_state_error_emission import NODE_CLASS as _node_4
from .nodes.informative_censoring_hazard import NODE_CLASS as _node_5
from .nodes.eos_excess_mixing_enthalpy import NODE_CLASS as _node_6
from .nodes.modulated_satellite_structure_factor import NODE_CLASS as _node_7
from .nodes.paws_timestamp_drop import NODE_CLASS as _node_8
from .nodes.niche_model_food_web import NODE_CLASS as _node_9
from .nodes.island_biogeography_equilibrium import NODE_CLASS as _node_10
from .nodes.bourgouin_precipitation_type import RandomBourgouinPrecipitationTypeFunction as _node_11
from .nodes.temperature_dependent_instar_transition import RandomTemperatureDependentInstarTransitionFunction as _node_12
from .nodes.tetragonal_interplanar_spacing import RandomTetragonalInterplanarSpacingFunction as _node_13
from .nodes.bch_error_locator import NODE_CLASS as _node_14
from .nodes.moe_linear_experts import NODE_CLASS as _node_15

NODES = {
    'piecewise_route_engine': _node_0,
    'nori_anchor_rho_copy': _node_1,
    'gev_tail_warp': _node_2,
    'ewens_sampling_formula': _node_3,
    'markov_state_error_emission': _node_4,
    'informative_censoring_hazard': _node_5,
    'eos_excess_mixing_enthalpy': _node_6,
    'modulated_satellite_structure_factor': _node_7,
    'paws_timestamp_drop': _node_8,
    'niche_model_food_web': _node_9,
    'island_biogeography_equilibrium': _node_10,
    'bourgouin_precipitation_type': _node_11,
    'temperature_dependent_instar_transition': _node_12,
    'tetragonal_interplanar_spacing': _node_13,
    'bch_error_locator': _node_14,
    'moe_linear_experts': _node_15,
}
