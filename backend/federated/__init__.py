"""Federated orchestration layer: secure aggregation, DP, QAOA strategy."""

from backend.federated.aggregator import build_aggregation_qubo, select_cohort
from backend.federated.client import ClinicNode
from backend.federated.server import FederationServer

__all__ = ["ClinicNode", "FederationServer", "build_aggregation_qubo", "select_cohort"]
