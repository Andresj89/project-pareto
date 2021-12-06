from pyomo.environ import (
    Var,
    Param,
    Set,
    ConcreteModel,
    Constraint,
    Objective,
    minimize,
    NonNegativeReals,
    Reals,
    Binary,
)
from pareto.utilities.get_data import get_data
from importlib import resources
import pyomo.environ

# import gurobipy
from pyomo.common.config import ConfigBlock, ConfigValue, In
from enum import Enum

from pareto.utilities.solvers import get_solver


def create_model(df_sets, df_parameters):
    model = ConcreteModel()

    # *******************************************************************************
    #                                 SET DEFINITION
    # *******************************************************************************

    model.s_C = Set(initialize=df_sets["Customers"], doc="Customers")
    model.s_T = Set(initialize=df_sets["TimePeriods"], doc="Time Periods", ordered=True)
    model.s_PP = Set(initialize=df_sets["ProductionPads"], doc="Production Pads")
    model.s_CP = Set(initialize=df_sets["CompletionsPads"], doc="Completions Pads")
    model.s_P = Set(initialize=(model.s_PP | model.s_CP), doc="Pads")
    model.s_F = Set(initialize=df_sets["FreshwaterSources"], doc="Freshwater Sources")
    model.s_G = Set(initialize=df_sets["TransportMode"], doc="Transportation mode")
    model.s_K = Set(initialize=df_sets["SWDSites"], doc="Disposal Sites")
    model.s_S = Set(initialize=df_sets["StorageSites"], doc="Storage Sites")
    model.s_N = Set(initialize=df_sets["NetworkNodes"], doc=["Network Nodes"])
    model.s_L = Set(
        initialize=(
            model.s_C | model.s_P | model.s_F | model.s_K | model.s_S | model.s_N
        ),
        doc="Locations",
    )

    # *******************************************************************************
    #                               PARAMETER DEFINITION
    # *******************************************************************************

    model.p_topology = Param(
        model.s_L,
        model.s_G,
        model.s_L,
        default=0,
        initialize=df_parameters["Topology"],
        doc="Connectivity between locations and available transport modes",
    )

    model.p_completionsdemand = Param(
        model.s_L,
        model.s_T,
        default=0,
        initialize=df_parameters["CompletionsDemand"],
        doc="Water demand for completion operations in location l and time period t",
    )

    model.p_customersdemand = Param(
        model.s_L,
        model.s_T,
        default=0,
        initialize=df_parameters["CustomersDemand"],
        doc="Water required by customers in location l and time period t",
    )

    model.p_disposalcost = Param(
        model.s_K,
        default=0,
        initialize=df_parameters["DisposalCost"],
        doc="Cost for disposing water in location in time period t",
    )

    model.p_freshwater = Param(
        model.s_L,
        model.s_T,
        default=0,
        initialize=df_parameters["FreshWaterAvailability"],
        doc="Fresh water available in location l and time period t",
    )

    model.p_freshwatercost = Param(
        model.s_L,
        default=0,
        initialize=df_parameters["FreshWaterCost"],
        doc="Cost of procuring fresh water available in location l",
    )

    model.p_storage = Param(
        model.s_L,
        default=0,
        initialize=df_parameters["StorageCapacity"],
        doc="Storage capacity in location l",
    )

    model.p_init_storage = Param(
        model.s_L,
        default=0,
        initialize=df_parameters["StorageInit"],
        doc="Initial storage in location l",
    )

    model.p_transportcapacity = Param(
        model.s_L,
        model.s_G,
        model.s_L,
        default=0,
        initialize=df_parameters["TransportCapacity"],
        doc="Transportation capacity between location l and l' via transport mode g",
    )

    model.p_transportcost = Param(
        model.s_L,
        model.s_G,
        model.s_L,
        default=0,
        initialize=df_parameters["TransportCost"],
        doc="Transportation cost between location l and l' via transport mode g, pipeline [$/mile-bbl], truck [$/mile-bbl]",
    )

    model.p_transportdistances = Param(
        model.s_L,
        model.s_G,
        model.s_L,
        default=99999,
        initialize=df_parameters["TransportDistances"],
        doc="Distance between location l and l' via transport mode g",
    )

    model.p_waterprofiles = Param(
        model.s_L,
        model.s_T,
        default=0,
        initialize=df_parameters["WaterProfiles"],
        doc="Water production profiles in location l and time period t",
    )

    # *******************************************************************************
    #                               VARIABLE DEFINITION
    # *******************************************************************************

    model.v_D = Var(
        model.s_L,
        model.s_T,
        within=NonNegativeReals,
        doc="Water demand at location l in time period t",
    )

    model.v_P = Var(
        model.s_L,
        model.s_T,
        within=NonNegativeReals,
        doc="Water production at location l in time period t",
    )

    model.v_Q = Var(
        model.s_L,
        model.s_G,
        model.s_L,
        model.s_T,
        within=NonNegativeReals,
        doc="Water transportation between locations l and l', via transport mode g in time period t",
    )

    model.v_S = Var(
        model.s_L,
        model.s_T,
        within=NonNegativeReals,
        doc="Water sotrage at location l in time period t",
    )

    model.v_C_DISPOSAL = Var(
        model.s_K,
        model.s_T,
        within=NonNegativeReals,
        doc="Operational cost for disposing water in site K in time period t",
    )

    model.v_C_FRESHWATER = Var(
        model.s_F,
        model.s_T,
        within=NonNegativeReals,
        doc="Operational cost for procuring fresh water from source F in time period t",
    )

    model.v_C_TRANSPORT = Var(
        model.s_L,
        model.s_G,
        model.s_L,
        model.s_T,
        within=NonNegativeReals,
        doc="Operational cost between location l and l' via transport mode g in time period t",
    )

    model.v_Z = Var(within=Reals, doc="Objective function variable [$]")

    # *******************************************************************************
    #                               MATHEMATICAL FORMULATION
    # *******************************************************************************

    # OBJECTIVE FUNCTION
    def objective_rule(m):
        return (
            sum(
                sum(
                    sum(
                        sum(
                            m.v_C_TRANSPORT[l, g, l_tilde, t]
                            for l in m.s_L
                            if m.p_topology[l, g, l_tilde]
                        )
                        for g in m.s_G
                    )
                    for l_tilde in m.s_L
                )
                for t in m.s_T
            )
            + sum(sum(m.v_C_DISPOSAL[l, t] for l in m.s_K) for t in m.s_T)
            + sum(sum(m.v_C_FRESHWATER[l, t] for l in m.s_F) for t in m.s_T)
        )

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # OPEX: DISPOSAL COSTS
    def disposal_cost_rule(m, l, t):
        return m.v_C_DISPOSAL[l, t] == sum(
            sum(
                m.p_disposalcost[l] * m.v_Q[l_tilde, g, l, t]
                for l_tilde in m.s_L
                if m.p_topology[l_tilde, g, l]
            )
            for g in m.s_G
        )

    model.disposal_cost_constraint = Constraint(
        model.s_K, model.s_T, rule=disposal_cost_rule
    )

    # OPEX: FRESH WATER COSTS
    def freshwater_cost_rule(m, l, t):
        return m.v_C_FRESHWATER[l, t] == m.p_freshwatercost[l] * m.v_P[l, t]

    model.freshwater_cost_constraint = Constraint(
        model.s_F, model.s_T, rule=freshwater_cost_rule
    )

    # OPEX: TRANSPORT COSTS
    def transport_cost_rule(m, l, g, l_tilde, t):
        if m.p_topology[l, g, l_tilde]:
            return (
                m.v_C_TRANSPORT[l, g, l_tilde, t]
                == m.p_transportcost[l, g, l_tilde]
                * m.p_transportdistances[l, g, l_tilde]
                * m.v_Q[l, g, l_tilde, t]
            )
        else:
            return Constraint.Skip

    model.transport_cost_constraint = Constraint(
        model.s_L, model.s_G, model.s_L, model.s_T, rule=transport_cost_rule
    )

    # GLOBAL MATERIAL BALANCE
    def material_balance_rule(m, l, t):
        if t == model.s_T.first():
            return (
                m.v_P[l, t]
                + sum(
                    sum(
                        m.v_Q[l_tilde, g, l, t]
                        for l_tilde in m.s_L
                        if m.p_topology[l_tilde, g, l]
                    )
                    for g in m.s_G
                )
                + m.p_init_storage[l]
                == m.v_D[l, t]
                + sum(
                    sum(
                        m.v_Q[l, g, l_tilde, t]
                        for l_tilde in m.s_L
                        if m.p_topology[l, g, l_tilde]
                    )
                    for g in m.s_G
                )
                + m.v_S[l, t]
            )

        else:
            return (
                m.v_P[l, t]
                + sum(
                    sum(
                        m.v_Q[l_tilde, g, l, t]
                        for l_tilde in m.s_L
                        if m.p_topology[l_tilde, g, l]
                    )
                    for g in m.s_G
                )
                + m.v_S[l, m.s_T.prev(t)]
                == m.v_D[l, t]
                + sum(
                    sum(
                        m.v_Q[l, g, l_tilde, t]
                        for l_tilde in m.s_L
                        if m.p_topology[l, g, l_tilde]
                    )
                    for g in m.s_G
                )
                + m.v_S[l, t]
            )

    model.material_balance = Constraint(
        model.s_L, model.s_T, rule=material_balance_rule
    )

    # FRESH WATER AVAILABILITY
    def fresh_water_rule(m, l, t):
        return m.v_P[l, t] <= m.p_freshwater[l, t]

    model.fresh_water_constraint = Constraint(
        model.s_F, model.s_T, rule=fresh_water_rule
    )

    # PRODUCED WATER FLOWBACK/PRODUCTION PROFILES
    # def water_profiles_rule(m, l, t):
    #     if l in m.s_P:
    #         return m.v_P[l, t] == m.p_waterprofiles[l, t]
    #     elif l in m.s_N or l in m.s_S or l in m.s_C or l in m.s_K:
    #         return m.v_P[l, t] == 0
    #     else:
    #         return Constraint.Skip

    # model.water_profiles_constraint = Constraint(
    #     model.s_P, model.s_T, rule=water_profiles_rule
    # )

    # The previous constraint "water_profiles_constraint" can be replace by the following code in which the variable v_P is fixed to
    # the already known production profiles for pads or to zero for locations that do not produce water
    # This helps in reducing the number of decision variables
    for i in model.v_P:
        if i[0] in model.s_P:
            model.v_P[i].fix(model.p_waterprofiles[i])
        elif not (i[0] in model.s_F):
            model.v_P[i].fix(0)

    # PRODUCED WATER DEMAND FOR COMPLETION OPERATIONS
    # def completion_demand_rule(m, l, t):
    #     return m.v_D[l, t] == m.p_completionsdemand[l, t]

    # model.completion_demand_constraint = Constraint(
    #     model.s_CP, model.s_T, rule=completion_demand_rule
    # )

    # CUSTOMERS DEMAND
    # def customers_demand_rule(m, l, t):
    #     return m.v_D[l, t] == m.p_customersdemand[l, t]

    # model.customers_demand_constraint = Constraint(
    #     model.s_C, model.s_T, rule=customers_demand_rule
    # )

    # The previous constraints "completion_demand_constraint" and "customers_demand_constraint" can be replace by the following code in which the variable v_D is fixed to
    # the already known water demand for completion operations and water demand from customers. The variable is set to zero for locations that do not demand water
    # This helps in reducing the number of decision variables
    for i in model.v_D:
        if i[0] in model.s_CP:
            model.v_D[i].fix(model.p_completionsdemand[i])
        elif i[0] in model.s_C:
            model.v_D[i].fix(model.p_customersdemand[i])
        else:
            model.v_D[i].fix(0)

    # TRANSPORT CAPACITY
    def transport_capacity_rule(m, l, g, l_tilde, t):
        if m.p_topology[l, g, l_tilde]:
            return m.v_Q[l, g, l_tilde, t] <= m.p_transportcapacity[l, g, l_tilde]
        else:
            return Constraint.Skip

    model.transport_capacity_constraint = Constraint(
        model.s_L, model.s_G, model.s_L, model.s_T, rule=transport_capacity_rule
    )

    # STORAGE CAPACITY
    def storage_capacity_rule(m, l, t):
        if l in m.s_P or l in m.s_S:
            return m.v_S[l, t] <= m.p_storage[l]
        else:
            return Constraint.Skip

    model.storage_capacity_constraint = Constraint(
        model.s_L, model.s_T, rule=storage_capacity_rule
    )

    # FINAL STORAGE CONDITION
    def final_storage_rule(m, l, t):
        if t == m.s_T.last():
            if l in m.s_P or l in m.s_S:
                return m.v_S[l, t] == m.p_init_storage[l]
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip

    model.final_storage_constraint = Constraint(
        model.s_L, model.s_T, rule=final_storage_rule
    )

    return model


if __name__ == "__main__":

    set_list = [
        "Customers",
        "TimePeriods",
        "ProductionPads",
        "CompletionsPads",
        "FreshwaterSources",
        "TransportMode",
        "SWDSites",
        "StorageSites",
        "NetworkNodes",
    ]

    parameter_list = [
        "Topology",
        "CompletionsDemand",
        "CustomersDemand",
        "DisposalCost",
        "FreshWaterAvailability",
        "FreshWaterCost",
        "StorageCapacity",
        "StorageInit",
        "TransportCapacity",
        "TransportCost",
        "TransportDistances",
        "WaterProfiles",
    ]

    with resources.path("pareto.case_studies", "blending_problem.xlsx") as fpath:
        [df_sets, df_parameters] = get_data(fpath, set_list, parameter_list)

    model = create_model(df_sets, df_parameters)
