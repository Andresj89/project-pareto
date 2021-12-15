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
    value
)
from pareto.utilities.get_data import get_data
from importlib import resources
import pyomo.environ

# import gurobipy
from pyomo.common.config import ConfigBlock, ConfigValue, In
from enum import Enum

from pareto.utilities.solvers import get_solver, set_timeout

import pandas as pd


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

    # model.p_completionsdemand = Param(
    #     model.s_L,
    #     model.s_T,
    #     default=0,
    #     initialize=df_parameters["CompletionsDemand"],
    #     doc="Water demand for completion operations in location l and time period t",
    # )

    model.p_demand = Param(
        model.s_L,
        model.s_T,
        default=0,
        initialize=df_parameters["Demand"],
        doc="Water required for completion operations and customers in location l and time period t",
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

    model.p_prodcomp = Param(
        model.s_L,
        model.s_T,
        default=0,
        initialize=df_parameters["ProdComp"],
        doc="Compostion at production sites l and time period t",
    )

    model.p_storeageinitcomp = Param(
        model.s_L,
        default=0,
        initialize=df_parameters["StorageInitComp"],
        doc="Compostion at storage sites l and time period t",
    )

    # *******************************************************************************
    #                               VARIABLE DEFINITION
    # *******************************************************************************

    model.v_D = Var(
        model.s_L,
        model.s_T,
        initialize=0,
        within=NonNegativeReals,
        doc="Water demand at location l in time period t",
    )

    model.v_P = Var(
        model.s_L,
        model.s_T,
        initialize=0,
        within=NonNegativeReals,
        doc="Water production at location l in time period t",
    )

    model.v_Q = Var(
        model.s_L,
        model.s_G,
        model.s_L,
        model.s_T,
        initialize=0,
        within=NonNegativeReals,
        doc="Water transportation between locations l and l', via transport mode g in time period t",
    )

    model.v_S = Var(
        model.s_L,
        model.s_T,
        initialize=0,
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

    model.v_COMP = Var(
        model.s_L,
        model.s_T,
        initialize=0,
        within=NonNegativeReals,
        doc="Composition at location l and time period t",
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

    model.p_prodcomp

    model.p_storeageinitcomp

    # MATERIAL BALANCE PER COMPONENT
    def component_material_balance_rule(m, l, t):
        if t == model.s_T.first():
            return (
                m.v_P[l, t]*m.p_prodcomp[l,t]
                + sum(
                    sum(
                        m.v_Q[l_tilde, g, l, t]*m.v_COMP[l_tilde,t]
                        for l_tilde in m.s_L
                        if m.p_topology[l_tilde, g, l]
                    )
                    for g in m.s_G
                )
                + m.p_init_storage[l]*m.p_storeageinitcomp[l]
                == m.v_D[l, t]*m.v_COMP[l,t]
                + sum(
                    sum(
                        m.v_Q[l, g, l_tilde, t]*m.v_COMP[l,t]
                        for l_tilde in m.s_L
                        if m.p_topology[l, g, l_tilde]
                    )
                    for g in m.s_G
                )
                + m.v_S[l, t]*m.v_COMP[l,t]
            )

        else:
            return (
                m.v_P[l, t]*m.p_prodcomp[l,t]
                + sum(
                    sum(
                        m.v_Q[l_tilde, g, l, t]*m.v_COMP[l_tilde,t]
                        for l_tilde in m.s_L
                        if m.p_topology[l_tilde, g, l]
                    )
                    for g in m.s_G
                )
                + m.v_S[l, m.s_T.prev(t)]*m.v_COMP[l,m.s_T.prev(t)]
                == m.v_D[l, t]*m.v_COMP[l,t]
                + sum(
                    sum(
                        m.v_Q[l, g, l_tilde, t]*m.v_COMP[l,t]
                        for l_tilde in m.s_L
                        if m.p_topology[l, g, l_tilde]
                    )
                    for g in m.s_G
                )
                + m.v_S[l, t]*m.v_COMP[l,t]
            )

    model.component_material_balance = Constraint(
        model.s_L, model.s_T, rule=component_material_balance_rule
    )

    # FRESH WATER AVAILABILITY
    def fresh_water_rule(m, l, t):
        return m.v_P[l, t] <= m.p_freshwater[l, t]

    model.fresh_water_constraint = Constraint(
        model.s_F, model.s_T, rule=fresh_water_rule
    )

    # The following code fixes the variable v_P to he already known production profiles
    # for pads or to zero for locations that do not produce water
    # This helps in reducing the number of decision variables
    for i in model.v_P:
        if not i[0] in model.s_F:
            model.v_P[i].fix(model.p_waterprofiles[i])

    # The following code fixes the variable v_D to the already known water demand
    # for completion operations and water demand from customers. This helps in reducing the number of decision variables.
    # The variable is not fixed for disposal sites since the disposal is calcuated as "demand"
    for i in model.v_D:
        if not i[0] in model.s_K:
            model.v_D[i].fix(model.p_demand[i])

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
        return m.v_S[l, t] <= m.p_storage[l]

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
        "Demand",
        "DisposalCost",
        "FreshWaterAvailability",
        "FreshWaterCost",
        "StorageCapacity",
        "StorageInit",
        "TransportCapacity",
        "TransportCost",
        "TransportDistances",
        "WaterProfiles",
        "ProdComp",
        "StorageInitComp"
    ]

    with resources.path("pareto.case_studies", "small_blending_problem.xlsx") as fpath:
        [df_sets, df_parameters] = get_data(fpath, set_list, parameter_list)

    model = create_model(df_sets, df_parameters)

    solver = get_solver("gurobi_direct", "gurobi")
    set_timeout(solver, timeout_s=60*10)
    solver.options["mipgap"] = 0
    solver.options["NonConvex"] = 2

    # Deactivate component material balance and solve for flow rates
    model.component_material_balance.deactivate()
    results = solver.solve(model, tee=True)

    # Fix flow rates and activate component material balance to POSTCALCULATE compositions
    # model.v_P.fix()
    # model.v_Q.fix()
    # model.v_S.fix()
    # model.v_D.fix()

    # Or you can set a composition value
    for i in model.v_COMP:
        model.v_COMP[i].value = 50000

    solver = get_solver("ipopt")
    model.component_material_balance.activate()
    results = solver.solve(model, tee=True)

    # CREATE REPORT
    Q_results = []
    P_results = []
    D_results = []
    S_results = []
    COMP_results = []

    Q_results = [(l, g, l_tilde, t, value(v)) for (l, g, l_tilde, t), v in model.v_Q.items()]
    P_results = [(l, t, value(v)) for (l, t), v in model.v_P.items()]
    D_results = [(l, t, value(v)) for (l, t), v in model.v_D.items()]
    S_results = [(l, t, value(v)) for (l, t), v in model.v_S.items()]
    COMP_results = [(l, t, value(v)) for (l, t), v in model.v_COMP.items()]

    Q_df = pd.DataFrame(Q_results, columns=["origin","mode", "destination","time","variable value"])
    P_df = pd.DataFrame(P_results, columns=["origin", "time", "variable value"])
    D_df = pd.DataFrame(D_results, columns=["origin", "time", "variable value"])
    S_df = pd.DataFrame(S_results, columns=["origin", "time", "variable value"])
    COMP_df = pd.DataFrame(COMP_results, columns=["origin", "time", "variable value"])


    with pd.ExcelWriter("blendin_problem_results.xlsx") as writer:
        Q_df.to_excel(writer, sheet_name="Transfers")
        P_df.to_excel(writer, sheet_name="Production")
        D_df.to_excel(writer, sheet_name="Demand")
        S_df.to_excel(writer, sheet_name="Storage")
        COMP_df.to_excel(writer, sheet_name="Composition")

    print("end")

    
