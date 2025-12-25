import pulp


def optimize_battery_milp_1mwh(prices):
    """
    Optimize the operation of a 1 MW / 1 MWh battery for profit maximization.

    Args:
        prices (list): Hourly electricity prices for a single day (24 values).

    Returns:
        dict: Optimal profit, charge/discharge schedules, and SOC profile.
    """
    hours = list(range(1, 25))  # 24 hours

    # Define the MILP problem
    problem = pulp.LpProblem(
        "Battery_Optimization_1MWh",
        pulp.LpMaximize
    )

    # Decision variables
    P_charge = pulp.LpVariable.dicts("P_charge", hours, cat="Binary")
    P_discharge = pulp.LpVariable.dicts("P_discharge", hours, cat="Binary")
    E = pulp.LpVariable.dicts("E", hours, lowBound=0, upBound=1, cat="Continuous")

    # Objective function
    problem += pulp.lpSum(
        prices[t - 1] * (P_discharge[t] - P_charge[t])
        for t in hours
    )

    cumulative_charge = []
    cumulative_discharge = []

    for t in hours:
        # SOC dynamics
        if t == 1:
            problem += E[t] == P_charge[t] - P_discharge[t]
        else:
            problem += E[t] == E[t - 1] + P_charge[t] - P_discharge[t]

        # SOC bounds
        problem += E[t] >= 0
        problem += E[t] <= 1

        # No simultaneous charge & discharge
        problem += P_charge[t] + P_discharge[t] <= 1

        # Discharge only if energy available
        if t > 1:
            problem += P_discharge[t] <= E[t - 1]

        cumulative_charge.append(P_charge[t])
        cumulative_discharge.append(P_discharge[t])

        # Capacity constraints
        problem += (
            pulp.lpSum(cumulative_charge) -
            pulp.lpSum(cumulative_discharge)
        ) <= 1

        # No multiple discharges without recharge
        problem += (
            pulp.lpSum(cumulative_discharge) <=
            pulp.lpSum(cumulative_charge)
        )

    # Initial and final SOC
    problem += E[1] == 0
    problem += E[24] == 0

    # Solve
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    return {
        "Profit": pulp.value(problem.objective),
        "Charge Schedule": [P_charge[t].varValue for t in hours],
        "Discharge Schedule": [P_discharge[t].varValue for t in hours],
        "SOC Schedule": [E[t].varValue for t in hours],
    }


def optimize_battery_milp_2mwh_blocking(prices):
    """
    Optimize the operation of a 1 MW / 2 MWh battery with full & half operations
    and blocking constraints.

    Args:
        prices (list): Hourly electricity prices for a single day (24 values).

    Returns:
        dict: Optimal profit, schedules, and SOC profile.
    """
    hours = list(range(1, 25))  # 24 hours

    # Define the MILP problem
    problem = pulp.LpProblem(
        "Battery_Optimization_2MWh_Blocking",
        pulp.LpMaximize
    )

    # Decision variables
    P_charge_full = pulp.LpVariable.dicts("P_charge_full", hours, cat="Binary")
    P_charge_half = pulp.LpVariable.dicts("P_charge_half", hours, cat="Binary")
    P_discharge_full = pulp.LpVariable.dicts("P_discharge_full", hours, cat="Binary")
    P_discharge_half = pulp.LpVariable.dicts("P_discharge_half", hours, cat="Binary")

    # State of charge
    E = pulp.LpVariable.dicts("E", hours, lowBound=0, upBound=2, cat="Continuous")

    # Objective function
    problem += pulp.lpSum(
        prices[t - 1] * (
            2 * P_discharge_full[t] +
            P_discharge_half[t] -
            2 * P_charge_full[t] -
            P_charge_half[t]
        )
        for t in hours
    )

    for t in hours:
        # SOC dynamics
        if t == 1:
            problem += (
                E[t] ==
                2 * P_charge_full[t] + P_charge_half[t] -
                2 * P_discharge_full[t] - P_discharge_half[t]
            )
        else:
            problem += (
                E[t] ==
                E[t - 1] +
                2 * P_charge_full[t] + P_charge_half[t] -
                2 * P_discharge_full[t] - P_discharge_half[t]
            )

        # SOC bounds
        problem += E[t] >= 0
        problem += E[t] <= 2

        # Only one operation per hour
        problem += (
            P_charge_full[t] +
            P_charge_half[t] +
            P_discharge_full[t] +
            P_discharge_half[t]
        ) <= 1

        # Blocking after full operation
        if t < 24:
            z_t = P_charge_full[t] + P_discharge_full[t]
            problem += (
                P_charge_full[t + 1] +
                P_charge_half[t + 1] +
                P_discharge_full[t + 1] +
                P_discharge_half[t + 1]
            ) <= 1 - z_t

        # Discharge limits
        if t > 1:
            problem += P_discharge_full[t] <= E[t - 1] * 0.5
            problem += P_discharge_half[t] <= E[t - 1]

        # Charge limits
        if t > 1:
            problem += P_charge_full[t] <= (2 - E[t - 1]) * 0.5
            problem += P_charge_half[t] <= (2 - E[t - 1])

    # Initial and final SOC
    problem += E[1] == 0
    problem += E[24] == 0

    # Solve
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    return {
        "Profit": pulp.value(problem.objective)
        if problem.status == pulp.LpStatusOptimal else None,
        "Charge Full Schedule": [P_charge_full[t].varValue for t in hours],
        "Charge Half Schedule": [P_charge_half[t].varValue for t in hours],
        "Discharge Full Schedule": [P_discharge_full[t].varValue for t in hours],
        "Discharge Half Schedule": [P_discharge_half[t].varValue for t in hours],
        "SOC Schedule": [E[t].varValue for t in hours],
    }