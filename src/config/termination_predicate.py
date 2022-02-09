import pyprover

holding_passenger, destination, loc = pyprover.terms("holding_passenger destination, loc")
holding_passenger_1 = r"p = 1 -> holding_passenger(p)"
destination_1_5 = r"p = 1 -> destination(p) = 5"
loc_1_5 = r"p = 1 -> loc(p) = 5"

print(pyprover.proves(pyprover.expr(r"p = 1"), pyprover.expr(holding_passenger_1 + r' /\ ' + destination_1_5 + r' /\ ' + loc_1_5)))
