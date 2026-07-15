# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np

from pandapower.auxiliary import I_from_SV_elementwise, phase_shift_unit_operator, read_from_net
from pandapower.control.controller.trafo.DiscreteTapControl import DiscreteTapControl

_SQRT3 = np.sqrt(3.0)


class LineDropCompensationTapControl(DiscreteTapControl):
    """
    DiscreteTapControl variant that regulates a line-drop-compensated voltage
    instead of the trafo's own terminal voltage: a real feeder voltage regulator
    with line-drop compensation (LDC) estimates the voltage at a downstream
    point by subtracting the drop across a compensator impedance (r_ohm, x_ohm)
    from its own terminal voltage, using the current flowing out of that
    terminal. This lets the controller hold a remote point in band without
    actually modelling a bus there.

    Parameters:
        net (ADict): Pandapower struct
        element_index (int): ID of the trafo that is controlled
        vm_lower_pu (float): Lower voltage limit in pu, applied to the compensated voltage
        vm_upper_pu (float): Upper voltage limit in pu, applied to the compensated voltage
        r_ohm (float, 0.0): Compensator resistance in Ohm, line-to-neutral, referred to the
            controlled side's own voltage level (i.e. the actual line impedance to the point
            being emulated, not a relay/PT/CT-secondary-referred value). A single scalar,
            applied uniformly even if element_index controls more than one trafo.
        x_ohm (float, 0.0): Compensator reactance in Ohm, same base and same scalar-only caveat as r_ohm
        side (string, "lv"): Side of the transformer where the voltage/current is measured
            (hv or lv, or mv for a trafo3w)
        element (string, "trafo"): Trafo type ("trafo" or "trafo3w")
        tol (float, 0.001): Voltage tolerance band at bus in Percent (default: 1% = 0.01pu)
        in_service (bool, True): Indicates if the controller is currently in_service
        drop_same_existing_ctrl (bool, False): Indicates if already existing controllers of the same type and with the
            same matching parameters (e.g. at same element) should be dropped
    """

    def __init__(self, net, element_index, vm_lower_pu, vm_upper_pu, r_ohm=0.0, x_ohm=0.0,
                 side="lv", element="trafo", tol=1e-3, in_service=True, hunting_limit=None,
                 level=0, order=0, drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        super().__init__(net, element_index, vm_lower_pu, vm_upper_pu, side=side, element=element,
                         tol=tol, in_service=in_service, hunting_limit=hunting_limit, level=level,
                         order=order, drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)
        self.r_ohm = r_ohm
        self.x_ohm = x_ohm

    def _get_controlled_vm_pu(self, net):
        if self.r_ohm == 0.0 and self.x_ohm == 0.0:
            return super()._get_controlled_vm_pu(net)

        # Voltage comes from res_bus (like the parent class), not res_trafo/res_trafo3w:
        # an isolated-but-still-in_service trafo (e.g. cut off by an open switch during
        # the connectivity check) gets its res_bus.vm_pu forced to NaN, which res_trafo's
        # own vm_<side>_pu is not -- reading it from res_bus keeps that NaN propagating
        # into the compensated result instead of computing on stale/placeholder values.
        vm_pu = read_from_net(net, "res_bus", self.trafobus, "vm_pu", self._read_write_flag)
        va_degree = read_from_net(net, "res_bus", self.trafobus, "va_degree", self._read_write_flag)
        vn_kv = read_from_net(net, "bus", self.trafobus, "vn_kv", self._read_write_flag)
        res_element = "res_" + self.element
        p_mw = read_from_net(net, res_element, self.element_index, f"p_{self.side}_mw", self._read_write_flag)
        q_mvar = read_from_net(net, res_element, self.element_index, f"q_{self.side}_mvar", self._read_write_flag)

        # Everything below is complex arithmetic in physical kV/kA/MW/Mvar/Ohm, not pu --
        # the R/X compensation is only meaningful with actual phase angles, and pu would
        # need its own (current) base to stay consistent, which pandapower doesn't define.
        v_ll_complex_kv = vm_pu * vn_kv * phase_shift_unit_operator(va_degree)
        v_ln_complex_kv = v_ll_complex_kv / _SQRT3
        s_complex_mva = p_mw + 1j * q_mvar
        # Current flowing INTO the trafo from this bus (pandapower's p_/q_<side>_mw sign
        # convention) -- matches the direction OpenDSS's own LDC current uses, which is why
        # this is added, not subtracted, from the terminal voltage below.
        i_complex_ka = I_from_SV_elementwise(s_complex_mva, _SQRT3 * v_ll_complex_kv)
        drop_complex_kv = i_complex_ka * (self.r_ohm + 1j * self.x_ohm)
        v_ln_comp_complex_kv = v_ln_complex_kv + drop_complex_kv
        return np.abs(v_ln_comp_complex_kv) * _SQRT3 / vn_kv
