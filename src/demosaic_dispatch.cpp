// SPDX-License-Identifier: AGPL-3.0-or-later
// Dispatch shims. Real RCD lives in demosaic_rcd.cpp (Task 9).
// Real Markesteijn lives in demosaic_markesteijn.cpp (Task 11).
//
// Both entry points are declared in their respective headers and defined in
// their respective translation units. This file remains as a registration
// seam in case future demosaic algorithms need a central dispatcher; it is
// intentionally empty of definitions today.

#include "demosaic_rcd.h"
#include "demosaic_markesteijn.h"
