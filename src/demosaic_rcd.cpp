// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Real RCD (Ratio Corrected Demosaicing) implementation arrives in Task 9.
// Until then the rcd_demosaic() entry point is a bilinear stub that lives in
// demosaic_dispatch.cpp. This file is intentionally empty so CMake can compile
// the translation unit now and the real port can drop in here without touching
// the build system.
