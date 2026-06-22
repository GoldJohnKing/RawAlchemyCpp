// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Real 3-pass Markesteijn (X-Trans) implementation arrives in Task 13.
// Until then the markesteijn_demosaic() entry point is a bilinear stub that
// lives in demosaic_dispatch.cpp. This file is intentionally empty so CMake
// can compile the translation unit now and the real port can drop in here
// without touching the build system.
