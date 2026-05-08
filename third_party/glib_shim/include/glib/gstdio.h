// third_party/glib_shim/include/glib/gstdio.h
// Redirects to our main glib.h which already provides g_open() and other stdio functions.
// Lensfun's database.cpp includes this header.
#ifndef __GLIB_GSTDIO_H__
#define __GLIB_GSTDIO_H__
#include <glib.h>
#endif
