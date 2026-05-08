// third_party/glib_shim/include/glib.h
// GLib2 compatibility shim — provides all GLib2 types and functions needed by Lensfun
// All implementations are in glib_shim.cpp
#ifndef __GLIB_SHIM_H__
#define __GLIB_SHIM_H__

#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdarg>
#include <cassert>

#ifdef _WIN32
#include <direct.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#endif

// ============================================================================
// Basic types (matching GLib2 ABI)
// ============================================================================
typedef char             gchar;
typedef unsigned char    guchar;
typedef signed char      gint8;
typedef unsigned char    guint8;
typedef short            gint16;
typedef unsigned short   guint16;
typedef int              gint;
typedef unsigned int     guint;
typedef int              gint32;
typedef unsigned int     guint32;
typedef long long        gint64;
typedef unsigned long long guint64;
typedef int              gboolean;
typedef void*            gpointer;
typedef const void*      gconstpointer;
typedef size_t           gsize;
typedef long             gssize;
typedef unsigned int     gunichar;
typedef unsigned int     GQuark;
typedef int              (*GCompareFunc)(const void*, const void*);
typedef int              (*GCompareDataFunc)(const void*, const void*, void*);

#define TRUE  1
#define FALSE 0

// ============================================================================
// Memory management
// ============================================================================
#define g_new(type, count) ((type*)g_malloc(sizeof(type) * (count)))
#define g_new0(type, count) ((type*)calloc(count, sizeof(type)))

#ifdef __cplusplus
extern "C" {
#endif

void*     g_malloc(size_t size);
void*     g_realloc(void* ptr, size_t size);
void      g_free(void* ptr);
gchar*    g_strdup(const gchar* str);

// ============================================================================
// String / UTF-8 (ASCII-only, sufficient for camera/lens names)
// ============================================================================
gunichar  g_utf8_get_char(const gchar* str);
const gchar* g_utf8_next_char(const gchar* str);
gchar*    g_utf8_casefold(const gchar* str, gssize len);
gboolean  g_unichar_isspace(gunichar c);
gunichar  g_unichar_tolower(gunichar c);

// ============================================================================
// GError
// ============================================================================
typedef struct _GError {
    GQuark domain;
    gint   code;
    gchar* message;
} GError;

void g_set_error(GError** error, GQuark domain, gint code, const gchar* format, ...);
void g_error_free(GError* error);
#define g_clear_error(errp) do { if ((errp) && *(errp)) { g_error_free(*(errp)); *(errp) = NULL; } } while(0)

// ============================================================================
// File/Directory I/O
// ============================================================================
gchar*    g_build_filename(const gchar* first, ...);
const gchar* g_get_user_data_dir(void);

typedef struct _GDir GDir;
GDir*     g_dir_open(const gchar* path, guint flags, void** error);
const gchar* g_dir_read_name(GDir* dir);
void      g_dir_close(GDir* dir);

#define G_FILE_TEST_IS_DIR 1
gboolean  g_file_test(const gchar* filename, int test);
gboolean  g_file_get_contents(const gchar* filename, gchar** contents, gsize* length, GError** error);
int       g_open(const gchar* filename, int flags, int mode);

// ============================================================================
// XML Parsing — GMarkupParser (bridged to pugixml DOM internally)
// ============================================================================
typedef struct _GMarkupParseContext GMarkupParseContext;

typedef struct _GMarkupParser {
    void (*start_element) (GMarkupParseContext* ctx, const gchar* name,
                           const gchar** attr_names, const gchar** attr_values,
                           gpointer user_data, GError** error);
    void (*end_element)   (GMarkupParseContext* ctx, const gchar* name,
                           gpointer user_data, GError** error);
    void (*text)          (GMarkupParseContext* ctx, const gchar* text, gsize len,
                           gpointer user_data, GError** error);
    void (*passthrough)   (GMarkupParseContext* ctx, const gchar* text, gsize len,
                           gpointer user_data, GError** error);
    void (*error_cb)      (GMarkupParseContext* ctx, GError* error,
                           gpointer user_data);
} GMarkupParser;

typedef int GMarkupParseFlags;

GMarkupParseContext* g_markup_parse_context_new(
    const GMarkupParser* parser, GMarkupParseFlags flags,
    gpointer user_data, void (*destroy_notify)(gpointer));
gboolean g_markup_parse_context_parse(
    GMarkupParseContext* context, const gchar* text, gsize len, GError** error);
void g_markup_parse_context_free(GMarkupParseContext* context);
void g_markup_parse_context_get_position(
    GMarkupParseContext* context, gint* line, gint* col);
const gchar* g_markup_parse_context_get_element(GMarkupParseContext* context);

// Error domain for markup parsing
GQuark g_markup_error_quark(void);
#define G_MARKUP_ERROR              g_markup_error_quark()
#define G_MARKUP_ERROR_UNKNOWN_ATTRIBUTE 1
#define G_MARKUP_ERROR_INVALID_CONTENT   2

// ============================================================================
// GString (string builder)
// ============================================================================
typedef struct _GString {
    gchar* str;
    gsize  len;
    gsize  allocated_len;
} GString;

GString* g_string_sized_new(gsize default_size);
GString* g_string_append(GString* str, const gchar* val);
gchar*   g_string_free(GString* str, gboolean free_segment);

// ============================================================================
// GPtrArray (pointer array)
// ============================================================================
typedef struct _GPtrArray {
    gpointer* pdata;
    guint     len;
} GPtrArray;

GPtrArray* g_ptr_array_new(void);
void       g_ptr_array_free(GPtrArray* array, gboolean free_seg);
void       g_ptr_array_set_size(GPtrArray* array, gint length);
#define    g_ptr_array_index(array, index) ((array)->pdata[index])

// ============================================================================
// GPatternSpec (glob matching)
// ============================================================================
typedef struct _GPatternSpec GPatternSpec;
GPatternSpec* g_pattern_spec_new(const gchar* pattern);
gboolean      g_pattern_match(GPatternSpec* spec, gsize string_length,
                              const gchar* str, const gchar* str_reversed);
void          g_pattern_spec_free(GPatternSpec* spec);

// ============================================================================
// Threading (wraps std::mutex)
// GMutex is a complete type with an opaque buffer for std::mutex placement.
// ============================================================================
typedef struct _GMutex {
    char _opaque[64];
} GMutex;
void g_mutex_lock(GMutex* mutex);
void g_mutex_unlock(GMutex* mutex);

typedef struct _GStaticMutex {
    gpointer mutex_ptr;
    char     buffer[64];
} GStaticMutex;

#define G_STATIC_MUTEX_INIT { NULL, {0} }
void g_static_mutex_lock(GStaticMutex* mutex);
void g_static_mutex_unlock(GStaticMutex* mutex);

// ============================================================================
// Logging / Assertions
// ============================================================================
#define g_warning(fmt, ...) fprintf(stderr, "WARNING: " fmt "\n", ##__VA_ARGS__)
#define g_message(fmt, ...) fprintf(stderr, "MESSAGE: " fmt "\n", ##__VA_ARGS__)
#define g_critical(fmt, ...) fprintf(stderr, "CRITICAL: " fmt "\n", ##__VA_ARGS__)
#define g_debug(fmt, ...) fprintf(stderr, "DEBUG: " fmt "\n", ##__VA_ARGS__)

#ifndef g_assert
#define g_assert(expr) assert(expr)
#endif

#define g_return_if_fail(expr) do { if (!(expr)) return; } while(0)
#define g_return_val_if_fail(expr, val) do { if (!(expr)) return (val); } while(0)

// ============================================================================
// Markup output (used in Save functions — must compile)
// In real GLib2, g_markup_vprintf_escaped returns gchar* (allocated string)
// ============================================================================
gchar*    g_markup_vprintf_escaped(const gchar* format, va_list args);
void      g_markup_printf_escaped(GString* str, const gchar* format, ...);

// ============================================================================
// Misc macros and functions
// ============================================================================
#define G_GINT64_FORMAT "lld"
#define G_GUINT64_FORMAT "llu"
#define G_STRLOC __FILE__ ":" __LINE__
#define GLIB_CHECK_VERSION(major, minor, micro) (1)

#define G_FILE_ERROR_ACCES 1

#define g_ascii_strcasecmp(a, b) strcasecmp(a, b)
gchar* g_strstrip_impl(gchar* str);
#define g_strstrip(str) g_strstrip_impl(str)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __GLIB_SHIM_H__
