// GLib2 compatibility shim — C++17 implementations
// Replaces all GLib2 functions used by Lensfun with standard C++ equivalents.
#ifndef _WIN32
#define _GNU_SOURCE  // for vasprintf on glibc
#endif

#include <glib.h>
#include <pugixml.hpp>

#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <new>
#include <filesystem>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// ============================================================================
// Memory management
// ============================================================================
void* g_malloc(size_t size) {
    void* p = malloc(size ? size : 1);
    if (!p) abort();
    return p;
}

void* g_realloc(void* ptr, size_t size) {
    if (!size) { free(ptr); return nullptr; }
    void* p = realloc(ptr, size);
    if (!p) abort();
    return p;
}

void g_free(void* ptr) {
    free(ptr);
}

gchar* g_strdup(const gchar* str) {
    if (!str) return nullptr;
    return strdup(str);
}

// ============================================================================
// String / UTF-8 (ASCII-only)
// ============================================================================
gunichar g_utf8_get_char(const gchar* str) {
    if (!str || !*str) return 0;
    return (gunichar)(unsigned char)str[0];
}

const gchar* g_utf8_next_char(const gchar* str) {
    return str + 1;  // ASCII: 1 byte per char
}

gchar* g_utf8_casefold(const gchar* str, gssize len) {
    if (!str) return nullptr;
    gsize actual_len = (len < 0) ? strlen(str) : (gsize)len;
    gchar* result = (gchar*)g_malloc(actual_len + 1);
    for (gsize i = 0; i < actual_len; i++)
        result[i] = (gchar)tolower((unsigned char)str[i]);
    result[actual_len] = '\0';
    return result;
}

gboolean g_unichar_isspace(gunichar c) {
    return (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') ? TRUE : FALSE;
}

gunichar g_unichar_tolower(gunichar c) {
    return (gunichar)tolower((int)c);
}

// ============================================================================
// File/Directory I/O
// ============================================================================

// g_build_filename — variadic, join paths with '/'
gchar* g_build_filename(const gchar* first, ...) {
    if (!first) return g_strdup("");
    std::string result(first);
    va_list args;
    va_start(args, first);
    const gchar* part;
    while ((part = va_arg(args, const gchar*)) != nullptr) {
        if (!result.empty() && result.back() != '/' && result.back() != '\\')
            result += '/';
        // Skip leading slash in part if result already has one
        while (*part == '/' || *part == '\\') part++;
        if (*part) {
            result += part;
        }
    }
    va_end(args);
    return g_strdup(result.c_str());
}

// g_get_user_data_dir — returns user data directory
const gchar* g_get_user_data_dir(void) {
    static std::string cached;
    if (cached.empty()) {
#ifdef _WIN32
        const char* appdata = getenv("APPDATA");
        cached = appdata ? appdata : "C:/Users";
#else
        const char* home = getenv("HOME");
        if (home) cached = std::string(home) + "/.local/share";
        else cached = "/tmp";
#endif
    }
    return cached.c_str();
}

// GDir — wraps std::filesystem::directory_iterator
struct _GDir {
    std::filesystem::directory_iterator it;
    std::filesystem::directory_iterator end;
};

GDir* g_dir_open(const gchar* path, guint /*flags*/, void** /*error*/) {
    if (!path) return nullptr;
    std::error_code ec;
    auto dir_iter = std::filesystem::directory_iterator(path, ec);
    if (ec) return nullptr;
    GDir* gdir = (GDir*)g_malloc(sizeof(GDir));
    new (gdir) _GDir();  // placement new for non-trivial members
    gdir->it = std::move(dir_iter);
    gdir->end = std::filesystem::directory_iterator();
    return gdir;
}

const gchar* g_dir_read_name(GDir* dir) {
    if (!dir) return nullptr;
    // Use a static string to hold the filename — valid until next call.
    // Lensfun only uses the return value immediately (to copy into std::string).
    static std::string current_name;
    while (dir->it != dir->end) {
        std::string name = dir->it->path().filename().string();
        dir->it++;  // advance for next call
        if (name == "." || name == "..") continue;
        current_name = std::move(name);
        return current_name.c_str();
    }
    return nullptr;
}

void g_dir_close(GDir* dir) {
    if (dir) {
        dir->it.~directory_iterator();
        dir->end.~directory_iterator();
        g_free(dir);
    }
}

gboolean g_file_test(const gchar* filename, int test) {
    if (!filename) return FALSE;
    if (test == G_FILE_TEST_IS_DIR) {
        std::error_code ec;
        return std::filesystem::is_directory(filename, ec) ? TRUE : FALSE;
    }
    return FALSE;
}

gboolean g_file_get_contents(const gchar* filename, gchar** contents, gsize* length, GError** error) {
    if (!filename || !contents) return FALSE;
    FILE* f = fopen(filename, "rb");
    if (!f) {
        // Lensfun expects error to be set on failure (dereferences err->code)
        if (error) {
            g_set_error(error, 0, G_FILE_ERROR_ACCES,
                       "Failed to open file: %s", filename);
        }
        return FALSE;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        if (error) g_set_error(error, 0, 1, "Failed to get file size: %s", filename);
        fclose(f);
        return FALSE;
    }
    *contents = (gchar*)g_malloc((size_t)sz + 1);
    (*contents)[sz] = '\0';
    if ((long)fread(*contents, 1, (size_t)sz, f) != sz) {
        g_free(*contents);
        *contents = nullptr;
        if (error) g_set_error(error, 0, 1, "Failed to read file: %s", filename);
        fclose(f);
        return FALSE;
    }
    if (length) *length = (gsize)sz;
    fclose(f);
    return TRUE;
}

int g_open(const gchar* filename, int flags, int mode) {
#ifdef _WIN32
    return _open(filename, flags, mode);
#else
    return open(filename, flags, mode);
#endif
}

// ============================================================================
// GError
// ============================================================================
GQuark g_markup_error_quark(void) {
    static GQuark q = 0;
    if (!q) q = 12345;  // Arbitrary non-zero value
    return q;
}

void g_set_error(GError** error, GQuark domain, gint code, const gchar* format, ...) {
    if (!error || *error) return;  // Don't overwrite existing error
    GError* err = (GError*)g_malloc(sizeof(GError));
    err->domain = domain;
    err->code = code;
    va_list args;
    va_start(args, format);
#ifdef _WIN32
    // MSVC doesn't have vasprintf; use _vscprintf + malloc + vsprintf_s
    int len = _vscprintf(format, args);
    if (len < 0) {
        err->message = g_strdup("unknown error");
    } else {
        err->message = (gchar*)g_malloc((size_t)len + 1);
        va_list args2;
        va_start(args2, format);
        vsprintf_s(err->message, (size_t)len + 1, format, args2);
        va_end(args2);
    }
#else
    if (vasprintf(&err->message, format, args) == -1)
        err->message = g_strdup("unknown error");
#endif
    va_end(args);
    *error = err;
}

void g_error_free(GError* error) {
    if (!error) return;
    free(error->message);
    g_free(error);
}

// ============================================================================
// GMarkupParser Bridge (DOM -> SAX using pugixml)
// ============================================================================
struct _GMarkupParseContext {
    GMarkupParser parser;
    GMarkupParseFlags flags;
    gpointer user_data;
    void (*destroy_notify)(gpointer);
    pugi::xml_document doc;
    std::string current_element;
    unsigned int current_line;
    unsigned int current_col;
};

GMarkupParseContext* g_markup_parse_context_new(
    const GMarkupParser* parser, GMarkupParseFlags flags,
    gpointer user_data, void (*destroy_notify)(gpointer))
{
    GMarkupParseContext* ctx = (GMarkupParseContext*)g_malloc(sizeof(GMarkupParseContext));
    new (ctx) GMarkupParseContext();  // placement new for std::string/doc
    ctx->parser = *parser;
    ctx->flags = flags;
    ctx->user_data = user_data;
    ctx->destroy_notify = destroy_notify;
    ctx->current_line = 1;
    ctx->current_col = 0;
    return ctx;
}

// Forward declaration
static void walk_dom(GMarkupParseContext* ctx, pugi::xml_node node, GError** error);

gboolean g_markup_parse_context_parse(
    GMarkupParseContext* context, const gchar* text, gsize len, GError** error)
{
    if (!context || !text) return FALSE;

    gsize actual_len = (len == (gsize)-1) ? strlen(text) : len;
    pugi::xml_parse_result result = context->doc.load_buffer(
        text, actual_len, pugi::parse_default | pugi::parse_pi);

    if (!result) {
        if (error) {
            g_set_error(error, g_markup_error_quark(), G_MARKUP_ERROR_INVALID_CONTENT,
                       "XML parse error: %s at offset %d",
                       result.description(), (int)result.offset);
        }
        return FALSE;
    }

    // Walk the document children
    for (pugi::xml_node child = context->doc.first_child(); child; child = child.next_sibling()) {
        if (child.type() == pugi::node_element) {
            walk_dom(context, child, error);
            if (error && *error) return FALSE;
        }
    }
    return TRUE;
}

static void walk_dom(GMarkupParseContext* ctx, pugi::xml_node node, GError** error) {
    if (node.type() != pugi::node_element) return;

    // Collect attributes into arrays (null-terminated)
    const int MAX_ATTRS = 64;
    const gchar* attr_names[MAX_ATTRS + 1];
    const gchar* attr_values[MAX_ATTRS + 1];
    int attr_count = 0;
    for (pugi::xml_attribute attr = node.first_attribute(); attr && attr_count < MAX_ATTRS;
         attr = attr.next_attribute()) {
        attr_names[attr_count] = attr.name();
        attr_values[attr_count] = attr.value();
        attr_count++;
    }
    attr_names[attr_count] = nullptr;
    attr_values[attr_count] = nullptr;

    // Track current element for get_element()
    ctx->current_element = node.name();

    // Call start_element callback
    if (ctx->parser.start_element) {
        ctx->parser.start_element(ctx, node.name(), attr_names, attr_values,
                                  ctx->user_data, error);
        if (error && *error) {
            if (ctx->parser.error_cb)
                ctx->parser.error_cb(ctx, *error, ctx->user_data);
            return;
        }
    }

    // Recurse into children (text nodes + sub-elements)
    for (pugi::xml_node child = node.first_child(); child; child = child.next_sibling()) {
        if (child.type() == pugi::node_pcdata || child.type() == pugi::node_cdata) {
            const char* text = child.value();
            if (text && *text) {
                if (ctx->parser.text) {
                    ctx->parser.text(ctx, text, strlen(text), ctx->user_data, error);
                    if (error && *error) {
                        if (ctx->parser.error_cb)
                            ctx->parser.error_cb(ctx, *error, ctx->user_data);
                        return;
                    }
                }
            }
        } else if (child.type() == pugi::node_element) {
            walk_dom(ctx, child, error);
            if (error && *error) return;
        }
    }

    // Call end_element callback
    if (ctx->parser.end_element) {
        ctx->parser.end_element(ctx, node.name(), ctx->user_data, error);
        if (error && *error) {
            if (ctx->parser.error_cb)
                ctx->parser.error_cb(ctx, *error, ctx->user_data);
            return;
        }
    }
}

void g_markup_parse_context_free(GMarkupParseContext* context) {
    if (context) {
        if (context->destroy_notify && context->user_data)
            context->destroy_notify(context->user_data);
        // Manually destroy std::string and xml_document
        context->current_element.~basic_string();
        context->doc.~xml_document();
        g_free(context);
    }
}

void g_markup_parse_context_get_position(GMarkupParseContext* ctx, gint* line, gint* col) {
    if (line) *line = ctx ? (gint)ctx->current_line : 0;
    if (col)  *col  = ctx ? (gint)ctx->current_col : 0;
}

const gchar* g_markup_parse_context_get_element(GMarkupParseContext* ctx) {
    return ctx ? ctx->current_element.c_str() : nullptr;
}

// ============================================================================
// GString (string builder)
// ============================================================================
GString* g_string_sized_new(gsize default_size) {
    GString* gs = (GString*)g_malloc(sizeof(GString));
    gs->allocated_len = default_size > 0 ? default_size + 1 : 16;
    gs->str = (gchar*)g_malloc(gs->allocated_len);
    gs->str[0] = '\0';
    gs->len = 0;
    return gs;
}

GString* g_string_append(GString* str, const gchar* val) {
    if (!str || !val) return str;
    gsize add_len = strlen(val);
    if (str->len + add_len + 1 > str->allocated_len) {
        str->allocated_len = (str->len + add_len) * 2 + 1;
        str->str = (gchar*)g_realloc(str->str, str->allocated_len);
    }
    memcpy(str->str + str->len, val, add_len + 1);
    str->len += add_len;
    return str;
}

gchar* g_string_free(GString* str, gboolean free_segment) {
    if (!str) return nullptr;
    gchar* result = free_segment ? nullptr : str->str;
    if (free_segment) g_free(str->str);
    g_free(str);
    return result;
}

// ============================================================================
// GPtrArray (pointer array)
// ============================================================================
GPtrArray* g_ptr_array_new(void) {
    GPtrArray* arr = (GPtrArray*)g_malloc(sizeof(GPtrArray));
    arr->pdata = nullptr;
    arr->len = 0;
    return arr;
}

void g_ptr_array_free(GPtrArray* array, gboolean free_seg) {
    if (!array) return;
    if (free_seg) {
        for (guint i = 0; i < array->len; i++)
            g_free(array->pdata[i]);
    }
    g_free(array->pdata);
    g_free(array);
}

void g_ptr_array_set_size(GPtrArray* array, gint length) {
    if (!array || length < 0) return;
    guint new_len = (guint)length;
    if (new_len > array->len) {
        array->pdata = (gpointer*)g_realloc(array->pdata, new_len * sizeof(gpointer));
        memset(array->pdata + array->len, 0, (new_len - array->len) * sizeof(gpointer));
    }
    array->len = new_len;
}

// ============================================================================
// GPatternSpec (simple wildcard matching)
// ============================================================================
struct _GPatternSpec {
    std::string pattern;
};

GPatternSpec* g_pattern_spec_new(const gchar* pattern) {
    GPatternSpec* spec = new _GPatternSpec();
    spec->pattern = pattern ? pattern : "";
    return spec;
}

gboolean g_pattern_match(GPatternSpec* spec, gsize /*string_length*/,
                         const gchar* str, const gchar* /*str_reversed*/) {
    if (!spec || !str) return FALSE;
    // Simple wildcard: * matches anything, ? matches one char
    const char* p = spec->pattern.c_str();
    const char* s = str;
    const char* star_p = nullptr;
    const char* star_s = nullptr;
    while (*s) {
        if (*p == '*') { star_p = p++; star_s = s; continue; }
        if (*p == '?' || *p == *s) { p++; s++; continue; }
        if (star_p) { p = star_p + 1; s = ++star_s; continue; }
        return FALSE;
    }
    while (*p == '*') p++;
    return *p == '\0' ? TRUE : FALSE;
}

void g_pattern_spec_free(GPatternSpec* spec) {
    delete spec;
}

// ============================================================================
// Threading (std::mutex wrapper)
// GMutex has a 64-byte opaque buffer; we placement-new std::mutex into it.
// For static GMutex (zero-initialized), we lazily construct on first use.
// ============================================================================

// Helper to get or create the std::mutex inside a GMutex
static std::mutex* get_mutex(GMutex* mutex) {
    if (!mutex) return nullptr;
    auto* m = reinterpret_cast<std::mutex*>(mutex->_opaque);
    // Check if already constructed (very simple heuristic: if all zeros)
    // We use a global map to track which GMutex objects have been initialized
    static std::unordered_map<GMutex*, bool> init_map;
    static std::mutex init_map_mutex;
    std::lock_guard<std::mutex> guard(init_map_mutex);
    if (init_map.find(mutex) == init_map.end()) {
        new (m) std::mutex();
        init_map[mutex] = true;
    }
    return m;
}

void g_mutex_lock(GMutex* mutex) {
    auto* m = get_mutex(mutex);
    if (m) m->lock();
}

void g_mutex_unlock(GMutex* mutex) {
    if (!mutex) return;
    auto* m = reinterpret_cast<std::mutex*>(mutex->_opaque);
    m->unlock();
}

void g_static_mutex_lock(GStaticMutex* mutex) {
    if (!mutex) return;
    if (!mutex->mutex_ptr) {
        mutex->mutex_ptr = (gpointer)new unsigned char[sizeof(std::mutex)];
        new (mutex->mutex_ptr) std::mutex();
    }
    ((std::mutex*)mutex->mutex_ptr)->lock();
}

void g_static_mutex_unlock(GStaticMutex* mutex) {
    if (mutex && mutex->mutex_ptr)
        ((std::mutex*)mutex->mutex_ptr)->unlock();
}

// ============================================================================
// Markup output (XML escaping for Save functions)
// ============================================================================
static void xml_escape_append(std::string& out, const gchar* text) {
    for (; *text; text++) {
        switch (*text) {
            case '&':  out += "&amp;"; break;
            case '<':  out += "&lt;"; break;
            case '>':  out += "&gt;"; break;
            case '"':  out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default:   out += *text; break;
        }
    }
}

gchar* g_markup_vprintf_escaped(const gchar* format, va_list args) {
    if (!format) return g_strdup("");
    gchar buf[4096];
    vsnprintf(buf, sizeof(buf), format, args);
    std::string escaped;
    xml_escape_append(escaped, buf);
    return g_strdup(escaped.c_str());
}

void g_markup_printf_escaped(GString* str, const gchar* format, ...) {
    if (!str || !format) return;
    va_list args;
    va_start(args, format);
    gchar* escaped = g_markup_vprintf_escaped(format, args);
    va_end(args);
    if (escaped) {
        g_string_append(str, escaped);
        g_free(escaped);
    }
}

// ============================================================================
// Misc
// ============================================================================
gchar* g_strstrip_impl(gchar* str) {
    if (!str) return nullptr;
    gchar* start = str;
    while (*start && (*start == ' ' || *start == '\t' || *start == '\n' || *start == '\r'))
        start++;
    if (*start == '\0') { *str = '\0'; return str; }
    gchar* end = start + strlen(start) - 1;
    while (end > start && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r'))
        *end-- = '\0';
    if (start != str) memmove(str, start, strlen(start) + 1);
    return str;
}
