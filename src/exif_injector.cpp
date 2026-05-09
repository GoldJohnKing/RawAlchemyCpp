/**
 * @file exif_injector.cpp
 * @brief EXIF metadata extraction from RAW files and injection into JPEG output.
 *
 * Uses LibRaw's exif_parser_callback to collect EXIF tags during open_file(),
 * then libexif to serialize them into an APP1 blob for JPEG embedding.
 * MakerNote is deliberately dropped to avoid offset corruption issues.
 *
 * See design spec: docs/superpowers/specs/2026-05-09-jpeg-exif-embedding-design.md
 */

#include "exif_injector.h"

#include <libraw/libraw.h>
#include <libexif/exif-data.h>
#include <libexif/exif-content.h>
#include <libexif/exif-entry.h>
#include <libexif/exif-byte-order.h>
#include <libexif/exif-format.h>
#include <libexif/exif-ifd.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace rawalchemy {

// ----------------------------------------------------------------
//  TIFF data type sizes (per EXIF 2.3 spec)
// ----------------------------------------------------------------
static size_t typeSize(int type) {
    static const size_t sizes[] = {0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8};
    return (type >= 1 && type <= 12) ? sizes[type] : 1;
}

// ----------------------------------------------------------------
//  ExifCollector — internal structure
// ----------------------------------------------------------------
struct ExifTagRecord {
    uint16_t tag;
    uint16_t type;
    uint32_t count;
    std::vector<uint8_t> data;
    ExifIfd ifd;
};

struct ExifCollector {
    std::vector<ExifTagRecord> tags;
    unsigned int byteOrder = 0;
    bool byteOrderSet = false;
};

// ----------------------------------------------------------------
//  Factory functions
// ----------------------------------------------------------------
ExifCollector* createExifCollector() {
    return new ExifCollector();
}

void destroyExifCollector(ExifCollector* collector) {
    delete collector;
}

// ----------------------------------------------------------------
//  LibRaw exif_parser_callback handler
// ----------------------------------------------------------------
static void exifCallback(void* context, int tag, int type, int len,
                          unsigned int ord, void* ifp, long long /*base*/) {
    auto* collector = static_cast<ExifCollector*>(context);

    // Guard: skip invalid entries
    if (type <= 0 || len <= 0) return;

    // Capture byte order from first tag
    // TIFF byte order is always 0x4949 (Intel/LE) or 0x4D4D (Motorola/BE)
    if (!collector->byteOrderSet) {
        collector->byteOrder = ord;
        collector->byteOrderSet = true;
    }

    // Extract real tag and determine IFD (exact match, NOT bitmask AND)
    // See LibRaw source: tiff.cpp, exif_gps.cpp, kodak.cpp, cr3_parser.cpp
    uint16_t realTag = static_cast<uint16_t>(tag & 0xFFFF);
    ExifIfd ifd;
    uint32_t upper = static_cast<uint32_t>(tag) & 0xFFF0000;

    if (upper == 0x50000)       ifd = EXIF_IFD_GPS;
    else if (upper == 0x40000)  ifd = EXIF_IFD_INTEROPERABILITY;
    else if (upper >= 0x100000) ifd = EXIF_IFD_0;          // (ifd+1)<<20
    else if (upper == 0x20000)  ifd = EXIF_IFD_0;          // Kodak
    else if (upper == 0x30000)  ifd = EXIF_IFD_0;          // Panasonic
    else if (upper == 0x60000)  ifd = EXIF_IFD_0;          // Sony SR2
    else if (upper == 0x70000 || upper == 0x80000) return;  // CR3 MakerNote
    else                        ifd = EXIF_IFD_EXIF;       // Plain ExifIFD

    // Skip MakerNote and IFD pointer tags
    if (realTag == 0x927c || realTag == 0x8769 ||
        realTag == 0xA005 || realTag == 0x8825) return;

    // Read raw data from ifp (LibRaw_abstract_datastream, NOT FILE*)
    size_t dataSize = static_cast<size_t>(len) * typeSize(type);
    if (dataSize == 0 || dataSize > 65536) return;  // Sanity check

    auto* stream = static_cast<LibRaw_abstract_datastream*>(ifp);
    ExifTagRecord rec;
    rec.tag = realTag;
    rec.type = static_cast<uint16_t>(type);
    rec.count = static_cast<uint32_t>(len);
    rec.ifd = ifd;
    rec.data.resize(dataSize);

    // Check read return value — skip incomplete reads
    if (stream->read(rec.data.data(), 1, dataSize) != static_cast<int>(dataSize)) return;

    collector->tags.push_back(std::move(rec));
}

void (*getExifCallback())(void*, int, int, int, unsigned int, void*, long long) {
    return exifCallback;
}

// ----------------------------------------------------------------
//  Helper: add a LONG entry to a specific IFD
// ----------------------------------------------------------------
static void addLongEntry(ExifData* d, ExifIfd ifd, ExifTag tag, uint32_t value) {
    ExifEntry* e = exif_entry_new();
    if (!e) return;
    e->tag = tag;
    e->format = EXIF_FORMAT_LONG;
    e->components = 1;
    e->size = 4;
    // NOTE: exif_entry_alloc() is static/private in libexif — use malloc().
    // Safe because exif_data_new() uses the default allocator (calloc/free),
    // and exif_entry_free() calls exif_mem_free() -> free() on entry->data.
    e->data = static_cast<uint8_t*>(malloc(4));
    if (!e->data) { exif_entry_unref(e); return; }
    exif_set_long(e->data, exif_data_get_byte_order(d), value);
    exif_content_add_entry(d->ifd[ifd], e);
    exif_entry_unref(e);
}

// ----------------------------------------------------------------
//  Build EXIF APP1 blob from collected tags
// ----------------------------------------------------------------
std::vector<uint8_t> buildExifBlob(const ExifCollector& collector,
                                    int outWidth, int outHeight) {
    if (collector.tags.empty()) return {};

    ExifData* exifData = exif_data_new();
    if (!exifData) return {};

    // Set byte order to match source file
    ExifByteOrder order = (collector.byteOrder == 0x4949)
                          ? EXIF_BYTE_ORDER_INTEL
                          : EXIF_BYTE_ORDER_MOTOROLA;
    exif_data_set_byte_order(exifData, order);

    for (const auto& rec : collector.tags) {
        // Override dimension tags with actual output dimensions
        if (rec.tag == 0x0100 || rec.tag == 0xA002) {
            addLongEntry(exifData, rec.ifd, static_cast<ExifTag>(rec.tag),
                         static_cast<uint32_t>(outWidth));
            continue;
        }
        if (rec.tag == 0x0101 || rec.tag == 0xA003) {
            addLongEntry(exifData, rec.ifd, static_cast<ExifTag>(rec.tag),
                         static_cast<uint32_t>(outHeight));
            continue;
        }

        // Create generic entry
        ExifEntry* entry = exif_entry_new();
        if (!entry) continue;
        entry->tag = static_cast<ExifTag>(rec.tag);
        entry->format = static_cast<ExifFormat>(rec.type);
        entry->components = rec.count;
        // malloc() because exif_entry_alloc() is private in libexif
        entry->data = static_cast<uint8_t*>(malloc(rec.data.size()));
        if (!entry->data) { exif_entry_unref(entry); continue; }
        memcpy(entry->data, rec.data.data(), rec.data.size());
        entry->size = rec.data.size();
        exif_content_add_entry(exifData->ifd[rec.ifd], entry);
        exif_entry_unref(entry);
    }

    // Ensure EXIF spec compliance — adds mandatory tags, fixes inconsistencies
    exif_data_set_option(exifData, EXIF_DATA_OPTION_FOLLOW_SPECIFICATION);
    exif_data_fix(exifData);

    // Serialize — output starts with "Exif\0\0" prefix (ready for JPEG APP1)
    unsigned char* blob = nullptr;
    unsigned int blobSize = 0;
    exif_data_save_data(exifData, &blob, &blobSize);

    std::vector<uint8_t> result;
    if (blob && blobSize > 0) {
        // EXIF APP1 max payload: 65533 bytes (JPEG marker constraint)
        if (blobSize <= 65533) {
            result.assign(blob, blob + blobSize);
        } else {
            fprintf(stderr, "[ExifInjector] EXIF blob too large (%u bytes), skipping\n", blobSize);
        }
    }

    exif_data_unref(exifData);
    // blob allocated by libexif's default allocator (calloc), free() is correct
    if (blob) free(blob);

    return result;
}

// ----------------------------------------------------------------
//  Inject EXIF APP1 into JPEG buffer
// ----------------------------------------------------------------
std::vector<uint8_t> injectExifIntoJpeg(const uint8_t* jpegData, size_t jpegSize,
                                          const std::vector<uint8_t>& exifBlob) {
    if (!jpegData || jpegSize < 4 || exifBlob.empty()) {
        // Return original data unchanged
        return std::vector<uint8_t>(jpegData, jpegData + jpegSize);
    }

    // Verify JPEG SOI marker (FFD8)
    if (jpegData[0] != 0xFF || jpegData[1] != 0xD8) {
        fprintf(stderr, "[ExifInjector] Not a valid JPEG (missing SOI marker)\n");
        return std::vector<uint8_t>(jpegData, jpegData + jpegSize);
    }

    // Build APP1 marker: FFE1 + 2-byte big-endian length + blob
    // Length includes the 2-byte length field itself
    uint16_t app1PayloadLen = static_cast<uint16_t>(exifBlob.size() + 2);
    std::vector<uint8_t> app1Marker;
    app1Marker.push_back(0xFF);
    app1Marker.push_back(0xE1);
    app1Marker.push_back(static_cast<uint8_t>((app1PayloadLen >> 8) & 0xFF));
    app1Marker.push_back(static_cast<uint8_t>(app1PayloadLen & 0xFF));
    app1Marker.insert(app1Marker.end(), exifBlob.begin(), exifBlob.end());

    // Find insertion point: after SOI (2 bytes) + all APP0/APP2 markers before SOS
    // TurboJPEG output structure: SOI + APP0 + [APP2 for ICC] + DQT/SOF/DHT/SOS...
    // We insert APP1 right after SOI, before APP0 (standard JPEG EXIF placement)
    size_t insertOffset = 2;  // Right after SOI (FFD8)

    // Alternatively, find the end of APP0 and insert after it
    // (some EXIF readers expect APP0 before APP1)
    if (jpegSize > 4 && jpegData[2] == 0xFF && jpegData[3] == 0xE0) {
        // APP0 marker found — skip past it
        uint16_t app0Len = (static_cast<uint16_t>(jpegData[4]) << 8) |
                           static_cast<uint16_t>(jpegData[5]);
        insertOffset = 2 + 2 + app0Len;  // SOI + marker(2) + payload

        // Also skip past APP2 markers (ICC profile from TurboJPEG)
        while (insertOffset + 4 <= jpegSize &&
               jpegData[insertOffset] == 0xFF && jpegData[insertOffset + 1] == 0xE2) {
            uint16_t app2Len = (static_cast<uint16_t>(jpegData[insertOffset + 2]) << 8) |
                               static_cast<uint16_t>(jpegData[insertOffset + 3]);
            insertOffset += 2 + app2Len;
        }
    }

    // Assemble final JPEG: [0..insertOffset] + APP1 + [insertOffset..end]
    std::vector<uint8_t> result;
    result.reserve(insertOffset + app1Marker.size() + (jpegSize - insertOffset));
    result.insert(result.end(), jpegData, jpegData + insertOffset);
    result.insert(result.end(), app1Marker.begin(), app1Marker.end());
    result.insert(result.end(), jpegData + insertOffset, jpegData + jpegSize);

    return result;
}

} // namespace rawalchemy
