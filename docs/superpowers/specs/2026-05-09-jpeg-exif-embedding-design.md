# Design: JPEG EXIF Embedding via LibRaw Callback + libexif

**Date:** 2026-05-09
**Status:** Approved
**Scope:** JPEG output only; TIFF output unchanged (no EXIF)

## Problem

Generated JPEG files from RawAlchemyCpp contain no EXIF metadata from the original RAW file. Camera settings, lens info, exposure parameters, GPS data, timestamps, and other metadata are lost.

## Requirements

1. Extract standard EXIF fields from RAW files and embed into JPEG output
2. Cover all major RAW formats: NEF, CR2, CR3, ARW, RW2, RAF, ORF, DNG, PEF, SRW, KDC
3. Gracefully degrade: if EXIF extraction fails, produce valid JPEG without EXIF
4. **Drop MakerNote** — only preserve standard EXIF fields
5. Update image dimension tags to match actual output dimensions
6. No GPLv2 dependencies (project is AGPL v3)

## Approach: LibRaw exif_parser_callback + libexif

### Why this approach

The initial approach (extracting EXIF from RAW's embedded JPEG thumbnail) failed for Nikon NEF files — their thumbnails don't contain EXIF APP1 markers. The EXIF data is stored directly in the RAW file's TIFF structure, not duplicated in thumbnails.

Three alternatives were evaluated:

| Approach | Coverage | MakerNote | Complexity | License |
|----------|----------|-----------|------------|---------|
| Thumbnail extraction | Partial (fails for NEF) | Preserved | Low | N/A |
| Exiv2 library | Full | Preserved | Low (library) | ⚠️ GPLv2 — incompatible |
| LibRaw callback + libexif | Full standard EXIF | Dropped | Medium | ✅ LGPL |
| LibRaw callback + manual TIFF construction | Full standard EXIF | Dropped | High | ✅ LGPL |

**LibRaw callback + libexif** is chosen because:
- Covers all major RAW formats (NEF, CR2, CR3, ARW, etc.)
- LGPL-compatible (both LibRaw LGPL/CDDL and libexif LGPL v2.1)
- libexif handles TIFF IFD serialization (no manual byte-level TIFF construction)
- ~200KB library size, no external dependencies
- Dropping MakerNote simplifies implementation and avoids offset corruption

### MakerNote decision

MakerNote (EXIF tag 0x927c) contains manufacturer-specific binary data. Dropping it:
- **Preserves** 95%+ of photographer-relevant metadata via standard EXIF fields
- **Loses** camera-specific settings (Picture Control, D-Lighting, detailed AF info, firmware version)
- **Eliminates** the offset corruption problem (MakerNote internal offsets break when re-serialized)
- **Reduces** EXIF blob size significantly (MakerNotes can be 30-50KB)

## Architecture

### Data Flow

```
RAW file
  |
  v
LibRaw::open_file() + exif_parser_callback
  |  (fires for every EXIF tag during parsing)
  v
ExifCollector (struct)
  |  - Collects tag/type/length/raw-bytes per IFD
  |  - Routes to IFD0 / ExifIFD / GPS / Interop
  |  - Skips MakerNote (tag 0x927c)
  v
libexif ExifData construction
  |  - Creates ExifEntry for each collected tag
  |  - Sets byte order from original file
  |  - Adds entries to appropriate IFD
  v
exif_data_save_data()
  |  - Serializes to EXIF APP1 blob (with "Exif\0\0" prefix)
  v
patchExifDimensions()
  |  - Updates IFD0: 0x0100/0x0101
  |  - Updates ExifIFD: 0xA002/0xA003
  v
injectExifIntoJpeg()
  |  - Post-processes TurboJPEG output
  |  - Inserts APP1 between SOI+APP0 and rest of JPEG
  v
JPEG file with EXIF
```

### Callback Tag Routing

LibRaw's callback encodes the IFD source in upper bits of the tag number:

| Upper bits | Source | Route to | LibRaw source |
|------------|--------|-----------|---------------|
| `(ifd+1) << 20` (≥`0x100000`) | IFD0 / IFD1 | `EXIF_IFD_0` | `tiff.cpp:65` |
| `0x00000` | ExifIFD | `EXIF_IFD_EXIF` | `exif_gps.cpp:93` |
| `0x50000` | GPS IFD | `EXIF_IFD_GPS` | `exif_gps.cpp:359` |
| `0x40000` | Interop IFD | `EXIF_IFD_INTEROPERABILITY` | `exif_gps.cpp:41` |
| `0x20000` | Kodak private | `EXIF_IFD_0` | `kodak.cpp:115` |
| `0x30000` | Panasonic RAW | `EXIF_IFD_0` | `tiff.cpp:65` |
| `0x60000` | Sony SR2Private | `EXIF_IFD_0` | `tiff.cpp:1598` |
| `0x70000`, `0x80000` | CR3 parser | Skip (MakerNote-related) | `cr3_parser.cpp` |

Strip upper bits: `real_tag = tag & 0xFFFF`

**IMPORTANT:** Use exact equality for mask checks, NOT bitmask AND. `0x50000 & 0x40000 = 0x40000` (truthy) — interop tags would be misrouted to GPS if checked with `&`.

```cpp
uint32_t upper = tag & 0xFFF0000;
if (upper == 0x50000)       ifd = EXIF_IFD_GPS;
else if (upper == 0x40000)  ifd = EXIF_IFD_INTEROPERABILITY;
else if (upper >= 0x100000) ifd = EXIF_IFD_0;  // (ifd+1)<<20
else if (upper == 0x20000)  ifd = EXIF_IFD_0;  // Kodak
else if (upper == 0x30000)  ifd = EXIF_IFD_0;  // Panasonic
else if (upper == 0x60000)  ifd = EXIF_IFD_0;  // Sony
else if (upper == 0x70000 || upper == 0x80000) return; // CR3 MakerNote
else                        ifd = EXIF_IFD_EXIF; // Plain ExifIFD
```

### Tags to Skip

| Tag | Reason |
|-----|--------|
| `0x927c` (MakerNote) | Dropped per design decision |
| `0x8769` (ExifIFD pointer) | libexif manages IFD pointers internally |
| `0xA005` (InteropIFD pointer) | libexif manages internally |
| `0x8825` (GPSInfoIFD pointer) | libexif manages internally |

### Tags with Special Handling (Overridden During Build)

| Tag | Override | Reason |
|-----|----------|--------|
| `0x0100` (ImageWidth) | Set to output width | Output dimensions differ from RAW sensor |
| `0x0101` (ImageLength) | Set to output height | Output dimensions differ from RAW sensor |
| `0xA002` (PixelXDimension) | Set to output width | Must match actual JPEG pixel dimensions |
| `0xA003` (PixelYDimension) | Set to output height | Must match actual JPEG pixel dimensions |

These tags are **collected normally** in the callback and **overridden** in `buildExifBlob()` with actual output dimensions.

## New Dependency: libexif

| Aspect | Details |
|--------|---------|
| **Version** | v0.6.25 |
| **License** | LGPL v2.1 (compatible with AGPL v3) |
| **Language** | Pure C |
| **Dependencies** | None |
| **Binary size** | ~200-300 KB static lib |
| **Integration** | Git submodule + custom CMakeLists.txt |
| **Source files needed** | ~12 core .c files + camera-specific mnote files |

## File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `include/exif_injector.h` | Public API: extract, patch, inject functions |
| `src/exif_injector.cpp` | Callback handler, libexif construction, dimension patching, JPEG injection |

### Modified Files

| File | Change |
|------|--------|
| `include/jpeg_writer.h` | Add optional `exifData` parameter to `writeJpeg()` |
| `src/jpeg_writer.cpp` | Post-process TurboJPEG output with EXIF injection |
| `src/main.cpp` | Extract EXIF in JPEG save path, pass to `writeJpeg()` |
| `src/raw_alchemy_capi.cpp` | C API: extract and pass EXIF for JPEG output |
| `CMakeLists.txt` | Add libexif submodule + exif_injector.cpp |
| `.gitmodules` | Add libexif submodule |

## Implementation Details

### 1. ExifCollector (callback data structure)

```cpp
struct ExifTagRecord {
    uint16_t tag;       // Real tag number (upper bits stripped)
    uint16_t type;      // TIFF data type (1=BYTE, 2=ASCII, 3=SHORT, 4=LONG, 5=RATIONAL, 7=UNDEFINED, 10=SRATIONAL)
    uint32_t count;     // Number of components
    std::vector<uint8_t> data;  // Raw data bytes read from stream
    ExifIfd ifd;        // Target IFD (IFD_0, IFD_EXIF, IFD_GPS, IFD_INTEROPERABILITY)
};

struct ExifCollector {
    std::vector<ExifTagRecord> tags;
    unsigned int byteOrder = 0;  // TIFF byte order from source file
    bool byteOrderSet = false;   // Separate flag (ord could theoretically be 0)
};

// TIFF data type sizes (per EXIF 2.3 spec)
static size_t typeSize(int type) {
    static const size_t sizes[] = {0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8};
    return (type >= 1 && type <= 12) ? sizes[type] : 1;
}
```

### 2. Callback Handler

```cpp
static void exifCallback(void* context, int tag, int type, int len,
                          unsigned int ord, void* ifp, INT64 base) {
    auto* collector = static_cast<ExifCollector*>(context);

    // Guard: skip invalid entries
    if (type <= 0 || len <= 0) return;

    // Capture byte order from first tag (TIFF byte order is always 0x4949 or 0x4D4D)
    if (!collector->byteOrderSet) {
        collector->byteOrder = ord;
        collector->byteOrderSet = true;
    }

    // Extract real tag and determine IFD (exact match, NOT bitmask AND)
    uint16_t realTag = tag & 0xFFFF;
    ExifIfd ifd;
    uint32_t upper = tag & 0xFFF0000;
    if (upper == 0x50000)       ifd = EXIF_IFD_GPS;
    else if (upper == 0x40000)  ifd = EXIF_IFD_INTEROPERABILITY;
    else if (upper >= 0x100000) ifd = EXIF_IFD_0;       // (ifd+1)<<20
    else if (upper == 0x20000)  ifd = EXIF_IFD_0;       // Kodak
    else if (upper == 0x30000)  ifd = EXIF_IFD_0;       // Panasonic
    else if (upper == 0x60000)  ifd = EXIF_IFD_0;       // Sony
    else if (upper == 0x70000 || upper == 0x80000) return; // CR3 MakerNote
    else                        ifd = EXIF_IFD_EXIF;    // Plain ExifIFD

    // Skip MakerNote and IFD pointer tags
    if (realTag == 0x927c || realTag == 0x8769 ||
        realTag == 0xA005 || realTag == 0x8825) return;

    // Read raw data from ifp (LibRaw_abstract_datastream, NOT FILE*)
    size_t dataSize = len * typeSize(type);
    if (dataSize == 0 || dataSize > 65536) return;  // Sanity check
    auto* stream = static_cast<LibRaw_abstract_datastream*>(ifp);
    ExifTagRecord rec;
    rec.tag = realTag;
    rec.type = type;
    rec.count = len;
    rec.ifd = ifd;
    rec.data.resize(dataSize);
    stream->read(rec.data.data(), 1, dataSize);

    collector->tags.push_back(std::move(rec));
}
```

### 3. libexif ExifData Construction

```cpp
std::vector<uint8_t> buildExifBlob(ExifCollector& collector, int outWidth, int outHeight) {
    ExifData* exifData = exif_data_new();
    exif_data_set_byte_order(exifData,
        (collector.byteOrder == 0x4949) ? EXIF_BYTE_ORDER_INTEL : EXIF_BYTE_ORDER_MOTOROLA);

    for (auto& rec : collector.tags) {
        // Override dimension tags with output dimensions
        if (rec.tag == 0x0100 || rec.tag == 0xA002) {
            // Width — create LONG entry with output width
            addLongEntry(exifData, rec.ifd, rec.tag, (uint32_t)outWidth);
            continue;
        }
        if (rec.tag == 0x0101 || rec.tag == 0xA003) {
            // Height
            addLongEntry(exifData, rec.ifd, rec.tag, (uint32_t)outHeight);
            continue;
        }

        // Create generic entry
        ExifEntry* entry = exif_entry_new();
        entry->tag = rec.tag;
        entry->format = (ExifFormat)rec.type;
        entry->components = rec.count;
        entry->data = (uint8_t*)exif_entry_alloc(entry, rec.data.size());
        if (!entry->data) { exif_entry_unref(entry); continue; }
        memcpy(entry->data, rec.data.data(), rec.data.size());
        entry->size = rec.data.size();
        exif_content_add_entry(exif_data->ifd[rec.ifd], entry);
        exif_entry_unref(entry);
    }

    // Serialize (output starts with "Exif\0\0" prefix — ready for JPEG APP1)
    unsigned char* blob = nullptr;
    unsigned int blobSize = 0;
    exif_data_save_data(exifData, &blob, &blobSize);

    std::vector<uint8_t> result(blob, blob + blobSize);
    exif_data_unref(exifData);
    // blob allocated by libexif's default allocator (malloc), free() is correct
    if (blob) free(blob);

    return result;
}
```

### 4. JPEG Injection (unchanged from previous design)

Post-process TurboJPEG output buffer to insert EXIF APP1 between SOI+APP0 and the remaining JPEG data.

## Error Handling

All errors result in graceful degradation — the output JPEG is always valid:

| Failure Point | Behavior |
|---------------|----------|
| LibRaw open_file fails | Return empty EXIF → no EXIF in output |
| Callback collects nothing | Skip EXIF injection |
| libexif serialization fails | Skip EXIF injection |
| EXIF blob exceeds 65533 bytes | Skip EXIF injection, log warning |
| JPEG injection fails | Write original JPEG buffer |

## Size Constraints

- EXIF APP1 max payload: 65,533 bytes
- Without MakerNote, typical EXIF is 2-5 KB — well within limits
- Standard EXIF tags are small (SHORT/LONG/RATIONAL entries)

## Not Supported

- CIFF (old Canon CRW format) — no callback fires
- Phase One IIQ — no callback fires
- Sigma X3F — no callback fires
- MakerNote — deliberately dropped
- IPTC/XMP metadata — out of scope
- TIFF EXIF embedding — out of scope

## Testing Strategy

1. Process `Test/Sample.NEF` → output JPEG → verify EXIF with `exiftool` or `python3 -c "import PIL; ..."`
2. Verify standard EXIF fields present: Make=Nikon, Model=Z 8, ExposureTime, FNumber, ISO, FocalLength
3. Verify image dimensions match output (not RAW sensor dimensions)
4. Verify JPEG is valid (can be opened in image viewers)
5. Test TIFF output still works (no EXIF change)
