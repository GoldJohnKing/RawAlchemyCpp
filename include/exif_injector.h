#pragma once
/**
 * @file exif_injector.h
 * @brief EXIF metadata extraction from RAW files and injection into JPEG output.
 *
 * Uses LibRaw's exif_parser_callback to collect EXIF tags from RAW files,
 * then libexif to serialize them into an APP1 blob for JPEG embedding.
 * MakerNote is deliberately dropped to avoid offset corruption issues.
 */

#include <cstdint>
#include <cstdio>
#include <vector>

// Forward declarations — avoid exposing LibRaw/libexif headers
class LibRaw_abstract_datastream;

namespace rawalchemy {

/**
 * @brief Collected EXIF tag data from LibRaw's callback.
 *
 * Set via LibRaw::set_exifparser_handler() before open_file().
 * Thread-safe when used from a single LibRaw instance (callback is sequential).
 */
struct ExifCollector;

/**
 * @brief Create an ExifCollector ready for LibRaw callback.
 */
ExifCollector* createExifCollector();

/**
 * @brief Free an ExifCollector and all collected tag data.
 */
void destroyExifCollector(ExifCollector* collector);

/**
 * @brief Get the LibRaw exif_parser_callback function pointer.
 *
 * Pass this + the collector to LibRaw::set_exifparser_handler().
 */
void (*getExifCallback())(void*, int, int, int, unsigned int, void*, long long);

/**
 * @brief Build an EXIF APP1 blob from collected tags for JPEG embedding.
 *
 * @param collector  Tags collected during LibRaw::open_file()
 * @param outWidth   Output image width (overrides EXIF dimension tags)
 * @param outHeight  Output image height (overrides EXIF dimension tags)
 * @return EXIF APP1 blob (including "Exif\0\0" prefix), or empty on failure
 */
std::vector<uint8_t> buildExifBlob(const ExifCollector& collector,
                                    int outWidth, int outHeight);

/**
 * @brief Inject an EXIF APP1 blob into a JPEG buffer.
 *
 * Inserts the APP1 marker between SOI+APP0 and the rest of the JPEG data.
 * If injection fails, returns the original buffer unchanged (graceful degradation).
 *
 * @param jpegData   Original JPEG data (from TurboJPEG compression)
 * @param jpegSize   Size of original JPEG data
 * @param exifBlob   EXIF APP1 blob (from buildExifBlob)
 * @return Modified JPEG data with EXIF embedded, or original data on failure
 */
std::vector<uint8_t> injectExifIntoJpeg(const uint8_t* jpegData, size_t jpegSize,
                                          const std::vector<uint8_t>& exifBlob);

} // namespace rawalchemy
