// SPDX-License-Identifier: AGPL-3.0-or-later
// Flood-fill segmentation engine for the segmentation-based highlight reconstruction.
// Faithful port of darktable src/iop/hlreconstruct/segmentation.c (Hanno Schwalm, 2022).
//
// Works on int32 arrays with an extra border. The segmentation algorithm uses a
// modified floodfill that, while filling, also tracks each segment's bounding
// rectangle and marks border locations (high bit set via kSegIdMask).
//
// Ported 1:1; only the storage (std::vector vs dt_alloc_align) and OpenMP spellings
// differ. The dilation/erosion ring offset tables (radius 1-8) are copied verbatim —
// do not "simplify".
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rawalchemy {

// High bit flags a location as a segment BORDER (set during floodfill). The segment
// id is the low bits: data[loc] & (kSegIdMask - 1).
constexpr uint32_t kSegIdMask = 0x40000u;

struct Segmentation {
    std::vector<uint32_t> data;   // segment id per location (| kSegIdMask on borders)
    std::vector<uint32_t> tmp;    // scratch for morphological closing
    std::vector<int> size;        // pixel count per segment id
    std::vector<int> xmin, xmax, ymin, ymax;  // bounding box per segment id
    std::vector<float> val1, val2;  // caller scratch (segbased stores candidate + refavg)
    int nr = 2;       // next free segment id (0 and 1 reserved)
    int border = 0;
    int slots = 0;
    int width = 0;
    int height = 0;
};

// Allocate arrays for a width×height plane with `border` margin and `slots` segment
// ids. Caller pre-fills .data with 1 (= segment candidate) / 0 (= background) before
// segmentizePlane. Returns false on success, true on alloc failure (darktable convention).
bool segmentationInit(Segmentation& seg, int width, int height, int border, int slots);

// Flood-fill connected components of data==1, assigning ascending ids from 2.
// Records bounding boxes; marks borders with kSegIdMask. Drops segments < 4 px.
void segmentizePlane(Segmentation& seg);

// Morphological closing (dilate radius, then erode radius-3) to merge nearby segments.
void segmentsCombine(Segmentation& seg, int radius);

// Look up the segment id at a linear location. Returns the id (2..nr-1) or 0.
inline uint32_t getSegmentId(const Segmentation& seg, size_t loc) {
    if (loc >= (size_t)seg.width * (size_t)(seg.height - seg.border)) return 0;
    const uint32_t id = seg.data[loc] & (kSegIdMask - 1u);
    return (id < (uint32_t)seg.nr && id > 1u) ? id : 0u;
}

} // namespace rawalchemy
