// SPDX-License-Identifier: AGPL-3.0-or-later
// Segmentation engine — see nn_segmentation.h.
// Faithful port of darktable src/iop/hlreconstruct/segmentation.c.
// The dilation/erosion ring offsets and the floodfill border-marking logic are
// copied verbatim; only storage (std::vector) and OpenMP spellings differ.

#include "nn_segmentation.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace rawalchemy {
namespace {

struct Pos { int xpos; int ypos; };

// Morphological dilation test: returns nonzero if any cell in the dilation ring
// of the given radius (1..8) around img[i] is set. Verbatim offset tables from
// darktable segmentation.c:104-209 — the ring shape is tuned, not arbitrary.
int testDilate(const uint32_t* img, size_t i, size_t w1, int radius) {
    int retval = img[i - w1 - 1] | img[i - w1] | img[i - w1 + 1] |
                 img[i - 1]      | img[i]      | img[i + 1] |
                 img[i + w1 - 1] | img[i + w1] | img[i + w1 + 1];
    if (retval || radius < 2) return retval;

    const size_t w2 = 2 * w1;
    retval = img[i - w2 - 1] | img[i - w2]    | img[i - w2 + 1] |
             img[i - w1 - 2] | img[i - w1 + 2] |
             img[i - 2]      | img[i + 2] |
             img[i + w1 - 2] | img[i + w1 + 2] |
             img[i + w2 - 1] | img[i + w2]    | img[i + w2 + 1];
    if (retval || radius < 3) return retval;

    const size_t w3 = 3 * w1;
    retval = img[i - w3 - 2] | img[i - w3 - 1] | img[i - w3] | img[i - w3 + 1] | img[i - w3 + 2] |
             img[i - w2 - 3] | img[i - w2 - 2] | img[i - w2 + 2] | img[i - w2 + 3] |
             img[i - w1 - 3] | img[i - w1 + 3] |
             img[i - 3]      | img[i + 3] |
             img[i + w1 - 3] | img[i + w1 + 3] |
             img[i + w2 - 3] | img[i + w2 - 2] | img[i + w2 + 2] | img[i + w2 + 3] |
             img[i + w3 - 2] | img[i + w3 - 1] | img[i + w3] | img[i + w3 + 1] | img[i + w3 + 2];
    if (retval || radius < 4) return retval;

    const size_t w4 = 4 * w1;
    retval = img[i - w4 - 2] | img[i - w4 - 1] | img[i - w4] | img[i - w4 + 1] | img[i - w4 + 2] |
             img[i - w3 - 3] | img[i - w3 + 3] |
             img[i - w2 - 4] | img[i - w2 + 4] |
             img[i - w1 - 4] | img[i - w1 + 4] |
             img[i - 4]      | img[i + 4] |
             img[i + w1 - 4] | img[i + w1 + 4] |
             img[i + w2 - 4] | img[i + w2 + 4] |
             img[i + w3 - 3] | img[i + w3 + 3] |
             img[i + w4 - 2] | img[i + w4 - 1] | img[i + w4] | img[i + w4 + 1] | img[i + w4 + 2];
    if (retval || radius < 5) return retval;

    const size_t w5 = 5 * w1;
    retval = img[i - w5 - 2] | img[i - w5 - 1] | img[i - w5] | img[i - w5 + 1] | img[i - w5 + 2] |
             img[i - w4 - 4] | img[i - w4 - 3] | img[i - w4 + 3] | img[i - w4 + 4] |
             img[i - w3 - 4] | img[i - w3 + 4] |
             img[i - w2 - 5] | img[i - w2 + 5] |
             img[i - w1 - 5] | img[i - w1 + 5] |
             img[i - 5]      | img[i + 5] |
             img[i + w1 - 5] | img[i + w1 + 5] |
             img[i + w2 - 5] | img[i + w2 + 5] |
             img[i + w3 - 4] | img[i + w3 + 4] |
             img[i + w4 - 4] | img[i + w4 - 3] | img[i + w4 + 3] | img[i + w4 + 4] |
             img[i + w5 - 2] | img[i + w5 - 1] | img[i + w5] | img[i + w5 + 1] | img[i + w5 + 2];
    if (retval || radius < 6) return retval;

    const size_t w6 = 6 * w1;
    retval = img[i - w6 - 2] | img[i - w6 - 1] | img[i - w6] | img[i - w6 + 1] | img[i - w6 + 2] |
             img[i - w5 - 4] | img[i - w5 - 3] | img[i - w5 + 3] | img[i - w5 + 4] |
             img[i - w4 - 5] | img[i - w4 + 5] |
             img[i - w3 - 5] | img[i - w3 + 5] |
             img[i - w2 - 6] | img[i - w2 + 6] |
             img[i - w1 - 6] | img[i - w1 + 6] |
             img[i - 6]      | img[i + 6] |
             img[i + w1 - 6] | img[i + w1 + 6] |
             img[i + w2 - 6] | img[i + w2 + 6] |
             img[i + w3 - 5] | img[i + w3 + 5] |
             img[i + w4 - 5] | img[i + w4 + 5] |
             img[i + w5 - 4] | img[i + w5 - 3] | img[i + w5 + 3] | img[i + w5 + 4] |
             img[i + w6 - 2] | img[i + w6 - 1] | img[i + w6] | img[i + w6 + 1] | img[i + w6 + 2];
    if (retval || radius < 7) return retval;

    const size_t w7 = 7 * w1;
    retval = img[i - w7 - 3] | img[i - w7 - 2] | img[i - w7 - 1] | img[i - w7] | img[i - w7 + 1] | img[i - w7 + 2] | img[i - w7 + 3] |
             img[i - w6 - 4] | img[i - w6 - 3] | img[i - w6 + 3] | img[i - w6 + 4] |
             img[i - w5 - 6] | img[i - w5 - 5] | img[i - w5 + 5] | img[i - w5 + 6] |
             img[i - w4 - 6] | img[i - w4 + 6] |
             img[i - w3 - 7] | img[i - w3 - 6] | img[i - w3 + 6] | img[i - w3 + 7] |
             img[i - w2 - 7] | img[i - w2 + 7] |
             img[i - w1 - 7] | img[i - w1 + 7] |
             img[i - 7]      | img[i + 7] |
             img[i + w1 - 7] | img[i + w1 + 7] |
             img[i + w2 - 7] | img[i + w2 + 7] |
             img[i + w3 - 7] | img[i + w3 - 6] | img[i + w3 + 6] | img[i + w3 + 7] |
             img[i + w4 - 6] | img[i + w4 + 6] |
             img[i + w5 - 6] | img[i + w5 - 5] | img[i + w5 + 5] | img[i + w5 + 6] |
             img[i + w6 - 4] | img[i + w6 - 3] | img[i + w6 + 3] | img[i + w6 + 4] |
             img[i + w7 - 3] | img[i + w7 - 2] | img[i + w7 - 1] | img[i + w7] | img[i + w7 + 1] | img[i + w7 + 2] | img[i + w7 + 3];
    if (retval || radius < 8) return retval;

    const size_t w8 = 8 * w1;
    retval = img[i - w8 - 4] | img[i - w8 - 3] | img[i - w8 - 2] | img[i - w8 - 1] | img[i - w8] | img[i - w8 + 1] | img[i - w8 + 2] | img[i - w8 + 3] | img[i - w8 + 4] |
             img[i - w7 - 6] | img[i - w7 - 5] | img[i - w7 - 4] | img[i - w7 + 4] | img[i - w7 + 5] | img[i - w7 + 6] |
             img[i - w6 - 6] | img[i - w6 - 5] | img[i - w6 + 5] | img[i - w6 + 6] |
             img[i - w5 - 7] | img[i - w5 + 6] |
             img[i - w4 - 8] | img[i - w4 - 7] | img[i - w4 + 7] | img[i - w4 + 8] |
             img[i - w3 - 8] | img[i - w3 - 7] | img[i - w3 + 7] | img[i - w3 + 8] |
             img[i - w2 - 8] | img[i - w2 + 8] |
             img[i - w1 - 8] | img[i - w1 + 8] |
             img[i - 8]      | img[i + 8] |
             img[i + w1 - 8] | img[i + w1 + 8] |
             img[i + w2 - 8] | img[i + w2 + 8] |
             img[i + w3 - 8] | img[i + w3 - 7] | img[i + w3 + 7] | img[i + w3 + 8] |
             img[i + w4 - 8] | img[i + w4 - 7] | img[i + w4 + 7] | img[i + w4 + 8] |
             img[i + w5 - 7] | img[i + w5 + 7] |
             img[i + w6 - 6] | img[i + w6 - 5] | img[i + w6 + 5] | img[i + w6 + 6] |
             img[i + w7 - 6] | img[i + w7 - 5] | img[i + w7 - 4] | img[i + w7 + 4] | img[i + w7 + 5] | img[i + w7 + 6] |
             img[i + w8 - 4] | img[i + w8 - 3] | img[i + w8 - 2] | img[i + w8 - 1] | img[i + w8] | img[i + w8 + 1] | img[i + w8 + 2] | img[i + w8 + 3] | img[i + w8 + 4];
    return retval;
}

// Morphological erosion test: returns 1 only if EVERY cell in the erosion ring is set.
// Verbatim offset tables from darktable segmentation.c:229-280 (radius 1-5 only;
// segmentsCombine erodes with radius-3, and the closing radius is capped at 8 → erode ≤5).
int testErode(const uint32_t* img, size_t i, size_t w1, int radius) {
    int retval = img[i - w1 - 1] & img[i - w1] & img[i - w1 + 1] &
                 img[i - 1]     & img[i]     & img[i + 1] &
                 img[i + w1 - 1] & img[i + w1] & img[i + w1 + 1];
    if (retval == 0 || radius < 2) return retval;

    const size_t w2 = 2 * w1;
    retval = img[i - w2 - 1] & img[i - w2]    & img[i - w2 + 1] &
             img[i - w1 - 2] & img[i - w1 + 2] &
             img[i - 2]      & img[i + 2] &
             img[i + w1 - 2] & img[i + w1 + 2] &
             img[i + w2 - 1] & img[i + w2]    & img[i + w2 + 1];
    if (retval == 0 || radius < 3) return retval;

    const size_t w3 = 3 * w1;
    retval = img[i - w3 - 2] & img[i - w3 - 1] & img[i - w3] & img[i - w3 + 1] & img[i - w3 + 2] &
             img[i - w2 - 3] & img[i - w2 - 2] & img[i - w2 + 2] & img[i - w2 + 3] &
             img[i - w1 - 3] & img[i - w1 + 3] &
             img[i - 3]      & img[i + 3] &
             img[i + w1 - 3] & img[i + w1 + 3] &
             img[i + w2 - 3] & img[i + w2 - 2] & img[i + w2 + 2] & img[i + w2 + 3] &
             img[i + w3 - 2] & img[i + w3 - 1] & img[i + w3] & img[i + w3 + 1] & img[i + w3 + 2];
    if (retval == 0 || radius < 4) return retval;

    const size_t w4 = 4 * w1;
    retval = img[i - w4 - 2] & img[i - w4 - 1] & img[i - w4] & img[i - w4 + 1] & img[i - w4 + 2] &
             img[i - w3 - 3] & img[i - w3 + 3] &
             img[i - w2 - 4] & img[i - w2 + 4] &
             img[i - w1 - 4] & img[i - w1 + 4] &
             img[i - 4]      & img[i + 4] &
             img[i + w1 - 4] & img[i + w1 + 4] &
             img[i + w2 - 4] & img[i + w2 + 4] &
             img[i + w3 - 3] & img[i + w3 + 3] &
             img[i + w4 - 2] & img[i + w4 - 1] & img[i + w4] & img[i + w4 + 1] & img[i + w4 + 2];
    if (retval == 0 || radius < 5) return retval;

    const size_t w5 = 5 * w1;
    retval = img[i - w5 - 2] & img[i - w5 - 1] & img[i - w5] & img[i - w5 + 1] & img[i - w5 + 2] &
             img[i - w4 - 4] & img[i - w4 - 3] & img[i - w4 + 3] & img[i - w4 + 4] &
             img[i - w3 - 4] & img[i - w3 + 4] &
             img[i - w2 - 5] & img[i - w2 + 5] &
             img[i - w1 - 5] & img[i - w1 + 5] &
             img[i - 5]      & img[i + 5] &
             img[i + w1 - 5] & img[i + w1 + 5] &
             img[i + w2 - 5] & img[i + w2 + 5] &
             img[i + w3 - 4] & img[i + w3 + 4] &
             img[i + w4 - 4] & img[i + w4 - 3] & img[i + w4 + 3] & img[i + w4 + 4] &
             img[i + w5 - 2] & img[i + w5 - 1] & img[i + w5] & img[i + w5 + 1] & img[i + w5 + 2];
    return retval;
}

void dilating(const uint32_t* img, uint32_t* o, int w1, int height, int border, int radius) {
    for (int row = border; row < height - border; ++row) {
        for (int col = border; col < w1 - border; ++col) {
            const size_t i = (size_t)row * w1 + col;
            o[i] = testDilate(img, i, w1, radius) ? 1u : 0u;
        }
    }
}

void eroding(const uint32_t* img, uint32_t* o, int w1, int height, int border, int radius) {
    for (int row = border; row < height - border; ++row) {
        for (int col = border; col < w1 - border; ++col) {
            const size_t i = (size_t)row * w1 + col;
            o[i] = testErode(img, i, w1, radius) ? 1u : 0u;
        }
    }
}

void intimageBorderfill(uint32_t* d, int width, int height, int val, int border) {
    const size_t di = (size_t)(height - border - 1) * width;
    for (int i = 0; i < border * width; ++i)
        d[i] = d[i + di] = (uint32_t)val;
    for (int row = border; row < height - border; ++row) {
        const size_t j = (size_t)row * width;
        const int dj = width - border;
        for (int i = 0; i < border; ++i)
            d[j + i] = d[j + i + dj] = (uint32_t)val;
    }
}

// Clear a segment slot's accumulators.
void clearSegmentSlot(Segmentation& seg, uint32_t id) {
    if ((int)id > seg.slots - 1) return;
    seg.size[id] = seg.xmin[id] = seg.xmax[id] = seg.ymin[id] = seg.ymax[id] = 0;
    seg.val1[id] = seg.val2[id] = 0.0f;
}

// The core floodfill. Fills the connected component of data==1 starting at (yin,xin)
// with `id`, scanning left+right per row and pushing up/down neighbors. Marks border
// pixels (those adjacent to the fill but ==0) with (id | kSegIdMask) and grows the
// bounding box. Drops the segment (reverts) if it has ≤3 pixels.
bool floodfillSegmentize(int yin, int xin, Segmentation& seg, int w, int h, int id,
                         std::vector<Pos>& stack) {
    if (id >= seg.slots - 2) return false;

    const int border = seg.border;
    uint32_t* d = seg.data.data();
    int min_x = xin, max_x = xin, min_y = yin, max_y = yin;
    int cnt = 0;
    stack.clear();
    clearSegmentSlot(seg, id);

    stack.push_back({xin, yin});
    while (!stack.empty()) {
        Pos coord = stack.back();
        stack.pop_back();
        const int x = coord.xpos;
        const int y = coord.ypos;
        if (d[(size_t)y * w + x] != 1u) continue;

        const int yUp = y - 1;
        const int yDown = y + 1;
        bool lastXUp = false, lastXDown = false, firstXUp = false, firstXDown = false;
        d[(size_t)y * w + x] = (uint32_t)id;
        cnt++;

        // Up neighbor
        if (yUp >= border && d[(size_t)yUp * w + x] == 1u) {
            stack.push_back({x, yUp}); firstXUp = lastXUp = true;
        } else {
            const int xp = x, yp = yUp;
            const size_t rp = (size_t)yp * w + xp;
            if (yp > border + 1 && d[rp] == 0u) {
                min_x = std::min(min_x, xp); max_x = std::max(max_x, xp);
                min_y = std::min(min_y, yp); max_y = std::max(max_y, yp);
                d[rp] = (uint32_t)id | kSegIdMask;
            }
        }
        // Down neighbor
        if (yDown < h - border && d[(size_t)yDown * w + x] == 1u) {
            stack.push_back({x, yDown}); firstXDown = lastXDown = true;
        } else {
            const int xp = x, yp = yDown;
            const size_t rp = (size_t)yp * w + xp;
            if (yp < h - border - 2 && d[rp] == 0u) {
                min_x = std::min(min_x, xp); max_x = std::max(max_x, xp);
                min_y = std::min(min_y, yp); max_y = std::max(max_y, yp);
                d[rp] = (uint32_t)id | kSegIdMask;
            }
        }

        // Scan right
        int xr = x + 1;
        while (xr < w - border && d[(size_t)y * w + xr] == 1u) {
            d[(size_t)y * w + xr] = (uint32_t)id;
            cnt++;
            if (yUp >= border && d[(size_t)yUp * w + xr] == 1u) {
                if (!lastXUp) { stack.push_back({xr, yUp}); lastXUp = true; }
            } else {
                const int xp = xr, yp = yUp;
                const size_t rp = (size_t)yp * w + xp;
                if (yp > border + 1 && d[rp] == 0u) {
                    min_x = std::min(min_x, xp); max_x = std::max(max_x, xp);
                    min_y = std::min(min_y, yp); max_y = std::max(max_y, yp);
                    d[rp] = (uint32_t)id | kSegIdMask;
                }
                lastXUp = false;
            }
            if (yDown < h - border && d[(size_t)yDown * w + xr] == 1u) {
                if (!lastXDown) { stack.push_back({xr, yDown}); lastXDown = true; }
            } else {
                const int xp = xr, yp = yDown;
                const size_t rp = (size_t)yp * w + xp;
                if (yp < h - border - 2 && d[rp] == 0u) {
                    min_x = std::min(min_x, xp); max_x = std::max(max_x, xp);
                    min_y = std::min(min_y, yp); max_y = std::max(max_y, yp);
                    d[rp] = (uint32_t)id | kSegIdMask;
                }
                lastXDown = false;
            }
            xr++;
        }
        // Right border
        {
            const int xp = xr, yp = y;
            const size_t rp = (size_t)yp * w + xp;
            if (xp < w - border - 2 && d[rp] == 0u) {
                min_x = std::min(min_x, xp); max_x = std::max(max_x, xp);
                min_y = std::min(min_y, yp); max_y = std::max(max_y, yp);
                d[rp] = (uint32_t)id | kSegIdMask;
            }
        }

        // Scan left
        int xl = x - 1;
        lastXUp = firstXUp;
        lastXDown = firstXDown;
        while (xl >= border && d[(size_t)y * w + xl] == 1u) {
            d[(size_t)y * w + xl] = (uint32_t)id;
            cnt++;
            if (yUp >= border && d[(size_t)yUp * w + xl] == 1u) {
                if (!lastXUp) { stack.push_back({xl, yUp}); lastXUp = true; }
            } else {
                const int xp = xl, yp = yUp;
                const size_t rp = (size_t)yp * w + xp;
                if (yp > border + 1 && d[rp] == 0u) {
                    min_x = std::min(min_x, xp); max_x = std::max(max_x, xp);
                    min_y = std::min(min_y, yp); max_y = std::max(max_y, yp);
                    d[rp] = (uint32_t)id | kSegIdMask;
                }
                lastXUp = false;
            }
            if (yDown < h - border && d[(size_t)yDown * w + xl] == 1u) {
                if (!lastXDown) { stack.push_back({xl, yDown}); lastXDown = true; }
            } else {
                const int xp = xl, yp = yDown;
                const size_t rp = (size_t)yp * w + xp;
                if (yp < h - border - 2 && d[rp] == 0u) {
                    min_x = std::min(min_x, xp); max_x = std::max(max_x, xp);
                    min_y = std::min(min_y, yp); max_y = std::max(max_y, yp);
                    d[rp] = (uint32_t)id | kSegIdMask;
                }
                lastXDown = false;
            }
            xl--;
        }
        d[(size_t)y * w + x] = (uint32_t)id;
        // Left border
        {
            const int xp = xl, yp = y;
            const size_t rp = (size_t)yp * w + xp;
            if (xp > border + 1 && d[rp] == 0u) {
                min_x = std::min(min_x, xp); max_x = std::max(max_x, xp);
                min_y = std::min(min_y, yp); max_y = std::max(max_y, yp);
                d[rp] = (uint32_t)id | kSegIdMask;
            }
        }
    }

    const bool success = cnt > 3;
    if (!success) {
        // Revert: too small. Restore data and clear border marks within the bbox.
        for (int row = min_y; row <= max_y; ++row) {
            for (int col = min_x; col <= max_x; ++col) {
                const size_t loc = (size_t)w * row + col;
                if (d[loc] == (uint32_t)id) d[loc] = 1u;
                else if (d[loc] == ((uint32_t)id | kSegIdMask)) d[loc] = 0u;
            }
        }
    } else {
        seg.size[id] = cnt;
        seg.xmin[id] = min_x; seg.xmax[id] = max_x;
        seg.ymin[id] = min_y; seg.ymax[id] = max_y;
        seg.nr += 1;
        clearSegmentSlot(seg, id + 1);
    }
    return success;
}

} // namespace

void segmentizePlane(Segmentation& seg) {
    const int width = seg.width;
    const int height = seg.height;
    const int border = seg.border;
    std::vector<Pos> stack;
    stack.reserve((size_t)width * height / 32 + 16);
    int id = 2;
    for (int row = border; row < height - border; ++row) {
        for (int col = border; col < width - border; ++col) {
            if (id >= seg.slots - 2) return;
            if (seg.data[(size_t)width * row + col] == 1u) {
                if (floodfillSegmentize(row, col, seg, width, height, id, stack))
                    id++;
            }
        }
    }
}

void segmentsCombine(Segmentation& seg, int radius) {
    uint32_t* img = seg.data.data();
    const int width = seg.width;
    const int height = seg.height;
    const int border = seg.border;
    intimageBorderfill(img, width, height, 0, border);
    dilating(img, seg.tmp.data(), width, height, border, radius);
    if (radius > 3) {
        intimageBorderfill(seg.tmp.data(), width, height, 1, border);
        eroding(seg.tmp.data(), img, width, height, border, radius - 3);
    } else {
        std::memcpy(img, seg.tmp.data(), (size_t)width * height * sizeof(uint32_t));
    }
    intimageBorderfill(img, width, height, 0, border);
}

bool segmentationInit(Segmentation& seg, int width, int height, int border, int slots) {
    seg.nr = 2;
    seg.border = border;
    seg.slots = std::max(256, std::min(slots, (int)kSegIdMask - 2));
    seg.width = width;
    seg.height = height;
    const size_t planeSize = (size_t)width * height;
    seg.data.assign(planeSize, 0u);
    seg.tmp.assign(planeSize, 0u);
    seg.size.assign(seg.slots, 0);
    seg.xmin.assign(seg.slots, 0);
    seg.xmax.assign(seg.slots, 0);
    seg.ymin.assign(seg.slots, 0);
    seg.ymax.assign(seg.slots, 0);
    seg.val1.assign(seg.slots, 0.0f);
    seg.val2.assign(seg.slots, 0.0f);
    clearSegmentSlot(seg, 0);
    clearSegmentSlot(seg, 1);
    return false; // success (darktable convention: FALSE = ok)
}

} // namespace rawalchemy
