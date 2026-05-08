/**
 * @file lut_applier.cpp
 * @brief .cube LUT parser + tetrahedral interpolation.
 *
 * Direct C++ port of the Python project's:
 *   - colour.read_LUT() — .cube file parsing
 *   - utils.apply_lut_inplace() — Numba-accelerated tetrahedral interpolation
 *
 * The 6-case tetrahedral decomposition is identical to the Python code,
 * preserving bit-exact behavior for cross-validation.
 */

#include "lut_applier.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {

// ============================================================
//  .cube File Parser
// ============================================================

LUT3D loadCubeLUT(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("[LUT] Cannot open file: " + path);
    }

    LUT3D lut;
    std::string line;
    int expectedEntries = 0;
    int loadedEntries = 0;

    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        // Skip comments
        if (line[0] == '#') continue;

        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        // Parse header keywords
        if (line.compare(0, 11, "LUT_3D_SIZE") == 0) {
            lut.size = std::stoi(line.substr(11));
            expectedEntries = lut.size * lut.size * lut.size;
            lut.table.resize(expectedEntries * 3);
            continue;
        }
        if (line.compare(0, 10, "DOMAIN_MIN") == 0) {
            std::istringstream iss(line.substr(10));
            iss >> lut.domainMin[0] >> lut.domainMin[1] >> lut.domainMin[2];
            continue;
        }
        if (line.compare(0, 10, "DOMAIN_MAX") == 0) {
            std::istringstream iss(line.substr(10));
            iss >> lut.domainMax[0] >> lut.domainMax[1] >> lut.domainMax[2];
            continue;
        }
        // Skip other header keywords (TITLE, etc.)
        if (line.find_first_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ_") == 0) continue;

        // Parse data line: "R G B"
        if (lut.size == 0) {
            throw std::runtime_error("[LUT] LUT_3D_SIZE not found before data");
        }

        float r, g, b;
        std::istringstream iss(line);
        if (!(iss >> r >> g >> b)) continue;

        if (loadedEntries < expectedEntries) {
            lut.table[loadedEntries * 3 + 0] = r;
            lut.table[loadedEntries * 3 + 1] = g;
            lut.table[loadedEntries * 3 + 2] = b;
            loadedEntries++;
        }
    }

    if (loadedEntries != expectedEntries) {
        throw std::runtime_error(
            "[LUT] Expected " + std::to_string(expectedEntries) +
            " entries, got " + std::to_string(loadedEntries));
    }

    printf("[LUT] Loaded %dx%dx%d cube (domain [%.1f,%.1f,%.1f]-[%.1f,%.1f,%.1f])\n",
           lut.size, lut.size, lut.size,
           lut.domainMin[0], lut.domainMin[1], lut.domainMin[2],
           lut.domainMax[0], lut.domainMax[1], lut.domainMax[2]);

    return lut;
}

// ============================================================
//  Tetrahedral Interpolation (Direct port of Python Numba code)
// ============================================================

void applyLUT3D(ImageBuffer& img, const LUT3D& lut) {
    const int size = lut.size;
    const int sizeM1 = size - 1;
    const float sizeF = static_cast<float>(sizeM1);

    // Scale factors: domain -> LUT index
    const float scaleR = sizeF / (lut.domainMax[0] - lut.domainMin[0]);
    const float scaleG = sizeF / (lut.domainMax[1] - lut.domainMin[1]);
    const float scaleB = sizeF / (lut.domainMax[2] - lut.domainMin[2]);
    const float minR = lut.domainMin[0];
    const float minG = lut.domainMin[1];
    const float minB = lut.domainMin[2];

    const float* table = lut.table.data();
    const size_t nPixels = img.pixelCount();
    float* data = img.ptr();

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static, 8192)
    #endif
    for (size_t i = 0; i < nPixels; i++) {
        float* p = data + i * 3;
        float inR = p[0], inG = p[1], inB = p[2];

        // A. Normalize to LUT coordinates
        float rawIdxR = (inR - minR) * scaleR;
        float rawIdxG = (inG - minG) * scaleG;
        float rawIdxB = (inB - minB) * scaleB;

        // Clamp
        float idxR = std::max(0.0f, std::min(rawIdxR, sizeF));
        float idxG = std::max(0.0f, std::min(rawIdxG, sizeF));
        float idxB = std::max(0.0f, std::min(rawIdxB, sizeF));

        // B. Integer coords + fractional parts
        int x0 = static_cast<int>(idxR);
        int y0 = static_cast<int>(idxG);
        int z0 = static_cast<int>(idxB);

        int x1 = (x0 < sizeM1) ? x0 + 1 : x0;
        int y1 = (y0 < sizeM1) ? y0 + 1 : y0;
        int z1 = (z0 < sizeM1) ? z0 + 1 : z0;

        float dx = idxR - x0;
        float dy = idxG - y0;
        float dz = idxB - z0;

        // C. Tetrahedral interpolation — 6 cases
        // Table layout: table[(x*size + y)*size + z) * 3 + c]
        // Same as lut_table[x, y, z, c] in NumPy

        float rVal = 0.0f, gVal = 0.0f, bVal = 0.0f;

        // Helper: offset for (x,y,z) -> 3-channel index
        #define TBL(r, g, b, c) table[((r) + (g)*size + (b)*size*size) * 3 + (c)]

        if (dx >= dy) {
            if (dy >= dz) {
                // Case 1: dx >= dy >= dz
                float w0 = 1.0f - dx;
                float w1 = dx - dy;
                float w2 = dy - dz;
                rVal = TBL(x0,y0,z0,0)*w0 + TBL(x1,y0,z0,0)*w1 + TBL(x1,y1,z0,0)*w2 + TBL(x1,y1,z1,0)*dz;
                gVal = TBL(x0,y0,z0,1)*w0 + TBL(x1,y0,z0,1)*w1 + TBL(x1,y1,z0,1)*w2 + TBL(x1,y1,z1,1)*dz;
                bVal = TBL(x0,y0,z0,2)*w0 + TBL(x1,y0,z0,2)*w1 + TBL(x1,y1,z0,2)*w2 + TBL(x1,y1,z1,2)*dz;
            }
            else if (dx >= dz) {
                // Case 2: dx >= dz > dy
                float w0 = 1.0f - dx;
                float w1 = dx - dz;
                float w2 = dz - dy;
                rVal = TBL(x0,y0,z0,0)*w0 + TBL(x1,y0,z0,0)*w1 + TBL(x1,y0,z1,0)*w2 + TBL(x1,y1,z1,0)*dy;
                gVal = TBL(x0,y0,z0,1)*w0 + TBL(x1,y0,z0,1)*w1 + TBL(x1,y0,z1,1)*w2 + TBL(x1,y1,z1,1)*dy;
                bVal = TBL(x0,y0,z0,2)*w0 + TBL(x1,y0,z0,2)*w1 + TBL(x1,y0,z1,2)*w2 + TBL(x1,y1,z1,2)*dy;
            }
            else {
                // Case 3: dz > dx >= dy
                float w0 = 1.0f - dz;
                float w1 = dz - dx;
                float w2 = dx - dy;
                rVal = TBL(x0,y0,z0,0)*w0 + TBL(x0,y0,z1,0)*w1 + TBL(x1,y0,z1,0)*w2 + TBL(x1,y1,z1,0)*dy;
                gVal = TBL(x0,y0,z0,1)*w0 + TBL(x0,y0,z1,1)*w1 + TBL(x1,y0,z1,1)*w2 + TBL(x1,y1,z1,1)*dy;
                bVal = TBL(x0,y0,z0,2)*w0 + TBL(x0,y0,z1,2)*w1 + TBL(x1,y0,z1,2)*w2 + TBL(x1,y1,z1,2)*dy;
            }
        }
        else { // dy > dx
            if (dz >= dy) {
                // Case 6: dz > dy > dx
                float w0 = 1.0f - dz;
                float w1 = dz - dy;
                float w2 = dy - dx;
                rVal = TBL(x0,y0,z0,0)*w0 + TBL(x0,y0,z1,0)*w1 + TBL(x0,y1,z1,0)*w2 + TBL(x1,y1,z1,0)*dx;
                gVal = TBL(x0,y0,z0,1)*w0 + TBL(x0,y0,z1,1)*w1 + TBL(x0,y1,z1,1)*w2 + TBL(x1,y1,z1,1)*dx;
                bVal = TBL(x0,y0,z0,2)*w0 + TBL(x0,y0,z1,2)*w1 + TBL(x0,y1,z1,2)*w2 + TBL(x1,y1,z1,2)*dx;
            }
            else if (dz >= dx) {
                // Case 5: dy >= dz > dx
                float w0 = 1.0f - dy;
                float w1 = dy - dz;
                float w2 = dz - dx;
                rVal = TBL(x0,y0,z0,0)*w0 + TBL(x0,y1,z0,0)*w1 + TBL(x0,y1,z1,0)*w2 + TBL(x1,y1,z1,0)*dx;
                gVal = TBL(x0,y0,z0,1)*w0 + TBL(x0,y1,z0,1)*w1 + TBL(x0,y1,z1,1)*w2 + TBL(x1,y1,z1,1)*dx;
                bVal = TBL(x0,y0,z0,2)*w0 + TBL(x0,y1,z0,2)*w1 + TBL(x0,y1,z1,2)*w2 + TBL(x1,y1,z1,2)*dx;
            }
            else {
                // Case 4: dy > dx >= dz
                float w0 = 1.0f - dy;
                float w1 = dy - dx;
                float w2 = dx - dz;
                rVal = TBL(x0,y0,z0,0)*w0 + TBL(x0,y1,z0,0)*w1 + TBL(x1,y1,z0,0)*w2 + TBL(x1,y1,z1,0)*dz;
                gVal = TBL(x0,y0,z0,1)*w0 + TBL(x0,y1,z0,1)*w1 + TBL(x1,y1,z0,1)*w2 + TBL(x1,y1,z1,1)*dz;
                bVal = TBL(x0,y0,z0,2)*w0 + TBL(x0,y1,z0,2)*w1 + TBL(x1,y1,z0,2)*w2 + TBL(x1,y1,z1,2)*dz;
            }
        }

        #undef TBL

        p[0] = rVal;
        p[1] = gVal;
        p[2] = bVal;
    }
}

} // namespace rawalchemy
