#pragma once

#include <string>

#include "matio.h"

/*
 * Note that this implementation is focused only on writing MATLAB
 * files.
 */

class MAT_File {
public:
    MAT_File(const char* mat_fname, const char* hdr_str = NULL);
    ~MAT_File();

    void append(const char* name, double *v, size_t n, int opt = 0, enum matio_compression compress = MAT_COMPRESSION_ZLIB);
    void append(const char* name, float *v, size_t n, int opt = 0, enum matio_compression compress = MAT_COMPRESSION_ZLIB);
    void append(const char* name, int *v, size_t n, int opt = 0, enum matio_compression compress = MAT_COMPRESSION_ZLIB);
    void append(const char* name, size_t *v, size_t n, int opt = 0, enum matio_compression compress = MAT_COMPRESSION_ZLIB);
    void append(const char* name, char *v, size_t n, int opt = 0, enum matio_compression compress = MAT_COMPRESSION_ZLIB);

    mat_t *matfp;
};
