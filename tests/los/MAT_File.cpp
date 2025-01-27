#include <cassert>

#include "MAT_File.hpp"


MAT_File::MAT_File(const char* mat_fname, const char* hdr_str) {
    matfp = Mat_CreateVer(mat_fname, hdr_str, MAT_FT_MAT73);
    assert(matfp);
}


MAT_File::~MAT_File() {
    assert(Mat_Close(matfp) == 0);
}


void
MAT_File::append(const char* name, double *v, size_t n, int opt, enum matio_compression compress) {
    assert(name);
    assert(v);

    size_t dims[2] = {n, 1};
    matvar_t *matvar = Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, v, opt);
    assert(matvar);
    assert(Mat_VarWriteAppend(matfp, matvar, MAT_COMPRESSION_ZLIB, 1) == 0);
    Mat_VarFree(matvar);
}


void
MAT_File::append(const char* name, float *v, size_t n, int opt, enum matio_compression compress) {
    assert(name);
    assert(v);

    size_t dims[2] = {n, 1};
    matvar_t *matvar = Mat_VarCreate(name, MAT_C_SINGLE, MAT_T_SINGLE, 2, dims, v, opt);
    assert(matvar);
    assert(Mat_VarWriteAppend(matfp, matvar, MAT_COMPRESSION_ZLIB, 1) == 0);
    Mat_VarFree(matvar);
}


void
MAT_File::append(const char* name, int *v, size_t n, int opt, enum matio_compression compress) {
    assert(name);
    assert(v);

    size_t dims[2] = {n, 1};
    matvar_t *matvar;

    assert(sizeof(int) == 4);
    matvar = Mat_VarCreate(name, MAT_C_UINT32, MAT_T_UINT32, 2, dims, v, opt);
    assert(matvar);
    assert(Mat_VarWriteAppend(matfp, matvar, MAT_COMPRESSION_ZLIB, 1) == 0);
    Mat_VarFree(matvar);
}


void
MAT_File::append(const char* name, char *v, size_t n, int opt, enum matio_compression compress) {
    assert(name);
    assert(v);

    size_t dims[2] = {n, 1};
    matvar_t *matvar;

    matvar = Mat_VarCreate(name, MAT_C_CHAR, MAT_T_UTF8, 2, dims, v, opt);
    assert(matvar);
    assert(Mat_VarWriteAppend(matfp, matvar, MAT_COMPRESSION_ZLIB, 1) == 0);
    Mat_VarFree(matvar);
}


void
MAT_File::append(const char* name, size_t *v, size_t n, int opt, enum matio_compression compress) {
    assert(name);
    assert(v);

    size_t dims[2] = {n, 1};
    matvar_t *matvar;

    if (sizeof(size_t) == 4) {
        matvar = Mat_VarCreate(name, MAT_C_UINT32, MAT_T_UINT32, 2, dims, v, opt);
    }
    else if (sizeof(size_t) == 8) {
        matvar = Mat_VarCreate(name, MAT_C_UINT64, MAT_T_UINT64, 2, dims, v, opt);
    }
    else {
        assert(false);
    }
    assert(matvar);
    assert(Mat_VarWriteAppend(matfp, matvar, MAT_COMPRESSION_ZLIB, 1) == 0);
    Mat_VarFree(matvar);
}
