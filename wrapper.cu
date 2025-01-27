#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include "build_A_matrix_dyn.cuh"
#include "type.hpp"
#include "utility.cuh"
#include "utility.hpp"

#if BUILD_SOLVER
#include "optimizer.hpp"
#endif

namespace py = pybind11;
using namespace pybind11::literals;
using real = cudaSolarTomography::real;

/*
Helper function for building C++ sparse matrix from numpy arrays
*/
template<typename T>
cudaSolarTomography::SparseMatrix<real> from_np_arrays(py::array_t<int> row_ptr, py::array_t<int> col_idx, py::array_t<T> val)
{
    cudaSolarTomography::SparseMatrix<T> matrix;
    matrix.row_ptr_h = static_cast<int *>(row_ptr.request().ptr);
    matrix.col_idx_h = static_cast<int *>(col_idx.request().ptr);
    matrix.val_h = static_cast<T *>(val.request().ptr);
    matrix.n_rows = row_ptr.size() - 1;
    matrix.nnz = val.size();
    return matrix;
}

/*
Helper function for building C++ sparse matrix with image from numpy arrays
*/
template<typename T>
cudaSolarTomography::SparseMatrixAndImage<T> from_np_arrays(py::array_t<int> row_ptr, py::array_t<int> col_idx, py::array_t<T> val, py::array_t<T> y)
{
    cudaSolarTomography::SparseMatrixAndImage<T> matrix(from_np_arrays(row_ptr, col_idx, val));
    matrix.y_h = static_cast<T *>(y.request().ptr);
    return matrix;
}

/*
Helper function for extrating componenets from C++ spares array and converting to numpy arrays 
(can be used to build scipy sparse matrix format in Python code)
*/

template<typename T>
py::tuple to_np_arrays(const cudaSolarTomography::SparseMatrix<T> &matrix)
{
    py::array_t<int> row_ptr(matrix.n_rows + 1, matrix.row_ptr_h);
    py::array_t<int> col_idx(matrix.nnz, matrix.col_idx_h);
    py::array_t<T> val(matrix.nnz, matrix.val_h);
    return py::make_tuple(row_ptr, col_idx, val);
}

/*
Same as the other overload, but also include the y vector
*/

template<typename T>
py::tuple to_np_arrays(const cudaSolarTomography::SparseMatrixAndImage<T> &matrix)
{
    py::array_t<int> row_ptr(matrix.n_rows + 1, matrix.row_ptr_h);
    py::array_t<int> col_idx(matrix.nnz, matrix.col_idx_h);
    py::array_t<T> val(matrix.nnz, matrix.val_h);
    py::array_t<T> y(matrix.n_rows, matrix.y_h);
    return py::make_tuple(row_ptr, col_idx, val, y);
}

/*
Build A matrix from FITS files (filenames)
*/

py::tuple build_A_matrix(
    py::list filenames,
    cudaSolarTomography::GridParameters *grid_parameters,
    cudaSolarTomography::InstrParameters *instr_parameters)
{
    // std::cout << grid_parameters->n_rad_bins << '\n';
    // std::cout << instr_parameters->instr_r_max << '\n';
    auto matrix = cudaSolarTomography::build_A_matrix_dyn(filenames.cast<std::vector<std::string>>(), *grid_parameters, *instr_parameters);
    return to_np_arrays(matrix);
}

/*
Build A matrix from FITS files and synthetic images
The filenames provides the file names and corresponding FITS headers, 
The projection_dir should contains the synthetic images that have the same file names as the filenames
*/

py::tuple build_A_matrix_with_projection(
    py::list filenames, py::str projection_dir,
    cudaSolarTomography::GridParameters *grid_parameters,
    cudaSolarTomography::InstrParameters *instr_parameters)
{
    auto matrix = cudaSolarTomography::build_A_matrix_with_projection_dyn(filenames.cast<std::vector<std::string>>(), *grid_parameters, *instr_parameters, projection_dir.cast<std::string>());
    return to_np_arrays(matrix);
}

/*
Similar to the build_A_matrix, but provide extra mapping from y vector to images, to help
generate synthetic images from (A dot x)
*/

py::tuple build_A_matrix_with_mapping_to_y(
    py::list filenames,
    cudaSolarTomography::GridParameters *grid_parameters,
    cudaSolarTomography::InstrParameters *instr_parameters)
{
    // std::cout << grid_parameters->n_rad_bins << '\n';
    // std::cout << instr_parameters->instr_r_max << '\n';
    auto n_files = filenames.size();
    auto params = cudaSolarTomography::get_all_parameters_from_files(filenames.cast<std::vector<std::string>>());
    std::vector<int> y_mapping(n_files * instr_parameters->y_size + 1, 0);
    auto matrix = cudaSolarTomography::build_A_matrix_from_params_dyn(params, *grid_parameters, *instr_parameters, y_mapping);
    py::array_t<int> row_ptr(matrix.n_rows + 1, matrix.row_ptr_h);
    py::array_t<int> col_idx(matrix.nnz, matrix.col_idx_h);
    py::array_t<float> val(matrix.nnz, matrix.val_h);
    py::array_t<int> y_mapping_np(n_files * instr_parameters->y_size + 1, y_mapping.data());
    return py::make_tuple(row_ptr, col_idx, val, y_mapping_np);
}

/*
Build A matrix with n virtual viewpoints (deterimned by the first and last real viewpoints in filenames)
The viewpoints should be in ascending order in time
*/

py::tuple build_A_matrix_with_virtual_viewpoints(
    py::list filenames, py::int_ n_viewpoints,
    cudaSolarTomography::GridParameters *grid_parameters,
    cudaSolarTomography::InstrParameters *instr_parameters)
{
    auto params = cudaSolarTomography::get_all_parameters_from_files_with_virtual_viewpoints(filenames.cast<std::vector<std::string>>(), n_viewpoints);
    std::vector<int> y_mapping(n_viewpoints * instr_parameters->y_size + 1, 0);
    auto matrix = cudaSolarTomography::build_A_matrix_from_params_dyn(params, *grid_parameters, *instr_parameters, y_mapping);
    py::array_t<int> row_ptr(matrix.n_rows + 1, matrix.row_ptr_h);
    py::array_t<int> col_idx(matrix.nnz, matrix.col_idx_h);
    py::array_t<float> val(matrix.nnz, matrix.val_h);
    py::array_t<int> y_mapping_np(n_viewpoints * instr_parameters->y_size + 1, y_mapping.data());
    auto res = py::make_tuple(row_ptr, col_idx, val, y_mapping_np);
    return res;
}

/*
Build A matrix with n virtual viewpoints (deterimned by the first and last real viewpoints in filenames, but view angle can be arbitrary)
The viewpoints should be in ascending order in time
*/

py::tuple build_A_matrix_with_virtual_viewpoints(
    py::list filenames, py::int_ n_viewpoints, py::float_ degree,
    cudaSolarTomography::GridParameters *grid_parameters,
    cudaSolarTomography::InstrParameters *instr_parameters)
{
    auto params = cudaSolarTomography::get_all_parameters_from_files_with_virtual_viewpoints(filenames.cast<std::vector<std::string>>(), n_viewpoints, degree);
    std::vector<int> y_mapping(n_viewpoints * instr_parameters->y_size + 1, 0);
    auto matrix = cudaSolarTomography::build_A_matrix_from_params_dyn(params, *grid_parameters, *instr_parameters, y_mapping);
    py::array_t<int> row_ptr(matrix.n_rows + 1, matrix.row_ptr_h);
    py::array_t<int> col_idx(matrix.nnz, matrix.col_idx_h);
    py::array_t<float> val(matrix.nnz, matrix.val_h);
    py::array_t<int> y_mapping_np(n_viewpoints * instr_parameters->y_size + 1, y_mapping.data());
    auto res = py::make_tuple(row_ptr, col_idx, val, y_mapping_np);
    return res;
}

/*
The corresponding version of build_A_matrix_with_projection for build_A_matrix_with_virtual_viewpoints (fixed view angle)
*/

py::tuple build_A_matrix_with_projection(
    py::list filenames, py::str projection_dir,
    cudaSolarTomography::GridParameters *grid_parameters,
    cudaSolarTomography::InstrParameters *instr_parameters, py::list projection_filenames, py::int_ n_viewpoints)
{
    auto matrix = cudaSolarTomography::build_A_matrix_with_projection_dyn(filenames.cast<std::vector<std::string>>(), *grid_parameters, *instr_parameters, projection_dir.cast<std::string>(), projection_filenames.cast<std::vector<std::string>>(), n_viewpoints);
    return to_np_arrays(matrix);
}

/*
The corresponding version of build_A_matrix_with_projection for build_A_matrix_with_virtual_viewpoints (arbitrary view angle)
*/

py::tuple build_A_matrix_with_projection(
    py::list filenames, py::str projection_dir,
    cudaSolarTomography::GridParameters *grid_parameters,
    cudaSolarTomography::InstrParameters *instr_parameters, py::list projection_filenames, py::int_ n_viewpoints, py::float_ degree)
{
    auto matrix = cudaSolarTomography::build_A_matrix_with_projection_dyn(filenames.cast<std::vector<std::string>>(), *grid_parameters, *instr_parameters, projection_dir.cast<std::string>(), projection_filenames.cast<std::vector<std::string>>(), n_viewpoints, degree);
    return to_np_arrays(matrix);
}

/*
Deprecated: This should be done directly in Python
*/

py::array_t<real> get_simulation_x(
    py::str dir,
    cudaSolarTomography::GridParameters *grid_parameters)
{
    std::fstream x_file;
    std::vector<char> x;
    if constexpr (std::is_same_v<real, float>)
    {
        if (x_file.open(dir.cast<std::string>() + "x_corhel"); !x_file.is_open())
        {
            std::cerr << "Failed to open the corhel file\n";
            std::exit(-1);
        }
    }
    else
    {
        if (x_file.open(dir.cast<std::string>() + "x_corhel_db"); !x_file.is_open())
        {
            std::cerr << "Failed to open the corhel file\n";
            std::exit(-1);
        }
    }
    x.assign(std::istreambuf_iterator<char>(x_file), std::istreambuf_iterator<char>());
    assert(x.size() == grid_parameters->n_bins * sizeof(real));
    return py::array_t<real>(x.size() / sizeof(real), reinterpret_cast<real *>(x.data()));
}



#if BUILD_SOLVER
// this method integrates the process of building and reconstructing, avoids moving data back and forth

py::array_t<real> build_and_reconstruct(
    py::list filenames,
    cudaSolarTomography::GridParameters *grid_parameters,
    cudaSolarTomography::InstrParameters *instr_parameters,
    py::array_t<int> D_row_ptr, py::array_t<int> D_col_idx, py::array_t<real> D_val,
    double lambda_tik)
{
    int col_size = grid_parameters->n_bins;

    auto A_y = cudaSolarTomography::build_A_matrix_dyn(filenames.cast<std::vector<std::string>>(), *grid_parameters, *instr_parameters);

    cudaSolarTomography::cusparseContext context;

    auto A_y_d = A_y.to_cuda<real>();
    auto y_descr = A_y_d.createDnVecDescr();
    auto A_descr = A_y_d.createSpMatDescr(col_size);

    cudaSolarTomography::SparseMatrix<real> reg_matrix = from_np_arrays(D_row_ptr, D_col_idx, D_val);
    std::cout << "n_rows " << reg_matrix.n_rows << " nnz " << reg_matrix.nnz << '\n';
    auto reg_matrix_d = reg_matrix.to_cuda<real>();
    auto D_descr = reg_matrix_d.createSpMatDescr(col_size);
    reg_matrix.release(); // avoid double free

    std::vector<real> x_result_h(col_size, 1e4);
    real *x_result_d = nullptr;
    cudaMalloc(&x_result_d, col_size * sizeof(real));
    cudaMemset(x_result_d, 0, col_size * sizeof(real));

    cusparseDnVecDescr_t x_descr;
    cusparseCreateDnVec(&x_descr, col_size, x_result_d, cudaSolarTomography::CUDA_REAL);

    optimize(&context, A_descr, x_descr, y_descr, D_descr, A_y_d.y_d, x_result_d, A_y_d.n_rows, reg_matrix.n_rows, lambda_tik, col_size);

    cudaMemcpy(x_result_h.data(), x_result_d, col_size * sizeof(real), cudaMemcpyDeviceToHost);

    cusparseDestroySpMat(A_descr);
    cusparseDestroySpMat(D_descr);
    cusparseDestroyDnVec(x_descr);
    cusparseDestroyDnVec(y_descr);

    cudaFree(x_result_d);

    return py::array_t<real>(x_result_h.size(), x_result_h.data());
}

/*
Reconstruct x with y, A, regularization matrix D, and lambda
*/
py::array_t<real> reconstruct(
    cudaSolarTomography::GridParameters *grid_parameters,
    py::array_t<int> A_row_ptr, py::array_t<int> A_col_idx, py::array_t<real> A_val, py::array_t<real> y,
    py::array_t<int> D_row_ptr, py::array_t<int> D_col_idx, py::array_t<real> D_val,
    double lambda_tik)
{
    int col_size = grid_parameters->n_bins;

    cudaSolarTomography::SparseMatrixAndImage<real> A_y = from_np_arrays(A_row_ptr, A_col_idx, A_val, y);
    std::cout << "n_rows " << A_y.n_rows << " nnz " << A_y.nnz << '\n';
    auto A_y_d = A_y.to_cuda<real>();
    auto y_descr = A_y_d.createDnVecDescr();
    auto A_descr = A_y_d.createSpMatDescr(col_size);
    A_y.release();

    cudaSolarTomography::cusparseContext context;

    cudaSolarTomography::SparseMatrix<real> reg_matrix = from_np_arrays(D_row_ptr, D_col_idx, D_val);
    std::cout << "n_rows " << reg_matrix.n_rows << " nnz " << reg_matrix.nnz << '\n';
    auto reg_matrix_d = reg_matrix.to_cuda<real>();
    auto D_descr = reg_matrix_d.createSpMatDescr(col_size);
    reg_matrix.release(); // avoid double free

    std::vector<real> x_result_h(col_size, 1e4);
    real *x_result_d = nullptr;
    cudaMalloc(&x_result_d, col_size * sizeof(real));
    cudaMemset(x_result_d, 0, col_size * sizeof(real));

    cusparseDnVecDescr_t x_descr;
    cusparseCreateDnVec(&x_descr, col_size, x_result_d, cudaSolarTomography::CUDA_REAL);

    optimize(&context, A_descr, x_descr, y_descr, D_descr, A_y_d.y_d, x_result_d, A_y_d.n_rows, reg_matrix.n_rows, lambda_tik, col_size);

    cudaMemcpy(x_result_h.data(), x_result_d, col_size * sizeof(real), cudaMemcpyDeviceToHost);

    cusparseDestroySpMat(A_descr);
    cusparseDestroySpMat(D_descr);
    cusparseDestroyDnVec(x_descr);
    cusparseDestroyDnVec(y_descr);

    cudaFree(x_result_d);

    return py::array_t<real>(x_result_h.size(), x_result_h.data());
}

/*
Corresponding to build_and_reconstruct
*/

py::array_t<real> build_and_reconstruct_with_projection(
    py::list filenames, py::str projection_dir,
    cudaSolarTomography::GridParameters *grid_parameters,
    cudaSolarTomography::InstrParameters *instr_parameters,
    py::array_t<int> D_row_ptr, py::array_t<int> D_col_idx, py::array_t<real> D_val,
    double lambda_tik)
{
    int col_size = grid_parameters->n_bins;

    auto A_y = cudaSolarTomography::build_A_matrix_with_projection_dyn(filenames.cast<std::vector<std::string>>(), *grid_parameters, *instr_parameters, projection_dir.cast<std::string>());

    cudaSolarTomography::cusparseContext context;

    auto A_y_d = A_y.to_cuda<real>();
    auto y_descr = A_y_d.createDnVecDescr();
    auto A_descr = A_y_d.createSpMatDescr(col_size);

    cudaSolarTomography::SparseMatrix<real> reg_matrix = from_np_arrays(D_row_ptr, D_col_idx, D_val);
    std::cout << "n_rows " << reg_matrix.n_rows << " nnz " << reg_matrix.nnz << '\n';
    auto reg_matrix_d = reg_matrix.to_cuda<real>();
    auto D_descr = reg_matrix_d.createSpMatDescr(col_size);
    reg_matrix.release(); // avoid double free

    std::vector<real> x_result_h(col_size, 1e4);
    real *x_result_d = nullptr;
    cudaMalloc(&x_result_d, col_size * sizeof(real));
    cudaMemset(x_result_d, 0, col_size * sizeof(real));

    cusparseDnVecDescr_t x_descr;
    cusparseCreateDnVec(&x_descr, col_size, x_result_d, cudaSolarTomography::CUDA_REAL);

    optimize(&context, A_descr, x_descr, y_descr, D_descr, A_y_d.y_d, x_result_d, A_y_d.n_rows, reg_matrix.n_rows, lambda_tik, col_size);

    cudaMemcpy(x_result_h.data(), x_result_d, col_size * sizeof(real), cudaMemcpyDeviceToHost);

    cusparseDestroySpMat(A_descr);
    cusparseDestroySpMat(D_descr);
    cusparseDestroyDnVec(x_descr);
    cusparseDestroyDnVec(y_descr);

    cudaFree(x_result_d);

    return py::array_t<real>(x_result_h.size(), x_result_h.data());
}

#endif

/*
Calculate A.T: Not necessary to be done with CUDA 
*/

py::tuple A_transpose(
    cudaSolarTomography::GridParameters *grid_parameters,
    py::array_t<int> A_row_ptr, py::array_t<int> A_col_idx, py::array_t<real> A_val)
{
    int col_size = grid_parameters->n_bins;
    std::cerr << "col_size " << col_size << '\n';
    // create cusparse matrix A
    cudaSolarTomography::SparseMatrixAndImage<real> A_y = from_np_arrays(A_row_ptr, A_col_idx, A_val);
    std::cerr << "n_rows " << A_y.n_rows << " nnz " << A_y.nnz << '\n';
    auto A_y_d = A_y.to_cuda<real>();
    auto y_descr = A_y_d.createDnVecDescr();
    auto A_descr = A_y_d.createSpMatDescr(col_size);
    A_y.release();

    cudaSolarTomography::cusparseContext context;

    // compute A transpose
    // swap nnz and n_rows for A transpoe
    cudaSolarTomography::SparseMatrixGPU<real> A_T_d(col_size, A_y.nnz);
    context.transpose(col_size, A_y_d, A_T_d, 0);
    cudaCheckError();
    auto A_T_descr = A_T_d.createSpMatDescr(A_y.n_rows);

    cusparseDestroySpMat(A_descr);
    cusparseDestroyDnVec(y_descr);
    cusparseDestroySpMat(A_T_descr);

    return to_np_arrays(A_T_d.to_host<real>());
}

/*
Calculate the A dot A.T y as a normalized input for some tasks
*/

py::array_t<real> normalized_input(
    cudaSolarTomography::GridParameters *grid_parameters,
    py::array_t<int> A_row_ptr, py::array_t<int> A_col_idx, py::array_t<real> A_val, py::array_t<real> y)
{
    int col_size = grid_parameters->n_bins;

    // create cusparse matrix A
    cudaSolarTomography::SparseMatrixAndImage<real> A_y = from_np_arrays(A_row_ptr, A_col_idx, A_val, y);
    std::cout << "n_rows " << A_y.n_rows << " nnz " << A_y.nnz << '\n';
    auto A_y_d = A_y.to_cuda<real>();
    auto y_descr = A_y_d.createDnVecDescr();
    auto A_descr = A_y_d.createSpMatDescr(col_size);
    A_y.release();

    cudaSolarTomography::cusparseContext context;

    // compute A transpose
    // swap nnz and n_rows for A transpoe
    cudaSolarTomography::SparseMatrixGPU<real> A_T_d(col_size, A_y.nnz);
    context.transpose(col_size, A_y_d, A_T_d, 0);
    cudaCheckError();
    auto A_T_descr = A_T_d.createSpMatDescr(A_y.n_rows);

    // compute A.T * A
    cudaSolarTomography::SparseMatrixGPU<real> A_T_A(col_size, 0);
    cusparseSpMatDescr_t A_T_A_descr;
    // create empty descriptor for matC
    CHECK_CUSPARSE(cusparseCreateCsr(&A_T_A_descr, col_size, col_size, 0,
                                     A_T_A.row_ptr_d, nullptr, nullptr,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, cudaSolarTomography::CUDA_REAL));
    context.A_T_mult_A(A_T_descr, A_descr, A_T_A_descr, 1, 0, col_size, A_T_A);
    cudaCheckError();
    // compute the trace of A.T * A
    real *trace_d = nullptr;
    cudaMalloc(&trace_d, sizeof(real));
    cudaMemset(trace_d, 0, sizeof(real));
    cudaSolarTomography::computeTrace<<<512, 512>>>(col_size, A_T_A.row_ptr_d, A_T_A.col_idx_d, A_T_A.val_d, trace_d);
    cudaCheckError();
    real trace_h = 0;
    cudaMemcpy(&trace_h, trace_d, sizeof(real), cudaMemcpyDeviceToHost);
    std::cerr << "Trace " << trace_h << '\n';

    // compute A.T * y
    // since we have computed the A.T, non transpose operation is faster
    real *A_Ty_d = nullptr;
    cudaMalloc(&A_Ty_d, sizeof(real) * col_size);
    cusparseDnVecDescr_t A_Ty_descr;

    CHECK_CUSPARSE(cusparseCreateDnVec(&A_Ty_descr, col_size, A_Ty_d, cudaSolarTomography::CUDA_REAL));
    context.SpMV(CUSPARSE_OPERATION_NON_TRANSPOSE, A_T_descr, y_descr, A_Ty_descr, 1, 0, 1);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    CHECK_CUBLAS(cudaSolarTomography::scale(cublas_handle, col_size, 1 / trace_h, A_Ty_d));

    py::array_t<real> result(col_size);
    cudaMemcpy(result.request().ptr, A_Ty_d, sizeof(real) * col_size, cudaMemcpyDeviceToHost);

    cublasDestroy(cublas_handle);

    cusparseDestroySpMat(A_descr);
    cusparseDestroyDnVec(y_descr);
    cusparseDestroySpMat(A_T_descr);
    cusparseDestroySpMat(A_T_A_descr);
    cusparseDestroyDnVec(A_Ty_descr);
    cudaFree(trace_d);
    cudaFree(A_Ty_d);

    return result;
}

PYBIND11_MODULE(py_cuda_solartomography, m)
{
    m.doc() = "pybind11 cudaSolarTomography plugin";
    py::class_<cudaSolarTomography::GridParameters>(m, "GridParameters")
        .def(py::init<int, int, int, double, double>())
        .def_readwrite("n_rad_bins", &cudaSolarTomography::GridParameters::n_rad_bins)
        .def_readwrite("n_theta_bins", &cudaSolarTomography::GridParameters::n_theta_bins)
        .def_readwrite("n_phi_bins", &cudaSolarTomography::GridParameters::n_phi_bins)
        .def_readwrite("n_bins", &cudaSolarTomography::GridParameters::n_bins)
        .def_readwrite("AROW_SIZE", &cudaSolarTomography::GridParameters::AROW_SIZE)
        .def_readwrite("r_max", &cudaSolarTomography::GridParameters::r_max)
        .def_readwrite("r_min", &cudaSolarTomography::GridParameters::r_min)
        .def_readwrite("r_diff", &cudaSolarTomography::GridParameters::r_diff)
        .def_readwrite("r_max2", &cudaSolarTomography::GridParameters::r_max2)
        .def_readwrite("r_min2", &cudaSolarTomography::GridParameters::r_min2)
        .def_readwrite("rad_bin_size", &cudaSolarTomography::GridParameters::rad_bin_size)
        .def_readwrite("theta_bin_size", &cudaSolarTomography::GridParameters::theta_bin_size)
        .def_readwrite("phi_bin_size", &cudaSolarTomography::GridParameters::phi_bin_size);

    py::class_<cudaSolarTomography::InstrParameters>(m, "InstrParameters")
        .def(py::init<double, double, int, double, int, double>())
        .def_readwrite("instr_r_max", &cudaSolarTomography::InstrParameters::instr_r_max)
        .def_readwrite("instr_r_min", &cudaSolarTomography::InstrParameters::instr_r_min)
        .def_readwrite("image_size", &cudaSolarTomography::InstrParameters::image_size)
        .def_readwrite("pixel_size", &cudaSolarTomography::InstrParameters::pixel_size)
        .def_readwrite("binning_factor", &cudaSolarTomography::InstrParameters::binning_factor)
        .def_readwrite("scale_factor", &cudaSolarTomography::InstrParameters::scale_factor)
        .def_readwrite("bin_size", &cudaSolarTomography::InstrParameters::bin_size)
        .def_readwrite("row_size", &cudaSolarTomography::InstrParameters::row_size)
        .def_readwrite("y_size", &cudaSolarTomography::InstrParameters::y_size);

    m.def("build_A_matrix", &build_A_matrix, "A function that comptues the projection matrix");
    m.def("build_A_matrix_with_projection", static_cast<py::tuple (*)(py::list, py::str, cudaSolarTomography::GridParameters *, cudaSolarTomography::InstrParameters *)>(&build_A_matrix_with_projection), "A function that comptues the projection matrix and projected y");
    m.def("build_A_matrix_with_projection", static_cast<py::tuple (*)(py::list, py::str, cudaSolarTomography::GridParameters *, cudaSolarTomography::InstrParameters *, py::list, py::int_)>(&build_A_matrix_with_projection), "A function that comptues the projection matrix with virtual viewpoints and projected y");
    m.def("build_A_matrix_with_projection", static_cast<py::tuple (*)(py::list, py::str, cudaSolarTomography::GridParameters *, cudaSolarTomography::InstrParameters *, py::list, py::int_, py::float_)>(&build_A_matrix_with_projection), "A function that comptues the projection matrix with virtual viewpoints, specified degree and projected y");
    m.def("get_simulation_x", &get_simulation_x, "Get the ground truth x from simluation");
    m.def("build_A_matrix_with_mapping_to_y", &build_A_matrix_with_mapping_to_y, "For generating the projected images");
    m.def("build_A_matrix_with_virtual_viewpoints", static_cast<py::tuple (*)(py::list, py::int_, cudaSolarTomography::GridParameters *, cudaSolarTomography::InstrParameters *)>(&build_A_matrix_with_virtual_viewpoints), "Computes the projection matrix with virtual viewpoints");
    m.def("build_A_matrix_with_virtual_viewpoints", static_cast<py::tuple (*)(py::list, py::int_, py::float_, cudaSolarTomography::GridParameters *, cudaSolarTomography::InstrParameters *)>(&build_A_matrix_with_virtual_viewpoints), "Computes the projection matrix with virtual viewpoints and specified degree");
#if BUILD_SOLVER
    m.def("build_and_reconstruct", &build_and_reconstruct, "Perform the full reconstruction process");
    m.def("reconstruct", &reconstruct, "Reconstruct with A from python caller");
    m.def("build_and_reconstruct_with_projection", &build_and_reconstruct_with_projection, "Perform the full reconstruction process with projected images");
#endif
    m.def("normalized_input", &normalized_input, "Compute the normalized input");
    m.def("A_transpose", &A_transpose, "transpose a sparse matrix");
}