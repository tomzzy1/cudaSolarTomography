cd build
cmake -DCUDA_TOOLKIT_VERSION=11 -DCMAKE_PREFIX_PATH=$(python -m pybind11 --cmakedir) ..
make py_cuda_solartomography
cp *.so $(python -c "import site; print(site.getsitepackages()[0])")