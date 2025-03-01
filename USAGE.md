# Installation and Usage

## Dependencies
Some dependencies are installed automatically by CMake via `FetchContent` (e.g. Catch2) but you may still have to install Eigen and autodiff separately in order to build the library.

## Configuration, Building, Testing, and Installing
```bash
  cmake -S . -B build
  cmake --build build
  cmake --build build --target test
  sudo cmake --build build --config Release --target install
```

## Running Examples
After installing the library:
```bash
  cd example/quadrotor_hover
  mkdir build && cd build
  cmake ..
  make
  make run
```
