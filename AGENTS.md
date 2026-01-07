# Repository Guidelines

## Project Structure & Module Organization

- `src/polyfem/`: core C++ library (assembler, basis, mesh, solver, IO, utils).
- `app/`: optional GUI application (`polyfem_app`).
- `tests/`: Catch2 tests (`test_*.cpp`) built as `unit_tests`.
- `json-specs/`: JSON schemas for inputs/options/materials.
- `data/`, `pref-data/`, `polyspline-data/`, `diff-data/`: example datasets and reference inputs.
- `cmake/`: CMake modules/recipes; `patched/`: vendored third-party dependencies/patches.

## Build, Test, and Development Commands

- Configure + build (Makefiles):
  ```bash
  mkdir -p build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j
  ```
- Configure via presets (Ninja): `cmake --preset release` then `cmake --build build/release -j`.
- Enable GUI app: add `-DPOLYFEM_WITH_APP=ON` at configure time.
- Run (from your build dir): `./PolyFEM_bin` (CLI) and `./polyfem_app` (GUI, if enabled).
- Run tests: `ctest --test-dir build/release --output-on-failure` (or `./build/release/tests/unit_tests`).
- Build docs (optional): configure with `-DPOLYFEM_BUILD_DOCS=ON`; `cmake --build ...` runs Doxygen if installed.

## Coding Style & Naming Conventions

- C++ formatting: use `clang-format` with the repo’s `.clang-format` (4-space indentation; braces on new lines; no column limit).
- Naming patterns: `.cpp/.hpp` for sources/headers; new tests should follow `tests/test_<area>.cpp`.
- Keep changes focused; avoid reformatting unrelated code.

## Testing Guidelines

- Framework: Catch2 integrated via CTest (`catch_discover_tests`).
- Add new test files to `tests/CMakeLists.txt` (`test_sources`) so they build on CI.
- Prefer deterministic tests and small inputs checked into `tests/` or `data/`.

## Commit & Pull Request Guidelines

- Commits: short, imperative subjects matching existing history (e.g., “fix …”, “add …”, “update …”); put rationale/notes in the body.
- PRs: include a clear description, verification steps (commands), and ensure `ctest` passes on at least one config (`DebugNoSymbols` or `Release`). Add screenshots for GUI changes and link related issues/PRs.

## Configuration Tips

- For machine-specific defaults, copy `PolyFEMOptions.cmake.sample` to `PolyFEMOptions.cmake` (kept local).
- For editor integration, enable `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` (presets already do this).
