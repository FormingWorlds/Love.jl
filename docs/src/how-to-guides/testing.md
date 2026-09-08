
# Testing and profiling

This page covers the practical aspects of testing Obliqua: running tests, writing tests, checking coverage, and working with CI.

---

### Quick start

To run the tests, navigate to the root of the project and execute:
```bash
julia --project test/runtests.jl
```

The tests will run and print information on whether they passed or failed. The tests are located in the `test/` folder, and are written using the `Test` standard library. There are several test files, each corresponding to a different module in the code. The main test file is `runtests.jl`, which runs all the tests in the other files. The test file tree is shown below:

```bash
Obliqua/
├── 📂 test/
│   ├── 📄 runcoverage.jl
│   ├── 📄 runprofiler.jl
│   ├── 📄 runtests.jl
│   ├── 📄 test_common.jl
│   ├── 📄 test_constants.jl
│   ├── 📄 test_fluid0d.jl
│   ├── 📄 test_obliqua.jl
│   ├── 📄 test_setup.jl
│   ├── 📄 test_solid0d.jl
│   ├── 📄 test_solid1d_equil_relax.jl
│   ├── 📄 test_solid1d_mush_relax.jl
│   ├── 📄 test_solid1d_mush.jl
│   ├── 📄 test_solid1d_relax.jl
│   ├── 📄 test_solid1d.jl
│   └── 📝 test.toml
```

Note that the tests are designed to mirror the source tree structure, with each test file corresponding to a module in the `src/` folder. This makes it easy to locate and run tests for specific modules.

---

### Coverage

To check the code coverage of the tests, you can use the `Coverage` package in Julia. Run the following command:
```bash
julia --project test/runcoverage.jl
```

We target a minimum of 80% coverage for the codebase. The coverage report will be generated in the `test/coverage/` folder, and can be viewed in a web browser by opening the `index.html` file.

---

### Profiling

To profile the performance of the code, you can use the `Profile` package in Julia. Run the following command:
```bash
julia --project test/runprofiler.jl
```

---

### CI/CD pipeline

Pull requests are automatically tested using GitHub Actions. The workflow is defined in the `.github/workflows/ci.yml` file. The basic structure follows

1. **Check out repository**: Checks out the codebase using `actions/checkout@v4`.
2. **Set up Julia**: Initializes the Julia runtime environment (version `1.11`) via `julia-actions/setup-julia@v2`.
3. **Run test suite**: Instantiates the project environment and executes unit and integration tests with code coverage enabled (`Pkg.test(coverage=true)`).
4. **Process coverage**: Runs `test/runcoverage.jl` to compute coverage statistics, exports the total percentage into the environment, and appends the markdown coverage report to the job summary.
5. **Upload to Codecov**: Submits the generated coverage reports to Codecov (`FormingWorlds/Obliqua`) using `codecov/codecov-action@v5`.
6. **Publish test badges**: Updates the remote GitHub Gist containing test badge JSON files (`tests-unit.json`, `tests-integration.json`, `tests-total.json`) via the GitHub API (runs only on `push` events to `main`).