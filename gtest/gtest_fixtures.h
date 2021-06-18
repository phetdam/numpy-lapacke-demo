/**
 * @file gtest_fixtures.h
 * @brief Header file containing Google Test fixtures.
 */

#ifndef GTEST_FIXTURES_H
#define GTEST_FIXTURES_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#endif /* PY_SSIZE_T_CLEAN */

#include <cstdio>
#include <cstdlib>

#include <gtest/gtest.h>

using namespace std;

namespace gtest_fixtures {

// fixture for tests that require the Python C API, i.e. embed interpreter
class PyCTest: public testing::Test {
  protected:
  // calls Py_Initialize to start the (debug) Python interpreter
  void SetUp() override {
    Py_Initialize();
    // if initialization fails, print error and fatally abort
    if (!Py_IsInitialized()) {
      fprintf(
        stderr, "%s: fatal: Python interpreter not initialized\n", __func__
      );
      abort();
    }
  }
  // finalizes the interpreter
  void TearDown() override {
    // print to stderr if there's an issue but don't stop
    if (Py_FinalizeEx() < 0) {
      // pointer to TestInfo instance
      const testing::TestInfo * const test_info =
        testing::UnitTest::GetInstance()->current_test_info();
      // if NULL, there is no test running. abort right away
      if (test_info == NULL) {
        fprintf(stderr, "%s: fatal: no test running\n", __func__);
        abort();
      }
      // otherwise non-fatal, we have the name of the suite and the test name
      fprintf(
        stderr, "%s::%s: Py_FinalizeEx error\n",
        test_info->test_suite_name(), test_info->name()
      );
    }
  }
};
} /* gtest_fixtures */

#endif /* GTEST_FIXTURES_H */