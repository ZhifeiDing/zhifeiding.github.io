---
title: gtest Introduction
tags: [test,programming]
categories: [programming]
---

# gtest's API

```cpp
// Values() allows generating tests from explicitly specified list of
// parameters.
//
// Synopsis:
// Values(T v1, T v2, ..., T vN)
//   - returns a generator producing sequences with elements v1, v2, ..., vN.
// Currently, Values() supports from 1 to 50 parameters.
//
template <typename T1>
internal::ValueArray1<T1> Values(T1 v1) {
  return internal::ValueArray1<T1>(v1);
}
```


```cpp
# define TEST_P(test_case_name, test_name) \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
      : public test_case_name { \
   public: \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {} \
    virtual void TestBody(); \
   private: \
    static int AddToRegistry() { \
      ::testing::UnitTest::GetInstance()->parameterized_test_registry(). \
          GetTestCasePatternHolder<test_case_name>(\
              #test_case_name, __FILE__, __LINE__)->AddTestPattern(\
                  #test_case_name, \
                  #test_name, \
                  new ::testing::internal::TestMetaFactory< \
                      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>()); \
      return 0; \
    } \
    static int gtest_registering_dummy_ GTEST_ATTRIBUTE_UNUSED_; \
    GTEST_DISALLOW_COPY_AND_ASSIGN_(\
        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)); \
  }; \
  int GTEST_TEST_CLASS_NAME_(test_case_name, \
                             test_name)::gtest_registering_dummy_ = \
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::AddToRegistry(); \
  void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody()

# define INSTANTIATE_TEST_CASE_P(prefix, test_case_name, generator) \
  ::testing::internal::ParamGenerator<test_case_name::ParamType> \
      gtest_##prefix##test_case_name##_EvalGenerator_() { return generator; } \
  int gtest_##prefix##test_case_name##_dummy_ = \
      ::testing::UnitTest::GetInstance()->parameterized_test_registry(). \
          GetTestCasePatternHolder<test_case_name>(\
              #test_case_name, __FILE__, __LINE__)->AddTestCaseInstantiation(\
                  #prefix, \
                  &gtest_##prefix##test_case_name##_EvalGenerator_, \
                  __FILE__, __LINE__)

}
```

```cpp
// The abstract class that all tests inherit from.
//
// In Google Test, a unit test program contains one or many TestCases, and
// each TestCase contains one or many Tests.
//
// When you define a test using the TEST macro, you don't need to
// explicitly derive from Test - the TEST macro automatically does
// this for you.
//
// The only time you derive from Test is when defining a test fixture
// to be used a TEST_F.  For example:
//
//   class FooTest : public testing::Test {
//    protected:
//     void SetUp() override { ... }
//     void TearDown() override { ... }
//     ...
//   };
//
//   TEST_F(FooTest, Bar) { ... }
//   TEST_F(FooTest, Baz) { ... }
//
// Test is not copyable.
class GTEST_API_ Test
```
```cpp
// The result of a single Test.  This includes a list of
// TestPartResults, a list of TestProperties, a count of how many
// death tests there are in the Test, and how much time it took to run
// the Test.
//
// TestResult is not copyable.
class GTEST_API_ TestResult
```

```cpp
// A test case, which consists of a vector of TestInfos.
//
// TestCase is not copyable.
class GTEST_API_ TestCase
```

```cpp
// The interface for tracing execution of tests. The methods are organized in
// the order the corresponding events are fired.
class TestEventListener {
 public:
  virtual ~TestEventListener() {}

  // Fired before any test activity starts.
  virtual void OnTestProgramStart(const UnitTest& unit_test) = 0;

  // Fired before each iteration of tests starts.  There may be more than
  // one iteration if GTEST_FLAG(repeat) is set. iteration is the iteration
  // index, starting from 0.
  virtual void OnTestIterationStart(const UnitTest& unit_test,
                                    int iteration) = 0;

  // Fired before environment set-up for each iteration of tests starts.
  virtual void OnEnvironmentsSetUpStart(const UnitTest& unit_test) = 0;

  // Fired after environment set-up for each iteration of tests ends.
  virtual void OnEnvironmentsSetUpEnd(const UnitTest& unit_test) = 0;

  // Fired before the test case starts.
  virtual void OnTestCaseStart(const TestCase& test_case) = 0;

  // Fired before the test starts.
  virtual void OnTestStart(const TestInfo& test_info) = 0;

  // Fired after a failed assertion or a SUCCEED() invocation.
  virtual void OnTestPartResult(const TestPartResult& test_part_result) = 0;

  // Fired after the test ends.
  virtual void OnTestEnd(const TestInfo& test_info) = 0;

  // Fired after the test case ends.
  virtual void OnTestCaseEnd(const TestCase& test_case) = 0;

  // Fired before environment tear-down for each iteration of tests starts.
  virtual void OnEnvironmentsTearDownStart(const UnitTest& unit_test) = 0;

  // Fired after environment tear-down for each iteration of tests ends.
  virtual void OnEnvironmentsTearDownEnd(const UnitTest& unit_test) = 0;

  // Fired after each iteration of tests finishes.
  virtual void OnTestIterationEnd(const UnitTest& unit_test,
                                  int iteration) = 0;

  // Fired after all test activities have ended.
  virtual void OnTestProgramEnd(const UnitTest& unit_test) = 0;
};
```


```cpp
// A UnitTest consists of a vector of TestCases.
//
// This is a singleton class.  The only instance of UnitTest is
// created when UnitTest::GetInstance() is first called.  This
// instance is never deleted.
//
// UnitTest is not copyable.
//
// This class is thread-safe as long as the methods are called
// according to their specification.
class GTEST_API_ UnitTest
```

```cpp
// Initializes Google Test.  This must be called before calling
// RUN_ALL_TESTS().  In particular, it parses a command line for the
// flags that Google Test recognizes.  Whenever a Google Test flag is
// seen, it is removed from argv, and *argc is decremented.
//
// No value is returned.  Instead, the Google Test flag variables are
// updated.
//
// Calling the function for the second time has no user-visible effect.
GTEST_API_ void InitGoogleTest(int* argc, char** argv);
```



# 测试实例
