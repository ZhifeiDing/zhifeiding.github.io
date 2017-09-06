---
title: gtest Introduction
tags: [test,programming]
categories: [programming]
---

# gtest's API


`gtest`里有三种使用方式，分别是 `TEST`, `TEST_F`, `TESTP`, 都是宏定义。

## `TEST`的定义

`TEST`的定义如下，

```cpp
// Defines a test.
//
// The first parameter is the name of the test case, and the second
// parameter is the name of the test within the test case.
//
// The convention is to end the test case name with "Test".  For
// example, a test case for the Foo class can be named FooTest.
//
// Test code should appear between braces after an invocation of
// this macro.  Example:
//
//   TEST(FooTest, InitializesCorrectly) {
//     Foo foo;
//     EXPECT_TRUE(foo.StatusIsOK());
//   }

// Note that we call GetTestTypeId() instead of GetTypeId<
// ::testing::Test>() here to get the type ID of testing::Test.  This
// is to work around a suspected linker bug when using Google Test as
// a framework on Mac OS X.  The bug causes GetTypeId<
// ::testing::Test>() to return different values depending on whether
// the call is from the Google Test framework itself or from user test
// code.  GetTestTypeId() is guaranteed to always return the same
// value, as it always calls GetTypeId<>() from the Google Test
// framework.
#define GTEST_TEST(test_case_name, test_name)\
  GTEST_TEST_(test_case_name, test_name, \
              ::testing::Test, ::testing::internal::GetTestTypeId())

// Define this macro to 1 to omit the definition of TEST(), which
// is a generic name and clashes with some other libraries.
#if !GTEST_DONT_DEFINE_TEST
# define TEST(test_case_name, test_name) GTEST_TEST(test_case_name, test_name)
#endif
```

上面的`GTEST_TEST_`宏定义如下:

```cpp
// Expands to the name of the class that implements the given test.
#define GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
  test_case_name##_##test_name##_Test

// Helper macro for defining tests.
#define GTEST_TEST_(test_case_name, test_name, parent_class, parent_id)\
class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public parent_class {\
 public:\
  GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}\
 private:\
  virtual void TestBody();\
  static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;\
  GTEST_DISALLOW_COPY_AND_ASSIGN_(\
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name));\
};\
\
::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name, test_name)\
  ::test_info_ =\
    ::testing::internal::MakeAndRegisterTestInfo(\
        #test_case_name, #test_name, NULL, NULL, \
        (parent_id), \
        parent_class::SetUpTestCase, \
        parent_class::TearDownTestCase, \
        new ::testing::internal::TestFactoryImpl<\
            GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>);\
void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody()
```

通过上面代码， 我们可以看到，
通过`TEST`，我们实际是定义了一个以`test_case_name`,`test_name`组成的`class`,
并继承自`Test`,然后`TEST`后面代码实际是这个`class`里`TestBody`的函数体。

## `TEST_F`宏定义

```cpp
// Defines a test that uses a test fixture.
//
// The first parameter is the name of the test fixture class, which
// also doubles as the test case name.  The second parameter is the
// name of the test within the test case.
//
// A test fixture class must be declared earlier.  The user should put
// his test code between braces after using this macro.  Example:
//
//   class FooTest : public testing::Test {
//    protected:
//     virtual void SetUp() { b_.AddElement(3); }
//
//     Foo a_;
//     Foo b_;
//   };
//
//   TEST_F(FooTest, InitializesCorrectly) {
//     EXPECT_TRUE(a_.StatusIsOK());
//   }
//
//   TEST_F(FooTest, ReturnsElementCountCorrectly) {
//     EXPECT_EQ(0, a_.size());
//     EXPECT_EQ(1, b_.size());
//   }

#define TEST_F(test_fixture, test_name)\
  GTEST_TEST_(test_fixture, test_name, test_fixture, \
              ::testing::internal::GetTypeId<test_fixture>())

}  // namespace testing
```

其实`TEST_F`与上面`TEST`类似，上面主要不同就是`TEST`直接继承`Test`，而`TEST_F`需要`test_fixture`先继承`Test`,
然后`test_fixture`,`test_case`组成的`class`再继承`test_fixture`，
也就是通过这种方式我们可以用过在`test_fixture`里`override`来定制化操作。

## `TEST_P`定义

`TEST_P`的宏定义具体如下:


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
```

与`TEST_F`类似， 定义的`test_case_name`首先需要定义为一个继承了`Test`和`WithParamInterface`的`class`，不过一般只需要继承`TestWithParam`即可
`TestWithParam`与`Test`，`WithParamInterface`关系如下代码所示:

```cpp
// The pure interface class that all value-parameterized tests inherit from.
// A value-parameterized class must inherit from both ::testing::Test and
// ::testing::WithParamInterface. In most cases that just means inheriting
// from ::testing::TestWithParam, but more complicated test hierarchies
// may need to inherit from Test and WithParamInterface at different levels.
//
// This interface has support for accessing the test parameter value via
// the GetParam() method.
//
// Use it with one of the parameter generator defining functions, like Range(),
// Values(), ValuesIn(), Bool(), and Combine().
//
// class FooTest : public ::testing::TestWithParam<int> {
//  protected:
//   FooTest() {
//     // Can use GetParam() here.
//   }
//   virtual ~FooTest() {
//     // Can use GetParam() here.
//   }
//   virtual void SetUp() {
//     // Can use GetParam() here.
//   }
//   virtual void TearDown {
//     // Can use GetParam() here.
//   }
// };
// TEST_P(FooTest, DoesBar) {
//   // Can use GetParam() method here.
//   Foo foo;
//   ASSERT_TRUE(foo.DoesBar(GetParam()));
// }
// INSTANTIATE_TEST_CASE_P(OneToTenRange, FooTest, ::testing::Range(1, 10));

template <typename T>
class WithParamInterface
```

```cpp
// Most value-parameterized classes can ignore the existence of
// WithParamInterface, and can just inherit from ::testing::TestWithParam.

template <typename T>
class TestWithParam : public Test, public WithParamInterface<T> {
};
```

上面的`TestBody`里可以使用`GetParam()`来使用后面`INSTANTIATE_TEST_CASE_P(prefix, test_case_name, generator)`里`generator`里传递进来的参数。通过`INSTANTIATE_TEST_CASE_P(prefix, test_case_name, generator)`来定义需要传递的参数

```cpp
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

产生参数可以用`Values`和`Range`，`Values`是使用指定的参数列表， 而`Range`则是在指定的范围内按指定的步长来产生参数， 其中`Values`代码如下:

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

# `gtest`执行流程

使用上面的`Test`, `TEST_F`或`TEST_P`定义好`test_case`和`test`之后， 就需要在`main`函数里顺序执行`InitGoogleTest`来初始化`gtest`测试框架，然后调用`RUN_ALL_TESTS`来执行所有的测试。

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

```cpp

// Use this function in main() to run all tests.  It returns 0 if all
// tests are successful, or 1 otherwise.
//
// RUN_ALL_TESTS() should be invoked after the command line has been
// parsed by InitGoogleTest().
//
// This function was formerly a macro; thus, it is in the global
// namespace and has an all-caps name.
int RUN_ALL_TESTS() GTEST_MUST_USE_RESULT_;

inline int RUN_ALL_TESTS() {
  return ::testing::UnitTest::GetInstance()->Run();
}
```

# `gtest`基本类

上面我们使用的不管是`TEST`,`TEST_F`还是`TEST_P`最终都继承了`Test`类。在`gtest`里，`Test`是所有`test`都要继承的基类

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


`gtest`提供了一系列的`EXPECT_*`和`ASSERT_*`，`EXPECT_*`比较fail之后会继续执行后面的代码， 而`ASSERT_*`失败之后会停止执行。
其具体实现如下所示:

```cpp
// Boolean assertions. Condition can be either a Boolean expression or an
// AssertionResult. For more information on how to use AssertionResult with
// these macros see comments on that class.
#define EXPECT_TRUE(condition) \
  GTEST_TEST_BOOLEAN_(condition, #condition, false, true, \
                      GTEST_NONFATAL_FAILURE_)
#define EXPECT_FALSE(condition) \
  GTEST_TEST_BOOLEAN_(!(condition), #condition, true, false, \
                      GTEST_NONFATAL_FAILURE_)
#define ASSERT_TRUE(condition) \
  GTEST_TEST_BOOLEAN_(condition, #condition, false, true, \
                      GTEST_FATAL_FAILURE_)
#define ASSERT_FALSE(condition) \
  GTEST_TEST_BOOLEAN_(!(condition), #condition, true, false, \
                      GTEST_FATAL_FAILURE_)
```


```cpp
// Implements Boolean test assertions such as EXPECT_TRUE. expression can be
// either a boolean expression or an AssertionResult. text is a textual
// represenation of expression as it was passed into the EXPECT_TRUE.
#define GTEST_TEST_BOOLEAN_(expression, text, actual, expected, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (const ::testing::AssertionResult gtest_ar_ = \
      ::testing::AssertionResult(expression)) \
    ; \
  else \
    fail(::testing::internal::GetBoolAssertionFailureMessage(\
        gtest_ar_, text, #actual, #expected).c_str())
```




# 测试实例

简单使用`TEST`,`TEST_F`,`TEST_P`的测试代码:


```cpp
#include "gtest.h"

/*
 * TEST example
 */
TEST(test, EXPECT) {
    EXPECT_TRUE(1);
}

/*
 * TEST_F test example
 */
class testFixture : public testing::Test {};

TEST_F(testFixture, AssertEQ) {
    ASSERT_EQ(5, 5);
}

TEST_F(testFixture, AssertTrue) {
    ASSERT_TRUE(true);
    ASSERT_TRUE(false) << "failure example";
}

/* ------------------------- */

/*
 * TEST_P test example
 */
class testParameter : public testing::TestWithParam<int> {};

TEST_P(testParameter, TrueOrFalse) {
    auto para = GetParam();
    EXPECT_TRUE(para%2 == 0);
}

INSTANTIATE_TEST_CASE_P(
    Values, testParameter, ::testing::Values(2,4)
);

INSTANTIATE_TEST_CASE_P(
    Range, testParameter, ::testing::Range(6,8,2)
);

/* ------------------------- */

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

测试输出 ：

![gtest output](/assets/images/gtest_outputs.png)
