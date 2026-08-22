CXX = clang++

OPENMP_PREFIX := $(shell brew --prefix libomp)

CXXFLAGS = -std=c++20 -Wall -Wextra -O2 \
		   -Iinclude \
           -Xpreprocessor -fopenmp \
           -I$(OPENMP_PREFIX)/include

LDFLAGS = -L$(OPENMP_PREFIX)/lib -lomp

SRC = src/main.cpp \
      src/interpolation.cpp \
      src/calculus/differentiation.cpp

OBJ = $(SRC:src/%.cpp=build/%.o)

TARGET = build/main

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) $(LDFLAGS) -o $@

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf build

rebuild: clean all

count:
	find . -path './tests' -prune -o \
		\( -name '*.cpp' -o -name '*.hpp' -o -name '*.tpp' \) -print0 | xargs -0 wc -l

TEST_SRC = tests/test_vec.cpp \
           tests/test_matrix.cpp \
           tests/test_interop.cpp \
           tests/test_differentiation.cpp \
           src/calculus/differentiation.cpp
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    CXX = clang++
    OPENMP_PREFIX := $(shell brew --prefix libomp)
    OMP_FLAGS   = -Xpreprocessor -fopenmp -I$(OPENMP_PREFIX)/include
    OMP_LDFLAGS = -L$(OPENMP_PREFIX)/lib -lomp
    CATCH2_PREFIX := $(shell brew --prefix catch2)
    CATCH2_INC = -I$(CATCH2_PREFIX)/include
    CATCH2_LIB = -L$(CATCH2_PREFIX)/lib
else
    CXX = g++
    OMP_FLAGS   = -fopenmp
    OMP_LDFLAGS = -fopenmp
    CATCH2_INC  =
    CATCH2_LIB  =
endif

CXXFLAGS = -std=c++20 -Wall -Wextra -O2 \
		   -Iinclude \
           $(OMP_FLAGS)

LDFLAGS = $(OMP_LDFLAGS)

SRC = src/main.cpp \
      src/interpolation.cpp \
      src/calculus/differentiation.cpp

OBJ = $(SRC:src/%.cpp=build/%.o)

TARGET = build/main

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) $(LDFLAGS) -o $@

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf build

rebuild: clean all

count:
	find . -path './tests' -prune -o \
		\( -name '*.cpp' -o -name '*.hpp' -o -name '*.tpp' \) -print0 | xargs -0 wc -l

TEST_SRC = tests/test_vec.cpp \
           tests/test_matrix.cpp \
           tests/test_interop.cpp \
           tests/test_differentiation.cpp \
           src/calculus/differentiation.cpp \
           tests/test_integration.cpp \
           tests/test_nonlin.cpp \
           tests/test_linalg_solve.cpp \
		   tests/test_optimize.cpp

TEST_TARGET = build/tests

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRC)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) \
		$(CATCH2_INC) \
		$(CATCH2_LIB) \
		$(TEST_SRC) \
		-lCatch2Main -lCatch2 \
		$(LDFLAGS) \
		-o $(TEST_TARGET)

.PHONY: all run clean rebuild count testerentiation.cpp \
           tests/test_integration.cpp \
           tests/test_nonlin.cpp \
           tests/test_linalg_solve.cpp \
		   tests/test_optimize.cpp

TEST_TARGET = build/tests

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRC)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) \
		-I$(shell brew --prefix catch2)/include \
		-L$(shell brew --prefix catch2)/lib \
		$(TEST_SRC) \
		-lCatch2Main -lCatch2 \
		$(LDFLAGS) \
		-o $(TEST_TARGET)

.PHONY: all run clean rebuild count test