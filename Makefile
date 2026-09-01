UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)

    # macOS
    CXX = clang++

    # OpenMP
    OPENMP_PREFIX := $(shell brew --prefix libomp)
    OMP_FLAGS     = -Xpreprocessor -fopenmp -I$(OPENMP_PREFIX)/include
    OMP_LDFLAGS   = -L$(OPENMP_PREFIX)/lib -lomp

    # Catch2
    CATCH2_PREFIX := $(shell brew --prefix catch2)
    CATCH2_INC    = -I$(CATCH2_PREFIX)/include
    CATCH2_LIB    = -L$(CATCH2_PREFIX)/lib

else

    # Linux
    CXX = g++

    # OpenMP
    OMP_FLAGS     = -fopenmp
    OMP_LDFLAGS   = -fopenmp

    # Catch2
    CATCH2_INC    =
    CATCH2_LIB    =

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

ANHARMONIC_SRC = src/anharmonic_oscillator.cpp
ANHARMONIC_TARGET = build/anharmonic

anharmonic: $(ANHARMONIC_TARGET)

$(ANHARMONIC_TARGET): $(ANHARMONIC_SRC)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@


TEST_SRC = tests/test_vec.cpp \
           tests/test_matrix.cpp \
           tests/test_interop.cpp \
           tests/test_differentiation.cpp \
           tests/test_integration.cpp \
           tests/test_nonlin.cpp \
           tests/test_linalg_solve.cpp \
           tests/test_optimize.cpp \
           src/calculus/differentiation.cpp

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
		-o $@

clean:
	rm -rf build


rebuild: clean all


count:
	find . -path './tests' -prune -o \
		\( -name '*.cpp' -o -name '*.hpp' -o -name '*.tpp' \) -print0 | \
		xargs -0 wc -l


.PHONY: all run anharmonic test clean rebuild count
