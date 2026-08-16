# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++20 -Wall -Wextra -Iinclude -O3

# Source and object files
SRC = src/main.cpp \
	src/linalg/matrix.cpp \
	src/linalg/vec.cpp \
	src/linalg/linalg_interop.cpp \
	src/linalg/linalg_solve.cpp \
	src/interpolation.cpp \
	src/calculus/differentiation.cpp

OBJ = $(SRC:src/%.cpp=build/%.o)

# Executable
TARGET = build/main

# Default target
all: $(TARGET)

# Link object files
$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET)

# Compile source files into build/
build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run main program
run: $(TARGET)
	./$(TARGET)

# Clean build files
clean:
	rm -rf build

# Rebuild from scratch
rebuild: clean all

# Count lines
count:
	find . \( -name '*.cpp' -o -name '*.hpp' \) -print0 | xargs -0 wc -l

TEST_SRC = tests/test_vec.cpp \
	tests/test_matrix.cpp \
	tests/test_interop.cpp \
	tests/test_eigen.cpp \
	src/linalg/matrix.cpp \
	src/linalg/vec.cpp \
	src/linalg/linalg_interop.cpp \
	src/linalg/linalg_solve.cpp \
	tests/test_differentiation.cpp \
	tests/test_integration.cpp \
	src/calculus/differentiation.cpp \
	tests/test_nonlin.cpp \

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
		-o $(TEST_TARGET)

.PHONY: all run clean rebuild count test