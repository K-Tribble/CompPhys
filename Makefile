# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++20 -Wall -Wextra -Iinclude -O3

# Source and object files
SRC = src/main.cpp \
	src/matrix.cpp \
	src/vec.cpp \
	src/linalg_interop.cpp \
	src/linalg_solve.cpp \

OBJ = $(SRC:src/%.cpp=build/%.o)

# Executable name
TARGET = build/main

# Default target
all: $(TARGET)

# Link object files
$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET)

# Compile source files into build/
build/%.o: src/%.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf build

# Rebuild from scratch
rebuild: clean all

.PHONY: all clean rebuild

count:
	find . \( -name '*.cpp' -o -name '*.hpp' \) -print0 | xargs -0 wc -l

main:
	./build/main