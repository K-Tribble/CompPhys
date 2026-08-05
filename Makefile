# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++20 -Wall -Wextra -Iinclude -O3

# Source and object files
SRC = src/main.cpp src/matrix.cpp src/vec.cpp src/linalg_interop.cpp
OBJ = $(SRC:.cpp=.o)

# Executable name
TARGET = main

# Default target
all: $(TARGET)

# Link object files
$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET)

# Compile source files
src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJ) $(TARGET)

# Rebuild from scratch
rebuild: clean all

.PHONY: all clean rebuild

count:
	find . -name '*.cpp' -o -name '*.hpp' | xargs wc -l