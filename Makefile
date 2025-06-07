# Compiler to use
CC = gcc

# Compiler flags
CFLAGS = -mavx2 -mfma -O3 -Wall

# Target executable name
TARGET = matmul

# Source files
SOURCES = matmul.c

# Object files
OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Phony targets
.PHONY: all clean
