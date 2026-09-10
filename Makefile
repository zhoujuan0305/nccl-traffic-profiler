CXX ?= g++
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra
PYTHON ?= python3
BUILD_DIR ?= build
TARGET = $(BUILD_DIR)/libnccl-profiler-fine.so

all: $(TARGET)

$(TARGET): plugin.cc trace_writer.h $(wildcard nccl/*.h)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -shared -pthread -I. plugin.cc -o $@

$(BUILD_DIR)/test_plugin: plugin.cc test_plugin.cc trace_writer.h $(wildcard nccl/*.h)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -pthread -I. plugin.cc test_plugin.cc -o $@

check: $(BUILD_DIR)/test_plugin
	$(PYTHON) check_plugin.py $(BUILD_DIR)/test_plugin
	$(PYTHON) -m unittest -v test_aggregate.py

.PHONY: all check
