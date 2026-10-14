CXX = g++
CXXFLAGS = -O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mpopcnt -Wall -Wextra -pthread
CPPFLAGS = $(shell pkg-config --cflags libpng)
LIBS = $(shell pkg-config --libs libpng)

OUT = /tmp/halcon-study-threshold-test

.PHONY: test
test: $(OUT)
	$(OUT) img/bottle_crate_01.png

$(OUT): threshold.cpp threshold.hpp test_threshold.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $(OUT) threshold.cpp test_threshold.cpp $(LIBS)
