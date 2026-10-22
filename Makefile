CXX = g++
CXXFLAGS = -O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mpopcnt -Wall -Wextra -pthread
CPPFLAGS = $(shell pkg-config --cflags libpng)
LIBS = $(shell pkg-config --libs libpng)

OUT = /tmp/halcon-study-threshold-test
CONN = /tmp/halcon-study-connection-test
FEAT = /tmp/halcon-study-features-test

.PHONY: test
test: $(OUT) $(CONN) $(FEAT)
	$(OUT) img/bottle_crate_01.png
	$(CONN) img/bottle_crate_01.png
	$(FEAT) img/bottle_crate_01.png

$(OUT): threshold.cpp threshold.hpp test_threshold.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $(OUT) threshold.cpp test_threshold.cpp $(LIBS)

$(CONN): threshold.cpp connection.cpp threshold.hpp test_connection.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $(CONN) threshold.cpp connection.cpp test_connection.cpp $(LIBS)

$(FEAT): threshold.cpp connection.cpp features.cpp threshold.hpp features.hpp test_features.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $(FEAT) threshold.cpp connection.cpp features.cpp test_features.cpp $(LIBS)
