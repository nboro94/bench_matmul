#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <random>
#include <string>
#include <vector>

// Global configurations
extern bool VERBOSE_LOGGING;
extern int N;
extern int BLOCK_SIZE;
extern unsigned int NUM_THREADS;

// Logging
void log(const std::string &message);

// Hardware detection and configuration
void determineOptimalThreadCount();
void determineOptimalBlockSize();

// Memory management
extern int matrixAllocations;
extern int matrixDeallocations;
float **allocateMatrix();
void deallocateMatrix(float **matrix);
void checkMemoryLeaks();

// Matrix initialization
void initializeRandomMatrix(float **matrix, std::mt19937 &gen,
                            std::uniform_real_distribution<float> &dist);
void initializeRandomMatrixFast(float **matrix, std::mt19937 &gen,
                                std::uniform_real_distribution<float> &dist);

// Matrix operations
void printMatrix(float **matrix, const std::string &name);
void transposeMatrix(float **input, float **output);
void transposeMatrixParallel(float **input, float **output);
bool almostEqual(float a, float b);

#endif // UTILS_H
