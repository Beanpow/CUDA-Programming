#pragma once
#include "common.cuh"
#include "error.cuh"

void allocate_memory(int N, int MN, Atom *atom);
void deallocate_memory(Atom *atom);

