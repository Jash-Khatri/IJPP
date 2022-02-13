// constants.h
#if !defined(MYLIB_CONSTANTS_H)
#define MYLIB_CONSTANTS_H 1

const int NUM_ITER_FIXPOINT = 0; 
const int NUM_ITER_LOOP = 100; 
const double SPECT_SPARSE_P = 0.9;  // constant to adjust the avg vertex degree to add more inaccuracy reduce
const double SPECT_SPARSE_PROB = 0.4;  // threshold for the avg vertex degree to add more inaccuracy increase
const int ENABLE_SPEC_SPARSE = 1;  // Use this variable to enable the spectral sparsifaction
const int ENABLE_VERTEX_RENUMBERING = 1; // use this to enable the vertex renumbering
const int EXCESS_CYCLE_REMOVE = 1;   // use this for enabling excess cycle remvoal
const int ENABLE_REUSING_MIN_HEIGHTS = 1; // use this to enable reusing min-heights during MF computation
const int THRESHOLD = 0;            // use for memory access pruning
int UNDIRECTED = 0;	 // check if graph is undirected

#endif
