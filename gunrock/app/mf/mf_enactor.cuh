// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
* @file
* mf_enactor.cuh
*
* @brief Max Flow Problem Enactor
*/

#pragma once
#include <gunrock/util/sort_device.cuh>
#include <gunrock/util/constants.h>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>

#include <gunrock/app/mf/mf_helpers.cuh>
#include <gunrock/app/mf/mf_problem.cuh>

#include <gunrock/oprtr/1D_oprtr/for.cuh>
#include <gunrock/oprtr/oprtr.cuh>

// uncoment for debug output
// #define MF_DEBUG 1

#if MF_DEBUG
#define debug_aml(a...) printf(a);
#else
#define debug_aml(a...)
#endif
#define QUEUE_SZ 8192
namespace gunrock {
namespace app {
namespace mf {

/**
* @brief Speciflying parameters for MF Enactor
* @param parameters The util::Parameter<...> structure holding all parameter
*		      info
* \return cudaError_t error message(s), if any
*/
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}
/**
* @brief defination of MF iteration loop
* @tparam EnactorT Type of enactor
*/
template <typename EnactorT>
struct MFIterationLoop : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::Problem ProblemT;
  typedef typename ProblemT::GraphT GraphT;
  typedef typename GraphT::CsrT CsrT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  MFIterationLoop() : BaseIterationLoop() {}

  /**
  * @brief Core computation of mf, one iteration
  * @param[in] peer_ Which GPU peers to work on, 0 means local
  * \return cudaError_t error message(s), if any
  */
  cudaError_t Core(int peer_ = 0) {
    auto enactor = this->enactor;
    auto gpu_num = this->gpu_num;
    auto num_gpus = enactor->num_gpus;
    auto gpu_offset = num_gpus * gpu_num;
    auto &data_slice = enactor->problem->data_slices[gpu_num][0];
    auto &enactor_slice = enactor->enactor_slices[gpu_offset + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &iteration = enactor_stats.iteration;

    auto source = data_slice.source;
    auto sink = data_slice.sink;
    auto num_repeats = data_slice.num_repeats;
    auto &capacity = graph.CsrT::edge_values;
  auto &numofrelabel = data_slice.numofrelabel;		// added the new line...
    auto &ctr = data_slice.ctr;                // added the new line...
    auto &ifexcess = data_slice.ifexcess;                // added the new line...
    auto &height_indirection = data_slice.height_indirection;
    auto &v_que = data_slice.v_que;
    auto &reverse = data_slice.reverse;
    auto &flow = data_slice.flow;
    auto &residuals = data_slice.residuals;
    auto &excess = data_slice.excess;
    auto &height = data_slice.height;
    auto &lowest_neighbor = data_slice.lowest_neighbor;
    auto &local_vertices = data_slice.local_vertices;
    auto &active = data_slice.active;
    auto null_ptr = &local_vertices;
    null_ptr = NULL;
    auto &d_parent = data_slice.mark;
    auto &queue = data_slice.queue0;
    auto &queue0 = data_slice.queue0;
    auto &queue1 = data_slice.queue1;
    auto &d_currentQueue = data_slice.queue0;
    auto &d_nextQueue = data_slice.queue1;
    auto &st = data_slice.st;
    auto &en = data_slice.en;

    int *incrDegrees;
    int *d_degrees;
    cudaMalloc(&d_degrees, graph.nodes* sizeof(int));
    cudaMallocHost((void **) &incrDegrees, sizeof(int) * (graph.nodes/1024 +2));


    auto advance_preflow_op =
        [capacity, flow, excess, height, reverse, source, residuals,sink] __host__
        __device__(const VertexT &src, VertexT &dest, const SizeT &edge_id,
                  const VertexT &input_item, const SizeT &input_pos,
                  const SizeT &output_pos) -> bool {
      if (!util::isValid(dest) or !util::isValid(src)) return false;
      if (dest != source && src!=sink) residuals[edge_id] = capacity[edge_id];
      if (src != source && dest != sink) return false;
      auto c = capacity[edge_id];
      residuals[edge_id] = 0;
      residuals[reverse[edge_id]] = capacity[reverse[edge_id]] + c;
      atomicAdd(excess + src, -c);
      atomicAdd(excess + dest, c);
      return true;

    };
  auto nextLayer = [graph,height,residuals,reverse, d_parent,d_currentQueue, d_nextQueue,excess,capacity] __host__
    __device__ (int thid, int level, int queueSize,int ctr, int type) {

    if (thid < queueSize){
      int u;
      if (ctr %2 == 0){
          u = d_currentQueue[thid];
      }
      else{
          u = d_nextQueue[thid];
      }
      auto e_start = graph.CsrT::GetNeighborListOffset(u);
      auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
      auto e_end = e_start + num_neighbors;
      for (auto e = e_start; e < e_end; ++e) {
            auto neighbour = graph.CsrT::GetEdgeDest(e);
            if (type == 1) {
              if (residuals[reverse[e]] < MF_EPSILON && (excess[neighbour]>=0 || capacity[reverse[e]]<MF_EPSILON)) 
              continue;
            }
            else {
              if (residuals[e] < MF_EPSILON && (excess[neighbour]<=0 || capacity[e]<MF_EPSILON))
              continue;
            }
            if (d_parent[neighbour] == 2e9) {
              height[neighbour] = level + type;
              d_parent[neighbour] = e;
            }
      }
    }
  };


auto countDegrees = [graph, d_parent,d_currentQueue, d_degrees,d_nextQueue] __host__
__device__ (int thid, int queueSize, int ctr) {

  if (thid < queueSize){
    int u;
    if (ctr %2 == 0) {
      u = d_currentQueue[thid];
    }
    else {
      u = d_nextQueue[thid];
    }
    int degree = 0;
    auto e_start = graph.CsrT::GetNeighborListOffset(u);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
    auto e_end = e_start + num_neighbors;
    for (auto e = e_start; e < e_end; ++e) {
          auto neighbour = graph.CsrT::GetEdgeDest(e);
          if (d_parent[neighbour] == e && neighbour != u) {
            ++degree;
          }
    }
    d_degrees[thid] = degree;
  }
};


auto assignVerticesNextQueue = [graph, d_parent, d_currentQueue, d_nextQueue]
__device__ ( int thid, int *incrDegrees, int queueSize, int *d_degrees, int ctr) {

  if (ctr%2 == 0){

    if (thid < queueSize) {
        __shared__ int sharedIncrement;
        if (!threadIdx.x) {
            sharedIncrement = incrDegrees[thid >> 10];
        }
        __syncthreads();

        int sum = 0;
        if (threadIdx.x) {
            sum = d_degrees[thid - 1];
        }

        int u = d_currentQueue[thid];
        int counter = 0;
        auto e_start = graph.CsrT::GetNeighborListOffset(u);
        auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e) {
              auto neighbour = graph.CsrT::GetEdgeDest(e);
              if (d_parent[neighbour] == e && neighbour != u) {
                int nextQueuePlace = sharedIncrement + sum + counter;
                d_nextQueue[nextQueuePlace] = neighbour;
                counter++;
              }
        }
      }
    }
    else{


      if (thid < queueSize) {
        __shared__ int sharedIncrement;
        if (!threadIdx.x) {
            sharedIncrement = incrDegrees[thid >> 10];
        }
        __syncthreads();

        int sum = 0;
        if (threadIdx.x) {
            sum = d_degrees[thid - 1];
        }

        int u = d_nextQueue[thid];
        int counter = 0;
        auto e_start = graph.CsrT::GetNeighborListOffset(u);
        auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e) {
              auto neighbour = graph.CsrT::GetEdgeDest(e);
              if (d_parent[neighbour] == e && neighbour != u) {
                int nextQueuePlace = sharedIncrement + sum + counter;
                d_currentQueue[nextQueuePlace] = neighbour;
                counter++;
              }
        }

      }
    }
};

/*
auto runCudaScanBfs = [graph,d_parent,height,d_currentQueue,d_nextQueue,nextLayer,countDegrees,assignVerticesNextQueue,d_degrees,incrDegrees,oprtr_parameters] __host__

 (int startVertex) {

    //launch kernel
    printf("Starting scan parallel bfs.\n");
    //auto start = std::chrono::steady_clock::now();
    int queueSize = 1;
    int nextQueueSize = 0;
    int level = height[startVertex];
    int ct= 0;
    while (queueSize) {
        // next layer phase
        gunrock::oprtr::For([nextLayer, level, queueSize,ct] __device__(const int &i) { nextLayer(i, level,queueSize,ct); },
                queueSize, util::DEVICE, oprtr_parameters.stream, queueSize / 1024 + 1 , 1024);
                  //printf("dsff1");fflush(stdout);
        // counting degrees phase
        gunrock::oprtr::For([countDegrees, queueSize,ct] __device__(const int &i) { countDegrees(i,queueSize,ct); },
                  queueSize, util::DEVICE, oprtr_parameters.stream, queueSize / 1024 + 1 , 1024);
        // doing scan on degrees
        //printf("dsff2");fflush(stdout);
        gunrock::oprtr::scanDegrees <<<  queueSize / 1024 + 1 , 1024 , 0, oprtr_parameters.stream >>> ( queueSize,
                  d_degrees, incrDegrees); 
        cudaStreamSynchronize(oprtr_parameters.stream);
        //printf("dsff3");fflush(stdout);     

        //count prefix sums on CPU for ends of blocks exclusive
        //already written previous block sum
        incrDegrees[0] = 0;
        //printf("incr1 = %d\n",incrDegrees[1]);
        for (int i = 1024; i < queueSize + 1024; i += 1024) {
            incrDegrees[i / 1024] += incrDegrees[i / 1024 - 1];
        }
        nextQueueSize = incrDegrees[(queueSize - 1) / 1024 + 1];
        //printf("nq = %d\n",nextQueueSize);fflush(stdout);
        // assigning vertices to nextQueue
        gunrock::oprtr::For([assignVerticesNextQueue, incrDegrees, queueSize, d_degrees,ct] __device__(const int &i) { assignVerticesNextQueue(i, incrDegrees, queueSize,d_degrees,ct); },
                  queueSize, util::DEVICE, oprtr_parameters.stream, queueSize / 1024 + 1 , 1024);

        level++;
        queueSize = nextQueueSize;
        ++ct;
      }

};*/

    
    auto global_relabeling_op =
        [graph, source, sink, height, reverse, queue, residuals] __host__
        __device__(VertexT * v_q, const SizeT &pos) {
          VertexT first = 0, last = 0;
          queue[last++] = sink;
          auto H = (VertexT)0;
          height[sink] = 0;
          while (first < last) {
            auto v = queue[first++];
            auto e_start = graph.CsrT::GetNeighborListOffset(v);
            auto num_neighbors = graph.CsrT::GetNeighborListLength(v);
            auto e_end = e_start + num_neighbors;
            //++H;
            H = height[v];
            for (auto e = e_start; e < e_end; ++e) {
              auto neighbor = graph.CsrT::GetEdgeDest(e);
              if (residuals[reverse[e]] < MF_EPSILON) continue;
              if (height[neighbor] > H + 1) {
                height[neighbor] = H + 1;
                queue[last++] = neighbor;
              }
            }
          }
          height[source] = graph.nodes;
          first = 0;
          last = 0;
          queue[last++] = source;
          //H = (VertexT)graph.nodes;
          while (first < last) {
            auto v = queue[first++];
            auto e_start = graph.CsrT::GetNeighborListOffset(v);
            auto num_neighbors = graph.CsrT::GetNeighborListLength(v);
            auto e_end = e_start + num_neighbors;
            //++H;
            H = height[v];
            for (auto e = e_start; e < e_end; ++e) {
              auto neighbor = graph.CsrT::GetEdgeDest(e);
              if (residuals[reverse[e]] < MF_EPSILON) continue;
              if (height[neighbor] > H + 1) {
                height[neighbor] = H + 1;
                queue[last++] = neighbor;
              }
            }
          }
    };

auto cmp = [source,sink, height, excess]  __device__ (const VertexT &a, const VertexT &b){
  if (a == sink) return (bool)0;
  else if (a == source) {if (b == source) return (bool)0;else return (bool)1;}  
 else if (b == source || b ==sink) return (bool)1;
  if (excess[a] < 0.1) return (bool)0;
  else if (excess[b] < 0.1) return (bool)1;
  else return height[a] > height[b]; };
auto cmp1 = [height_indirection,v_que,sink] __host__ __device__(const int &i) {
  v_que[height_indirection[i]]=1;
  height_indirection[i+QUEUE_SZ]=sink;};
auto cmp2 = [height_indirection,v_que] __host__ __device__(const int &i) {
  v_que[height_indirection[i]]=0;};

        auto compute_lockfree_op =
        [graph, excess, residuals, reverse, height, iteration, source, sink,
        active, numofrelabel, lowest_neighbor, ifexcess, ctr, height_indirection,v_que] __host__
        __device__(const int &counter, int type,  const VertexT &v) {
          // v - current vertex

  //int v = height_indirection[v1];
  if(EXCESS_CYCLE_REMOVE)
  {
          if(v == 0)
	  {                  // added
                ifexcess[ (counter + 1) % 2 ] = 0;		
      		ctr[0] = ctr[0]+1;
          }
          if( ifexcess[counter%2] == 0 ) return; 				
  }

          if (v == 0){ active[(counter + 1) % 2] = 0; } // printf("type = %d\n",type);}
  
    if(v == 0) numofrelabel[(counter + 1) % 2] = 0;  
          // if not a valid vertex, do not apply compute:
          if (!util::isValid(v) || v == source || v == sink) return;
	
	VertexT neighbor_num = graph.CsrT::GetNeighborListLength(v);
          ValueT excess_v = excess[v];
          if (excess_v!=0){ active[counter%2] = 1;}
          //else return;
          if ( ( abs(excess_v) < ( THRESHOLD == 0 ? MF_EPSILON : THRESHOLD  ) ) && type == 1 ) return;     // Added logic for memory access pruning in this Line 227     
          if ((type == 1 && excess_v <= 0 )||(type == -1 && excess_v >= 0 )) {return;}     // Added logic for memory access pruning
          
          if (neighbor_num == 0) return;
          // turn off vertices which relabeling drop out from graph

          // else, try push-relable:
          VertexT e_start = graph.CsrT::GetNeighborListOffset(v);
          VertexT e_end = e_start + neighbor_num;

          VertexT lowest_id = util::PreDefinedValues<VertexT>::InvalidValue;
          VertexT lowest_h = util::PreDefinedValues<VertexT>::MaxValue;
          ValueT lowest_r = 0;
          VertexT lowest_n = 0;
          int max_h  = -1e9, min_h = 1e9;
          auto e_id = e_start;
          // look for lowest height among neighbors

  VertexT x = lowest_neighbor[v];  		

  if((numofrelabel[(counter) % 2] == 1) || ( !util::isValid(x) ) ){		

    for (auto e = e_start; e < e_end; ++e) {
      if (excess[v]>0){
        auto y = graph.CsrT::GetEdgeDest(e);
        auto move = residuals[e];
        if (move > MF_EPSILON && (min_h > height[y]||height[y] == height[v]-1)){ 
          min_h = height[y];
          e_id = e;
          if (min_h == height[v]-1)break;
        }
      }
      else if (excess[v]<0){
        auto y = graph.CsrT::GetEdgeDest(e);
        auto rev_e = reverse[e];
        auto move = residuals[rev_e];
        if (move > MF_EPSILON && (max_h < height[y]|| height[y] == height[v]+1) ){
          max_h = height[y];
          e_id = rev_e;
          if (max_h == height[v]+1) break;
        }
      }
      else break;
    
    }    
  }
  
  else{					
    VertexT e_id = lowest_neighbor[v];
    
    ValueT r = residuals[e_id];  
                VertexT n = graph.CsrT::GetEdgeDest(e_id);
          VertexT h = height[n];
    lowest_id = e_id;
                lowest_h = h;
                lowest_r = r;
                lowest_n = n;
    numofrelabel[(counter + 1) % 2] = 1;        
  }          
          active[counter % 2] = 1;
          if (excess[v] > 0){
            if (height[v]>min_h){
              
              auto move = fminf(residuals[e_id], excess[v]);
              auto y = graph.CsrT::GetEdgeDest(e_id);

              atomicAdd(excess+v,-move);
              atomicAdd(excess+y,move);
              //printf("push %d to %d with %f\n", v, y, move);
              residuals[e_id] -= move;
              residuals[reverse[e_id]] += move;
            }
            else if (min_h !=1e9){
              height[v] = min_h+1;
            }
          }
          else if (excess[v]<0){
            if (height[v]<max_h){
              auto y = graph.CsrT::GetEdgeDest(reverse[e_id]);
              auto move =fminf(residuals[e_id], -excess[v]);
              atomicAdd(excess+v,move);
              atomicAdd(excess+y,-move);
              //printf("pull %d to %d with %f\n", v, y, move);
              residuals[reverse[e_id]] += move;
              residuals[e_id] -= move;
            
          }
          else if (max_h != -1e9) height[v] = max_h-1;
           
            }
  	if(EXCESS_CYCLE_REMOVE)
	{
        	  if((type == 1 && excess_v > 0) || (type == -1 && excess_v <0 ))
		  {
                	  ifexcess[ (counter + 1) % 2 ] = 1;
          	  }
  	}

  };


    if (iteration == 0) {
      // ADVANCE_PREFLOW_OP
      oprtr_parameters.advance_mode = "ALL_EDGES";
      GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
          graph.csr(), &local_vertices, null_ptr, oprtr_parameters,
          advance_preflow_op));
          auto start = time(NULL);
          ifexcess[0] = start;
          st = high_resolution_clock::now();
      // GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      // GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
      //        "cudaStreamSynchronize failed.");
    }
    /*
    std::ifstream fin("del.txt");
    int no; fin>>no;
    int u1 = 1,v1 = 2;
    int cnt = 0;VertexT e;
    ValueT new_capacity;
    if (ctr[0]<no){
    while (cnt <= ctr[0]){
      fin>>u1>>v1>>new_capacity;
      --u1;--v1;
      ++cnt;
    }
    e = graph.CsrT::GetNeighborListOffset(u1);
    while (1){
      if (graph.CsrT::GetEdgeDest(e) == v1){
        break;
      }
      ++e;
    }
  }*/



    // Global relabeling
    // Height reset
    //    if (iteration == 0){
    //    fprintf(stderr, "global relabeling in iteration %d\n", iteration);
    /*GUARD_CU(height.ForAll(
        [graph,sink,source] __host__ __device__(int * h, const VertexT &pos) {
          if (pos == sink) h[pos] = -graph.nodes;
          else if (pos == source) h[pos] = graph.nodes;
          else h[pos] = 0;
        },
        graph.nodes, util::DEVICE, oprtr_parameters.stream));*/
    height[source] = graph.nodes;
    height[sink] = -graph.nodes;


    // Serial relabeling on the GPU (ignores moves)
    /*
    GUARD_CU(frontier.V_Q()->ForAll(global_relabeling_op, 1, util::DEVICE,
                                    //      GUARD_CU(frontier.V_Q()->ForAll(par_global_relabeling_op,
                                    //      2, util::DEVICE,
                                    oprtr_parameters.stream));*/
    int type = 2*((iteration+1)%2) - 1;
    int startVertex = (type == 1 ? sink : source);
    

    GUARD_CU(d_parent.ForAll(
      [sink,source] __host__ __device__(SizeT * d_parent_, const VertexT &pos) {
        if (pos == sink || pos == source)d_parent_[pos] = -1;
        else d_parent_[pos] = 2e9;
      },
      graph.nodes, util::DEVICE, oprtr_parameters.stream));

    GUARD_CU(d_currentQueue.ForAll(
      [startVertex] __host__ __device__(SizeT * d_currentQueue_, const VertexT &pos) {
        d_currentQueue_[pos] = startVertex;
      },
      1, util::DEVICE, oprtr_parameters.stream));

    int queueSize = 1;
    int nextQueueSize = 0;
    int level = height[startVertex];
    int ct= 0;
    while (queueSize) {
        // next layer phase
        gunrock::oprtr::For([nextLayer, level, queueSize,ct,type] __device__(const int &i) { nextLayer(i, level,queueSize,ct,type); },
                  queueSize, util::DEVICE, oprtr_parameters.stream, queueSize / 1024 + 1 , 1024);
        // counting degrees phase
        gunrock::oprtr::For([countDegrees, queueSize,ct] __device__(const int &i) { countDegrees(i,queueSize,ct); },
                  queueSize, util::DEVICE, oprtr_parameters.stream, queueSize / 1024 + 1 , 1024);
        // doing scan on degrees
        gunrock::oprtr::scanDegrees <<<  queueSize / 1024 + 1 , 1024 , 0, oprtr_parameters.stream >>> ( queueSize,
                  d_degrees, incrDegrees); 
                  GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
                  "cudaStreamSynchronize failed.");

        //count prefix sums on CPU for ends of blocks exclusive
        //already written previous block sum
        incrDegrees[0] = 0;
        //printf("incr1 = %d\n",incrDegrees[1]);
        for (int i = 1024; i < queueSize + 1024; i += 1024) {
            incrDegrees[i / 1024] += incrDegrees[i / 1024 - 1];
        }
        nextQueueSize = incrDegrees[(queueSize - 1) / 1024 + 1];
        //printf("nq = %d\n",nextQueueSize);fflush(stdout);
        // assigning vertices to nextQueue
        gunrock::oprtr::For([assignVerticesNextQueue, incrDegrees, queueSize, d_degrees,ct] __device__(const int &i) { assignVerticesNextQueue(i, incrDegrees, queueSize,d_degrees,ct); },
                  queueSize, util::DEVICE, oprtr_parameters.stream, queueSize / 1024 + 1 , 1024);

        level+=type;
        queueSize = nextQueueSize;
        ++ct;
      }

    startVertex = (type == 1 ? source : sink);;
  
      GUARD_CU(d_currentQueue.ForAll(
        [startVertex] __host__ __device__(SizeT * d_currentQueue_, const VertexT &pos) {
          d_currentQueue_[pos] = startVertex;
        },
        1, util::DEVICE, oprtr_parameters.stream));
  
      queueSize = 1;
      nextQueueSize = 0;
      level = height[startVertex] ;
      ct= 0;
      while (queueSize) {
          // next layer phase
          gunrock::oprtr::For([nextLayer, level, queueSize,ct,type] __device__(const int &i) { nextLayer(i, level,queueSize,ct,type); },
                    queueSize, util::DEVICE, oprtr_parameters.stream, queueSize / 1024 + 1 , 1024);
          // counting degrees phase
          gunrock::oprtr::For([countDegrees, queueSize,ct] __device__(const int &i) { countDegrees(i,queueSize,ct); },
                    queueSize, util::DEVICE, oprtr_parameters.stream, queueSize / 1024 + 1 , 1024);
          // doing scan on degrees
          gunrock::oprtr::scanDegrees <<<  queueSize / 1024 + 1 , 1024 , 0, oprtr_parameters.stream >>> ( queueSize,
                    d_degrees, incrDegrees); 
                    GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
                    "cudaStreamSynchronize failed.");
  
          //count prefix sums on CPU for ends of blocks exclusive
          //already written previous block sum
          incrDegrees[0] = 0;
          for (int i = 1024; i < queueSize + 1024; i += 1024) {
              incrDegrees[i / 1024] += incrDegrees[i / 1024 - 1];
          }
          nextQueueSize = incrDegrees[(queueSize - 1) / 1024 + 1];
          // assigning vertices to nextQueue
          gunrock::oprtr::For([assignVerticesNextQueue, incrDegrees, queueSize, d_degrees,ct] __device__(const int &i) { assignVerticesNextQueue(i, incrDegrees, queueSize,d_degrees,ct); },
                    queueSize, util::DEVICE, oprtr_parameters.stream, queueSize / 1024 + 1 , 1024);
  
          level+=type;
          queueSize = nextQueueSize;
          ++ct;
        }
        /*
    height.Move(util::DEVICE, util::HOST, graph.nodes, 0, oprtr_parameters.stream);

      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
            "cudaStreamSynchronize failed.");
    for (int i = 0 ; i < graph.nodes; ++i){
      printf("%d = %d ", i , height[i]);
    }*/
    debug_aml("[%d]frontier que length before compute op is %d\n", iteration,
              frontier.queue_length);

    // Run Lockfree Push-Relable
    // GUARD_CU(frontier.V_Q()->ForAll(compute_lockfree_op,
    //            graph.nodes, util::DEVICE, oprtr_parameters.stream));
    //num_repeats = 10000;
    SizeT loop_size = graph.nodes;
    num_repeats = 100;

    
    for (int r = 0; r < num_repeats; r++) 
    {
   	GUARD_CU(ifexcess.ForAll(
          [] __device__(SizeT * h, const VertexT &pos) {
            h[0] = h[1] = 1 ;
          },
          1, util::DEVICE, oprtr_parameters.stream));

	   retval = gunrock::oprtr::For([compute_lockfree_op, r,type] __host__ __device__(const int &i) { compute_lockfree_op(r,type, i); },
                  loop_size, util::DEVICE, oprtr_parameters.stream);
    }

  debug_aml("[%d]frontier que length after compute op is %d\n", iteration,
              frontier.queue_length);

    // GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
    //        "cudaStreamSynchronize failed");
    cudaFree(d_degrees);
    cudaFreeHost(incrDegrees);

    active.Move(util::DEVICE, util::HOST, 2, 0, oprtr_parameters.stream);

    GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
              "cudaStreamSynchronize failed");

    if (active[0] == 0 && active[1] == 0) {
      auto end = time(NULL);
      en = high_resolution_clock::now();
      std::ofstream fout("test.txt", std::ios_base::app);
      fout<<duration_cast<milliseconds>(en - st).count()<<"\n";
      std::cout << "The  time is "<<end - ifexcess[0]<<" "<<duration_cast<milliseconds>(en - st).count()<<"\n";
      //st = high_resolution_clock::now();
      ifexcess[0] = end;
      if (ctr[0]<0){      // make zero for no input del file.
        if (ctr[0] == 0) st = high_resolution_clock::now();
        excess.Move(util::DEVICE, util::HOST, 1, sink, oprtr_parameters.stream);
        GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
        "cudaStreamSynchronize failed");
        printf("The flow currently is %f\n",excess[sink]);
        /*
        GUARD_CU(oprtr::For(
        [residuals, e,reverse, excess,u1,v1,capacity, source, sink] __host__ __device__(const SizeT &e1) {
          excess[u1] +=  capacity[e] - residuals[e];
          excess[v1] += -capacity[e] + residuals[e];
          residuals[reverse[e]] -= capacity[e] - residuals[e];
          residuals[e] = 0;
          
          capacity[e] = 0;
        },
        1, util::DEVICE, oprtr_parameters.stream));*/
        std::ifstream fin("del"+std::to_string(ctr[0]));++ctr[0];
    int no; fin>>no;
    std::cout <<no<<std::endl;
    int u1 = 1,v1 = 2;
    for (int i = 0; i < no; ++i){
        VertexT e;
        ValueT new_capacity;
        fin>>u1>>v1>>new_capacity;
        --u1;--v1;
        e = graph.CsrT::GetNeighborListOffset(u1);
        while (1){
          if (graph.CsrT::GetEdgeDest(e) == v1){
            break;
          }
          ++e;
        }
        height_indirection[i] = u1;
        height_indirection[no+i] = v1;
        height_indirection[2*no+i] = e;
        height_indirection[3*no+i] = new_capacity;
    }
        /*GUARD_CU(oprtr::For(
          [residuals, e,reverse, excess,u1,v1,capacity, new_capacity] __host__ __device__(const SizeT &e1) {
            auto f = capacity[e] - residuals[e];
            if (new_capacity < capacity[e]){
              if (f > new_capacity){
                excess[u1] += f - new_capacity;
                excess[v1] += new_capacity - f;
                residuals[reverse[e]] -= f - new_capacity;
                f = new_capacity;
              }
              capacity[e] = new_capacity;
              residuals[e] = new_capacity - f;
            }
            else if (new_capacity > capacity[e]){
              if (residuals[e] == 0){
                excess[u1] += capacity[e] - new_capacity;
                excess[v1] += new_capacity - capacity[e];
                residuals[reverse[e]] -= capacity[e] - new_capacity;
                f = new_capacity;
              }
              capacity[e] = new_capacity;
              residuals[e] = new_capacity - f;
            }
          },
          1, util::DEVICE, oprtr_parameters.stream));*/
          height_indirection.Move(util::HOST, util::DEVICE, no*4, 0,oprtr_parameters.stream);
          GUARD_CU(oprtr::For(
            [residuals,reverse, excess,capacity, height_indirection,no] __host__ __device__(const SizeT &i) {
              auto e = height_indirection[2*no + i], new_capacity = height_indirection[3*no+i];
              auto u1 = height_indirection[i], v1 = height_indirection[no+i] ;
              auto f = capacity[e] - residuals[e];
              if (new_capacity < capacity[e]){
                if (f > new_capacity){
                  atomicAdd(excess+u1, f - new_capacity);
                  atomicAdd(excess+v1, new_capacity - f);
                  residuals[reverse[e]] -= f - new_capacity;
                  f = new_capacity;
                }
                capacity[e] = new_capacity;
                residuals[e] = new_capacity - f;
              }
              else if (new_capacity > capacity[e]){
                if (residuals[e] == 0){
                  atomicAdd(excess+u1, f - new_capacity);
                  atomicAdd(excess+v1, new_capacity - f);
                  residuals[reverse[e]] -= f - new_capacity;
                  f = new_capacity;
                }
                capacity[e] = new_capacity;
                residuals[e] = new_capacity - f;
              }
            },
            no, util::DEVICE, oprtr_parameters.stream));
    
    active[0] = active[1] = 1;
      }
      else{
      GUARD_CU(oprtr::For(
          [residuals, capacity, flow] __host__ __device__(const SizeT &e) {
            flow[e] = capacity[e] - residuals[e];
          },
          graph.edges, util::DEVICE, oprtr_parameters.stream));
        }
      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
                "cudaStreamSynchronize failed");
    }
    return retval;
  }

  /**
  * @brief Routine to combine received data and local data
  * @tparam NUM_VERTEX_ASSOCIATES  Number of data associated with each
  *				      transmition item, typed VertexT
  * @tparam NUM_VALUE__ASSOCIATES  Number of data associated with each
                                    transmition item, typed ValueT
  * @param[in] received_length     The number of transmition items received
  * @param[in] peer_		      which peer GPU the data came from
  * \return cudaError_t error message(s), if any
  */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t ExpandIncoming(SizeT &received_length, int peer_) {
    auto &enactor = this->enactor;
    auto &problem = enactor->problem;
    auto gpu_num = this->gpu_num;
    auto gpu_offset = gpu_num * enactor->num_gpus;
    auto &data_slice = problem->data_slices[gpu_num][0];
    auto &enactor_slice = enactor->enactor_slices[gpu_offset + peer_];
    auto iteration = enactor_slice.enactor_stats.iteration;

    auto &capacity = data_slice.sub_graph[0].edge_values;
    auto &flow = data_slice.flow;
    auto &excess = data_slice.excess;
    auto &height = data_slice.height;

    /*	for key " +
                        std::to_string(key) + " and for in_pos " +
                        std::to_string(in_pos) + " and for vertex ass ins " +
                        std::to_string(vertex_associate_ins[in_pos]) +
                        " and for value ass ins " +
                        std::to_string(value__associate_ins[in_pos]));*/

    auto expand_op = [capacity, flow, excess, height] __host__ __device__(
                        VertexT & key, const SizeT &in_pos,
                        VertexT *vertex_associate_ins,
                        ValueT *value__associate_ins) -> bool {
      // TODO: fill in the lambda to combine received and local data, e.g.:
      // ValueT in_val  = value__associate_ins[in_pos];
      // ValueT old_val = atomicMin(distances + key, in_val);
      // if (old_val <= in_val)
      //     return false;
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                      NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto enactor = this->enactor;
    int num_gpus = enactor->num_gpus;
    auto &data_slice = enactor->problem->data_slices[gpu_num][0];
    auto &enactor_slice = enactor->enactor_slices[0];
    auto &retval = enactor_slice.enactor_stats.retval;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;

    if (retval != cudaSuccess) {
      printf("(CUDA error %d @ GPU %d: %s\n", retval, 0 % num_gpus,
            cudaGetErrorString(retval));
      fflush(stdout);
      return true;
    }

    // if (enactor_slice.enactor_stats.iteration > 1)
    //    return true;
    if (data_slice.active[0] > 0 || data_slice.active[1] > 0) return false;
    return true;
  }

};  // end of MFIteration

/* MF enactor class.
* @tparam _Problem Problem type we process on
* @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
* @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
*/
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<typename _Problem::GraphT, typename _Problem::VertexT,
                        typename _Problem::ValueT, ARRAY_FLAG,
                        cudaHostRegisterFlag> {
public:
  typedef _Problem Problem;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::ValueT ValueT;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::GraphT GraphT;
  typedef EnactorBase<GraphT, VertexT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;

  typedef MFIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
  * @brief MFEnactor constructor
  */
  Enactor() : BaseEnactor("mf"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
  * @brief MFEnactor destructor
  */
  virtual ~Enactor() {
    // Release();
  }

  /*
  * @brief Releasing allocated memory space
  * @param target The location to release memory from
  * \return cudaError_t error message(s), if any
  */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Release(target));
    delete[] iterations;
    iterations = NULL;
    problem = NULL;
    return retval;
  }

  /**
  * \addtogroup PublicInterface
  * @{
  */

  /**
  * @brief Initialize the problem.
  * @param[in] problem The problem object.
  * @param[in] target Target location of data
  * \return cudaError_t error message(s), if any
  */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

    // Lazy initialization
    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 2, NULL, target, false));

    auto num_gpus = this->num_gpus;

    for (int gpu = 0; gpu < num_gpus; ++gpu) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      auto gpu_offset = gpu * num_gpus;
      auto &enactor_slice = this->enactor_slices[gpu_offset + 0];
      auto &graph = problem.sub_graphs[gpu];
      auto nodes = graph.nodes;
      auto edges = graph.edges;
      GUARD_CU(
          enactor_slice.frontier.Allocate(nodes, edges, this->queue_factors));
    }

    iterations = new IterationT[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      GUARD_CU(iterations[gpu].Init(this, gpu));
    }

    GUARD_CU(this->Init_Threads(
        this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));
    return retval;
  }

  /**
  * @brief one run of mf, to be called within GunrockThread
  * @param thread_data Data for the CPU thread
  * \return cudaError_t error message(s), if any
  */
  cudaError_t Run(ThreadSlice &thread_data) {
    debug_aml("Run enact\n");
    gunrock::app::Iteration_Loop<0, 1, IterationT>(
        thread_data, iterations[thread_data.thread_num]);

    return cudaSuccess;
  }

  /**
  * @brief Reset enactor
  * @param[in] src Source node to start primitive.
  * @param[in] target Target location of data
  * \return cudaError_t error message(s), if any
  */
  cudaError_t Reset(const VertexT &src, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    debug_aml("Enactor Reset, src %d\n", src);

    typedef typename GraphT::GpT GpT;

    GUARD_CU(BaseEnactor::Reset(target));

    SizeT nodes = this->problem->data_slices[0][0].sub_graph[0].nodes;

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if (this->num_gpus == 1) {
        this->thread_slices[gpu].init_size = nodes;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? nodes : 0;
          if (peer_ == 0) {
            util::Array1D<SizeT, VertexT> tmp;
            tmp.Allocate(nodes, target | util::HOST);
            for (SizeT i = 0; i < nodes; ++i) {
              tmp[i] = (VertexT)i % nodes;
            }
            GUARD_CU(tmp.Move(util::HOST, target));

            GUARD_CU(frontier.V_Q()->ForEach(
                tmp,
                [] __host__ __device__(VertexT & v, VertexT & i) { v = i; },
                nodes, target, 0));

            tmp.Release();
          }
        }
      } else {
        // MULTIGPU INCOMPLETE
      }
    }
    debug_aml("Enactor Reset end\n");
    GUARD_CU(BaseEnactor::Sync())
    return retval;
  }

  /**
  * @brief Enacts a MF computing on the specified graph.
  * @param[in] src Source node to start primitive.
  * \return cudaError_t error message(s), if any
  */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    debug_aml("enact\n");
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU MF Done.", this->flag & Debug);
    return retval;
  }

  /** @} */
};

}  // namespace mf
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

