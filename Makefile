all:
	cd examples/mf; rm -rf bin; make; cd bin; mv test_mf* maxflow 
	unzip Benchmarks/Bench.zip; mv t*.mtx examples/mf/bin;
	pwd; cp Vertex-Removal/vertex-removal-v1.cpp examples/mf/bin; cp Vertex-Removal/vertex-removal-v2.cpp examples/mf/bin; cp Vertex-Removal/vertex-removal-directed.cpp examples/mf/bin;
	cd examples/mf/bin; g++ -std=c++11 -o vertex-removal vertex-removal-v1.cpp;  number=1 ; while [[ $$number -le 4 ]] ; do \
                ./vertex-removal t$$number.mtx > vr_graph_$$number.txt ; \
                ((number = number + 1)) ;\
	done
	cd examples/mf/bin; g++ -std=c++11 -o vertex-removal vertex-removal-v1.cpp;  number=8 ; while [[ $$number -le 9 ]] ; do \
                ./vertex-removal t$$number.mtx > vr_graph_$$number.txt ; \
                ((number = number + 1)) ;\
        done
	cd examples/mf/bin; g++ -std=c++11 -o vertex-removal vertex-removal-directed.cpp;  number=12 ; while [[ $$number -le 21 ]] ; do \
                ./vertex-removal t$$number.mtx > vr_graph_$$number.txt ; \
                ((number = number + 1)) ; \
        done
	cd examples/mf/bin; number=1 ; while [[ $$number -le 4 ]] ; do \
		./maxflow market t$$number.mtx -quick=true -jsondir=./ --undirected=true -num-runs=1 > output_$$number.txt ; \
		((number = number + 1)) ; \
	done
	cd examples/mf/bin; number=8 ; while [[ $$number -le 9 ]] ; do \
                ./maxflow market t$$number.mtx -quick=true -jsondir=./ --undirected=true -num-runs=1 > output_$$number.txt ; \
                ((number = number + 1)) ; \
        done
	cd examples/mf/bin; number=12 ; while [[ $$number -le 21 ]] ; do \
                ./maxflow market t$$number.mtx -quick=true -jsondir=./ -source=1 -num-runs=1 > output_$$number.txt ; \
                ((number = number + 1)) ; \
	done
	

#all:	
#for number in 1 2 3 4 ; do \ 
#	./test_mf_9.1_x86_64 market graph_1500_$$number.txt -quick=true -jsondir=./ -source=1 -num-runs=1 > output.txt ; \
#done
