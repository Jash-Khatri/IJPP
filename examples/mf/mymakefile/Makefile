all:
	number=1 ; while [[ $$number -le 11 ]] ; do \
		./test_mf_9.1_x86_64 market t$$number.mtx -quick=true -jsondir=./ --undirected=true -num-runs=1 > output_$$number.txt ; \
		((number = number + 1)) ; \
	done
	number=1 ; while [[ $$number -le 13 ]] ; do \
                ./test_mf_9.1_x86_64 market s$$number.mtx -quick=true -jsondir=./ -source=1 -num-runs=1 > output_s$$number.txt ; \
                ((number = number + 1)) ; \
	done
	number=15 ; while [[ $$number -le 18 ]] ; do \
                ./test_mf_9.1_x86_64 market s$$number.mtx -quick=true -jsondir=./ --undirected=true -num-runs=1 > output_s$$number.txt ; \
                ((number = number + 1)) ; \
        done

#all:	
#for number in 1 2 3 4 ; do \ 
#	./test_mf_9.1_x86_64 market graph_1500_$$number.txt -quick=true -jsondir=./ -source=1 -num-runs=1 > output.txt ; \
#done
