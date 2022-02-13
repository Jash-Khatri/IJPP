#include<bits/stdc++.h>
#include<stdlib.h>
#include<iostream>

using namespace std;

int main(int argc, char **argv){
	
	int n;	// number of vertices
	int m;	// number of edges

	//File pointer declaration
	FILE *filePointer;

	//File Opening for read
	char *filename = argv[1]; 
    	filePointer = fopen( filename , "r") ; 
      
	//checking if file ptr is NULL
    	if ( filePointer == NULL ) 
    	{
        printf( "input.txt file failed to open." ) ; 
	      return 0;
    	}

	char *line = NULL;
    	size_t len = 0;
	ssize_t read;

	// skip all the comment lines in mtx file
	int comment_lines = 0;

	while(true) {
		read = getline(&line, &len, filePointer);
        	//printf("Retrieved line of length %zu:\n", read);
        	//printf("%s", line);
		//cout << line[0] << " ";
		if(line[0] != '%')
			break;
		comment_lines++;
    	}

	//cout << comment_lines;

	fclose(filePointer);

	filePointer = fopen( filename , "r") ; 
      
	//checking if file ptr is NULL
    	if ( filePointer == NULL ) 
    	{
        printf( "input.txt file failed to open." ) ; 
	      return 0;
    	}

	int itr = comment_lines;
	while(itr > 0) {
		read = getline(&line, &len, filePointer);
        	printf("Retrieved line of length %zu:\n", read);
        	printf("%s", line);
		//cout << line[0] << " ";
		if(line[0] != '%')
			break;
		itr--;
    	}

	fscanf(filePointer, "%d", &n );		//scaning the number of vertices
	fscanf(filePointer, "%d", &n );		//scaning the number of vertices
        fscanf(filePointer, "%d", &m );		//scaning the number of edges

	

	cout << "num of vertices:" << n << " num of edges:" << m << "\n";

	vector <vector<int>> edge_pairs;
	int rand_seed1 = 40;			// tunable knob one
	int edge_cap = 2; 			// max edge capacity
	
	for(int i=0;i<rand_seed1;i++){
		vector <int> v(3,1);
		v[1] = (rand() % n) ;
		while(v[1] < 2 )
			v[1]++;
		v[2] = 1; // rand() % edge_cap;
		//cout << v[0] << " " << v[1] << "\n";
		edge_pairs.push_back(v);
	}

	for(int i=0;i<m;i++){
		vector <int> v(3);
		fscanf(filePointer, "%d", &v[0] );
		fscanf(filePointer, "%d", &v[1] );
		fscanf(filePointer, "%d", &v[2] );
		edge_pairs.push_back(v);
	}

	fclose(filePointer);

	filePointer = fopen( filename , "w") ; 	

	fprintf(filePointer, "%d ", n );		//scaning the number of vertices
	fprintf(filePointer, "%d ", n );		//scaning the number of vertices
        fprintf(filePointer, "%d\n", m+rand_seed1 );		//scaning the number of edges

	for(int i=0;i<edge_pairs.size();i++){
		fprintf(filePointer, "%d %d %d \n", edge_pairs[i][0], edge_pairs[i][1], edge_pairs[i][2] );
	}

	fclose(filePointer);

return 0;
}
