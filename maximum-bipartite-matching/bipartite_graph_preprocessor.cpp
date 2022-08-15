/*
	This code figure out two sets of vertices s1 and s2 in the bipartite graph.
	It then create a new source vertex and adds the edges from source vertex to all vertices in s1.
	Similarly it creates a new sink vertex and adds the edges from sink vertex to all vertices in s2.
*/
#include<bits/stdc++.h>
#include<stdlib.h>
#include<iostream>
#include <chrono>
using namespace std::chrono;
using namespace std;

int main(int argc, char **argv)
{
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
	vector <string> comments;		// New line here

	while(true) {
		read = getline(&line, &len, filePointer);
        	//printf("Retrieved line of length %zu:\n", read);
        	//printf("%s", line);
		//cout << line[0] << " ";
		if(line[0] != '%')
			break;
		comments.push_back(line);		// New line here
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
	
	for(int i=0;i<m;i++)
	{
		vector <int> v(2);
		fscanf(filePointer, "%d", &v[0] );
		fscanf(filePointer, "%d", &v[1] );

		if( v[0] == v[1] )		// 1) ignore the duplicate edges
			continue;
		//fscanf(filePointer, "%d", &v[2] );	// comment for the undirected graph
		edge_pairs.push_back(v);
	}
	
	fclose(filePointer);

	cout << "reading done\n";

	unordered_set <int> s1;
	unordered_set <int> s2;

	// 2) when there is an edge (a,b) add 'a' to set1 and add 'b' to set2
	for(int i=0;i<(int)edge_pairs.size();i++){
		s1.insert(edge_pairs[i][0]);
		s2.insert(edge_pairs[i][1]);			
	}

	//long long int extra_edges=0;

	cout << "Sets populated\n";

	// 4) at the end add the edges from n+1 vertex to all vertices in set1
	for(auto itr = s1.begin() ; itr != s1.end() ; itr++){
		vector <int> v(2);
		v[0] = n+1;
		v[1] = *itr;
		edge_pairs.push_back(v);
		//extra_edges++;
	}

	// 5) from n+2 vertex to all vertices in set 2
	for(auto itr = s2.begin() ; itr != s2.end() ; itr++){
		vector <int> v(2);
		v[0] = n+2;
		v[1] = *itr;
		edge_pairs.push_back(v);
		//extra_edges++;	
	}

	n = n+2;			// 6) update the first line to N+2 N+2 E+EXTRA_ADDED_EDGES
	m = edge_pairs.size();

	filePointer = fopen( filename , "w"); 	

	fprintf(filePointer, "%d ", n );		//scaning the number of vertices
	fprintf(filePointer, "%d ", n );		//scaning the number of vertices
        fprintf(filePointer, "%d\n", m );		//scaning the number of edges

	for(int i=0;i<m;i++){
		fprintf(filePointer, "%d %d \n", edge_pairs[i][0], edge_pairs[i][1] );
	}

	fclose(filePointer);
}

