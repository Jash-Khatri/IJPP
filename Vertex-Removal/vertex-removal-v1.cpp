#include<bits/stdc++.h>
#include<stdlib.h>
#include<iostream>
#include <chrono>
using namespace std::chrono;
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
		vector <int> v(3);
		fscanf(filePointer, "%d", &v[0] );
		fscanf(filePointer, "%d", &v[1] );
		//fscanf(filePointer, "%d", &v[2] );	// comment for the undirected graph
		edge_pairs.push_back(v);
	}
	
	fclose(filePointer);
	
	  int isleft0deg = 1;
  	  int rv;
  	  int k=0;
	  int NUM_ITER_FIXPOINT = 1000;
	  vector <vector <int>> new_edge_pairs;
	  vector <int> empty_vertices;	
	  int total_removed = 0;
	  //int total_v_left = n;

	/*
	for(int i=0;i<edge_pairs.size();i++)
			cout << edge_pairs[i][0] << " " << edge_pairs[i][1] << "\n";

	std::vector <int> degree( n + 1, 0);
	        for(int i=0;i<edge_pairs.size();i++)
		{
		        degree[edge_pairs[i][0]]++;
		        degree[edge_pairs[i][1]]++;		// for undirected graphs, comment this line if graph is directed
		}
	for(int i=0;i<degree.size();i++)
		cout << degree[i] << " ";
	cout << "\n"; 
	*/
	
	// Get starting timepoint
    	  auto start = high_resolution_clock::now();
	
	
	  while(isleft0deg)
	  {
	        isleft0deg = 0;
		rv=0;
		  
	       if( k == NUM_ITER_FIXPOINT )
	          break;
	
	       k++;

	        std::vector <int> degree( n + 1, 0);
	        for(int i=0;i<edge_pairs.size();i++)
		{
		        degree[edge_pairs[i][0]]++;
		        degree[edge_pairs[i][1]]++;		// for undirected graphs, comment this line if graph is directed
		}

	        for(int i=0;i<edge_pairs.size();i++)
		{
	                if( ( degree[edge_pairs[i][0]] > 1 && degree[edge_pairs[i][1]] > 1 ) || ( edge_pairs[i][0] == n || edge_pairs[i][1] == n ) || ( edge_pairs[i][0] == (n-1) || edge_pairs[i][1] == (n-1) ) || ( edge_pairs[i][0] == 1 || edge_pairs[i][1] == 1 ) )
			{
			vector <int> arr(3);
	                arr[0] = edge_pairs[i][0];
	                arr[1] = edge_pairs[i][1];
			//arr[2] = edge_pairs[i][2];		// comment for the undirected graph
			arr[2] = 1;			
			new_edge_pairs.push_back(arr);
	                }
                	else
			{
                	isleft0deg = 1;
			rv++;

			if( degree[edge_pairs[i][0]] <= 1 && degree[edge_pairs[i][1]] <= 1  )
			{
				empty_vertices.push_back( edge_pairs[i][0] );
				empty_vertices.push_back( edge_pairs[i][1] );	
				total_removed += 2;	
			}

			else if( degree[edge_pairs[i][0]] <= 1 )
			{
				empty_vertices.push_back( edge_pairs[i][0] );
				total_removed++;
			}
			
			else if( degree[edge_pairs[i][1]] <= 1 )
			{
				empty_vertices.push_back( edge_pairs[i][1] );
				total_removed++;
			}	

	                }
	        }
		//for(int i=0;i<new_edge_pairs.size();i++)
		//	cout << new_edge_pairs[i][0] << " " << new_edge_pairs[i][1] << "\n";
		edge_pairs.clear();
		edge_pairs = new_edge_pairs;
		new_edge_pairs.clear();
		std::cout << "iteration " << k << " " << rv << "\n";
	 } // end of while loop
	
	std::cout << " Sorting the dead vertices\n ";

	//remove_duplicates(empty_vertices);
	set <int> st( empty_vertices.begin(), empty_vertices.end() );
	auto index = st.begin();	
	unordered_map <int,int> mp;

	/*	
	// print for checking purposes
	for(int i=0;i<empty_vertices.size();i++)
		cout << empty_vertices[i] << " ";
	cout << "\n";
	*/
	std::cout << " Removing the dead vertices\n ";

	if( st.size() > 0 )
	{
		int curr_vertex = *index;

		while( curr_vertex <= n && index != st.end() )
		{
			curr_vertex++;
			
			while( st.find( curr_vertex ) != st.end() && curr_vertex <= n )
				curr_vertex++;
			
			if( curr_vertex % 10000 == 0 )
                                cout << curr_vertex << " " <<  *index << "\n";

			if( curr_vertex <= n && index != st.end() )
			{
				mp[curr_vertex] = *index;
				st.insert( curr_vertex );
				index++;
			}
			//cout << curr_vertex << " \n"; 		
		}
	}

	//for(auto itr = mp.begin(); itr != mp.end(); itr++)
	//	cout << itr->first << " " << itr->second << "\n";

	std::cout << "Fixing the original graph\n ";

	for(int i=0;i<edge_pairs.size();i++)
	{
		if( mp.find( edge_pairs[i][0] ) != mp.end() )
			edge_pairs[i][0] = mp[ edge_pairs[i][0] ];

		 if( mp.find( edge_pairs[i][1] ) != mp.end() )
			edge_pairs[i][1] = mp[ edge_pairs[i][1] ];
	}

	// Get ending timepoint
    	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(stop - start);
  
       std::cout << "\nTime taken by vertex removal: "  << duration.count() << " microseconds" << std::endl;
	

	filePointer = fopen( filename , "w") ; 	
	int mhat = edge_pairs.size();
	int nhat = n - total_removed; 
	
	// added new loop here
	for(auto i=0;i<comments.size();i++)
	{
		int size = comments[i].size();
		char arr[size+1];
		strcpy(arr, comments[i].c_str());
		//if( i != comments.size()-1 )
		//	fprintf(filePointer, "%s \n", arr );	
		//else
		fprintf(filePointer, "%s", arr );
	}
	
	fprintf(filePointer, "%d ", nhat );		//scaning the number of vertices
	fprintf(filePointer, "%d ", nhat );		//scaning the number of vertices
        fprintf(filePointer, "%d\n", mhat );		//scaning the number of edges

	for(int i=0;i<mhat;i++){
		fprintf(filePointer, "%d %d %d \n", edge_pairs[i][0], edge_pairs[i][1], edge_pairs[i][2] );
	}

	fclose(filePointer);

return 0;
}

