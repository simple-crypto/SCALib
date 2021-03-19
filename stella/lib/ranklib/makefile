CXX_FLAGS=-O3 -lntl -lgmp
main: top.o main.o hel_execute.o hel_enum.o hel_histo.o hel_init.o hel_util.o aes.o scores_example_data.o scores_example.o
	g++ -o main  top.o main.o scores_example.o scores_example_data.o hel_execute.o hel_enum.o hel_histo.o hel_init.o hel_util.o aes.o $(CXX_FLAGS)
	#g++ -o main  main.o scores_example.o scores_example_data.o hel_execute.o hel_enum.o hel_histo.o hel_init.o hel_util.o aes.o $(CXX_FLAGS) 
test: main_example.o hel_execute.o hel_enum.o hel_histo.o hel_init.o hel_util.o aes.o scores_example_data.o scores_example.o
	g++ -o test  main_example.o scores_example.o scores_example_data.o hel_execute.o hel_enum.o hel_histo.o hel_init.o hel_util.o aes.o $(CXX_FLAGS) 
	
top.o: top.cpp scores_example.h hel_execute.h hel_enum.h hel_histo.h hel_init.h hel_util.h hel_struct.h aes.h
	g++  -c  top.cpp $(CXX_FLAGS) 

main.o: main.cpp
	g++  -c  main.cpp $(CXX_FLAGS) 

main_example.o: main_example.cpp scores_example.h hel_execute.h hel_enum.h hel_histo.h hel_init.h hel_util.h hel_struct.h aes.h
	g++  -c  main_example.cpp $(CXX_FLAGS) 

hel_execute.o: hel_execute.cpp hel_execute.h hel_enum.h hel_histo.h hel_init.h hel_util.h hel_struct.h aes.h
	g++  -c  hel_execute.cpp $(CXX_FLAGS) 

hel_enum.o: hel_enum.cpp hel_enum.h hel_histo.h hel_init.h hel_util.h hel_struct.h
	g++  -c  hel_enum.cpp $(CXX_FLAGS) 

hel_histo.o: hel_histo.cpp hel_histo.h hel_init.h hel_util.h hel_struct.h
	g++  -c  hel_histo.cpp $(CXX_FLAGS) 

hel_init.o: hel_init.cpp hel_init.h hel_util.h hel_struct.h
	g++  -c  hel_init.cpp $(CXX_FLAGS) 

scores_example.o: scores_example.cpp scores_example_data.cpp scores_example.h hel_util.h hel_struct.h
	g++  -c  scores_example.cpp $(CXX_FLAGS) 

scores_example_data.o: scores_example_data.cpp scores_example.h hel_util.h hel_struct.h
	g++  -c  scores_example_data.cpp $(CXX_FLAGS) 

hel_util.o: hel_util.cpp hel_util.h hel_struct.h
	g++  -c  hel_util.cpp $(CXX_FLAGS) 

aes.o: aes.cpp aes.h
	g++  -c  aes.cpp $(CXX_FLAGS) 

clean: 
	rm -f test hel_util.o hel_execute.o hel_enum.o hel_histo.o hel_init.o main.o

