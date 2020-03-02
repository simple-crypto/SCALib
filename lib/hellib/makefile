test: main_example.o hel_execute.o hel_enum.o hel_histo.o hel_init.o hel_util.o aes.o scores_example_data.o scores_example.o
	g++ -o test -O3 main_example.o scores_example.o scores_example_data.o hel_execute.o hel_enum.o hel_histo.o hel_init.o hel_util.o aes.o -lntl -lgmp -lm -lrt -shared -fPIC
	g++ -o hel_lib.so -O3 main_example.o scores_example.o scores_example_data.o hel_execute.o hel_enum.o hel_histo.o hel_init.o hel_util.o aes.o -lntl -lgmp -lm -lrt -shared -fPIC

main_example.o: main_example.cpp scores_example.h hel_execute.h hel_enum.h hel_histo.h hel_init.h hel_util.h hel_struct.h aes.h
	g++ -O3 -c  main_example.cpp -lntl -lgmp -lm -lrt -shared -fPIC 

hel_execute.o: hel_execute.cpp hel_execute.h hel_enum.h hel_histo.h hel_init.h hel_util.h hel_struct.h aes.h
	g++ -O3 -c  hel_execute.cpp -lntl -lgmp -lm -lrt -shared -fPIC

hel_enum.o: hel_enum.cpp hel_enum.h hel_histo.h hel_init.h hel_util.h hel_struct.h
	g++ -O3 -c  hel_enum.cpp -lntl -lgmp -lm -lrt -shared -fPIC

hel_histo.o: hel_histo.cpp hel_histo.h hel_init.h hel_util.h hel_struct.h
	g++ -O3 -c  hel_histo.cpp -lntl -lgmp -lm -lrt -shared -fPIC

hel_init.o: hel_init.cpp hel_init.h hel_util.h hel_struct.h
	g++ -O3 -c  hel_init.cpp -lntl -lgmp -lm -lrt -shared -fPIC

scores_example.o: scores_example.cpp scores_example_data.cpp scores_example.h hel_util.h hel_struct.h
	g++ -O3 -c  scores_example.cpp -lntl -lgmp -lm -lrt -shared -fPIC

scores_example_data.o: scores_example_data.cpp scores_example.h hel_util.h hel_struct.h
	g++ -O3 -c  scores_example_data.cpp -lntl -lgmp -lm -lrt -shared -fPIC

hel_util.o: hel_util.cpp hel_util.h hel_struct.h
	g++ -O3 -c  hel_util.cpp -lntl -lgmp -lm -lrt -shared -fPIC

aes.o: aes.cpp aes.h
	g++ -O3 -c  aes.cpp -lntl -lgmp -lm -lrt -shared -fPIC

clean: 
	rm -f test hel_util.o hel_execute.o hel_enum.o hel_histo.o hel_init.o main.o

