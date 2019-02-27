jpeg.o: jpeg/parse.c jpeg/compressed.h config.h
	cc jpeg/parse.c -c -o jpeg.o

main: main.c config.h jpeg.o
	cc main.c jpeg.o -o main

.PHONY: clean
clean: 
	rm *.o main
