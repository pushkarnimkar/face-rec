CFLAGS = 
debug: CFLAGS += -ggdb

huff.o: jpeg/huff.c jpeg/huff.h jpeg/enums.h
	$(CC) jpeg/huff.c $(CFLAGS) -c -o huff.o

quant.o: jpeg/quant.c jpeg/quant.h jpeg/enums.h
	$(CC) jpeg/quant.c $(CFLAGS) -c -o quant.o

sof.o: jpeg/sof.c jpeg/sof.h jpeg/enums.h
	$(CC) jpeg/sof.c $(CFLAGS) -c -o sof.o

sos.o: jpeg/sos.c jpeg/sos.h jpeg/enums.h
	$(CC) jpeg/sos.c $(CFLAGS) -c -o sos.o

coef.o: jpeg/coef.c jpeg/sos.h jpeg/coef.h jpeg/enums.h
	$(CC) jpeg/coef.c $(CFLAGS) -c -o coef.o

scan.o: jpeg/scan.c jpeg/scan.h jpeg/coef.h
	$(CC) jpeg/scan.c $(CFLAGS) -c -o scan.o

jpeg.o: jpeg/parse.c jpeg/compressed.h jpeg/enums.h config.h 
	$(CC) jpeg/parse.c $(CFLAGS) -c -o jpeg.o

math.o: maths/idct.c maths/idct.h
	$(CC) maths/idct.c $(CFLAGS) -c -o math.o

main: main.c config.h jpeg.o huff.o quant.o sof.o sos.o coef.o scan.o math.o jpeg/compressed.h
	$(CC) main.c jpeg.o huff.o quant.o sos.o sof.o coef.o scan.o math.o $(CFLAGS) -o main

.PHONY: clean
clean: 
	-rm main *.o

debug: clean main

