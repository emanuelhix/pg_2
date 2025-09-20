
test: build
	./run_loop_xform.x
build:
	gcc -std=c99 -g -O1 code.c  -o ./code.x

clean:
	rm -f *.x *~ *.o *.x
