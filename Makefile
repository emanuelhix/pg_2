
test: build
	./run_loop_xform.x
build:
	gcc -std=c99 -g -O1 loop_xform_code_student.c  -o ./run_loop_xform.x

clean:
	rm -f *.x *~ *.o *.x
