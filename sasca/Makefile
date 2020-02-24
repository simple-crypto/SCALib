BDIR = build
SRCDIR = libbp
object:= $(BDIR)/utils.o $(BDIR)/libbp.o $(BDIR)/graph.o $(BDIR)/graph_utils.o

.PHONY: lib 
OPTFLAG:= -DERROR_FACTOR=1E-20 -DZERO=1E-50
CFLAGS+= -std=gnu99 -g -O3 -lm -lpthread  $(OPTFLAG) 

$(BDIR)/%.o:$(SRCDIR)/%.c
	mkdir -p $(BDIR)
	gcc $(CFLAGS) -fPIC  -c $< -o $@
lib: $(object)
	gcc $(CFLAGS) -shared -fPIC $(object) -o libbp.so

clean:
	rm -rf $(BDIR) *.so exec

