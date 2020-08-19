
file_name = ""

def write_file(file_name,
                keyflag="secret",
                ptflag="public",
                indepk=False,
                nbytes=16):
    with open(file_name, 'w') as f:

        if not indepk:
            for b in range(nbytes):
                print("k_%d #"%(b)+keyflag,file=f)

        print('\n#indeploop\n',file=f)
        for b in range(nbytes):
            if indepk:
                print("k_%d #"%(b)+keyflag,file=f)
            print("p_%d #"%(b)+ptflag,file=f)
            print("x_%d #profile"%(b),file=f)
            print("x_%d = k_%d ^ p_%d\n"%(b,b,b),file=f)
        print('#endindeploop',file=f)

if __name__ == "__main__":
    write_file("example_graph.txt",ptflag="public",
            indepk=False,nbytes=16)
