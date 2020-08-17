
file_name = ""

def write_file(file_name,
                keyflag="secret",
                ptflag="public",
                indepk=False,
                nbytes=16):
    with open(file_name, 'w') as f:

        if not indepk:
            for b in range(nbytes):
                print("k%d #"%(b)+keyflag,file=f)

        print('\n#indeploop\n',file=f)
        for b in range(nbytes):
            if indepk:
                print("k%d #"%(b)+keyflag,file=f)
            print("p%d #"%(b)+ptflag,file=f)
            print("x%d #profile"%(b),file=f)
            print("x%d = k%d ^ p%d\n"%(b,b,b),file=f)
        print('#endindeploop',file=f)

if __name__ == "__main__":
    write_file("example.txt",ptflag="public",
            indepk=True)
