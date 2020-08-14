
file_name = ""

def write_file(file_name,
                keyflag="secret",
                ptflag="public",
                indepk=False,
                nbytes=16):

    if not indepk:
        for b in range(nbytes):
            print("k%d #"%(b)+keyflag)

    print('\n#indeploop\n')
    for b in range(nbytes):
        if indepk:
            print("k%d #"%(b)+keyflag)
        print("p%d #"%(b)+ptflag)
        print("x%d #profile"%(b))
        print("x%d = k%d ^ p%d\n"%(b,b,b))
    print('#endindeploop')

if __name__ == "__main__":
    write_file("example.txt",ptflag="public",
            indepk=True)
