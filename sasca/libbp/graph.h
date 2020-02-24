#include <stdint.h>
#include <float.h>

#ifndef GRAPH_H
#define GRAPH_H

typedef double proba_t;// __attribute__ ((aligned(8)));
typedef double lproba_t;

struct Vnode{
    uint32_t    id;         // id
    uint32_t    Ni;         // functions outputing this node
    uint32_t    Nf;         // Number of function using this variable
    uint32_t    Ns;         // dimention of the distribution at this node
    uint32_t    update;     // that node needs to be update

    uint32_t*   relative;   // the relative within the function node input (of size Ni)
    uint32_t    id_input;   // id of input function node
    uint32_t*   id_output;  // id of output function node
    proba_t*    msg;     // message to pass
    proba_t*    distri_orig; // initial log distribution of the node
    proba_t*    distri; // actual distribution of the nodes
}typedef Vnode;

struct Fnode{
    uint32_t    id;         // id
    uint32_t    li;         // number of inputs
    uint32_t    has_offset; // Does function requires cst
    uint32_t    offset;     // constant
    uint32_t    func_id;    // fct code (ie 0 = AND, 2 == XOR)

    uint32_t*   i;          // list of input nodes ids
    uint32_t    o;          // output node id
    uint32_t*   relative;   // the position within each related nodes 
    proba_t*    msg;        // msg send to the vnodes index(0) = output
    uint32_t*   indexes[3]; // sorted indexes for and gates (to avoid worst case complexity)
} typedef Fnode;

void update_vnode(Vnode *vnode);
void update_fnode(Fnode *fnode);
void print_vnode(Vnode *vnode);

#endif
