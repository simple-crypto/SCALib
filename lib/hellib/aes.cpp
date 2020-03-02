#include "aes.h"

#define nb_Rounds 10
#define nb_word 16

const unsigned char table2[256]= {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 27, 25, 31, 29, 19, 17, 23, 21, 11, 9, 15, 13, 3, 1, 7, 5, 59, 57, 63, 61, 51, 49, 55, 53, 43, 41, 47, 45, 35, 33, 39, 37, 91, 89, 95, 93, 83, 81, 87, 85, 75, 73, 79, 77, 67, 65, 71, 69, 123, 121, 127, 125, 115, 113, 119, 117, 107, 105, 111, 109, 99, 97, 103, 101, 155, 153, 159, 157, 147, 145, 151, 149, 139, 137, 143, 141, 131, 129, 135, 133, 187, 185, 191, 189, 179, 177, 183, 181, 171, 169, 175, 173, 163, 161, 167, 165, 219, 217, 223, 221, 211, 209, 215, 213, 203, 201, 207, 205, 195, 193, 199, 197, 251, 249, 255, 253, 243, 241, 247, 245, 235, 233, 239, 237, 227, 225, 231, 229};

const unsigned char SBOX[256]  =
{
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};


const unsigned char Rcon[10] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};




void aes(unsigned char* MK,unsigned char* plain,unsigned char* cipher)
{

	unsigned char k[176];
	unsigned char state[16];
	unsigned char tmp[16];
	unsigned char sum[4];

    memcpy(k,MK,16*sizeof(unsigned char));
    memcpy(state,plain,16*sizeof(unsigned char));

	unsigned char i,r;
    for(i=0; i<nb_word; i++)
    {
        state[i]= state[i]^k[i];
    }
	
	
    for(r=0; r<nb_Rounds-1; r++)
    {
        ///Sbox+shiftrows
        memcpy(tmp,state,16*sizeof(unsigned char));
	
	state[0]=SBOX[tmp[0]];
	state[1]=SBOX[tmp[5]];
	state[2]=SBOX[tmp[10]];
	state[3]=SBOX[tmp[15]];
	state[4]=SBOX[tmp[4]];
	state[5]=SBOX[tmp[9]];
	state[6]=SBOX[tmp[14]];
	state[7]=SBOX[tmp[3]];
	state[8]=SBOX[tmp[8]];
	state[9]=SBOX[tmp[13]];
	state[10]=SBOX[tmp[2]];
	state[11]=SBOX[tmp[7]];
	state[12]=SBOX[tmp[12]];
	state[13]=SBOX[tmp[1]];
	state[14]=SBOX[tmp[6]];
	state[15]=SBOX[tmp[11]];
	
///Mixcolumns


        for(i=0; i<4; i++)
            sum[i]=state[i*4]^state[i*4+1]^state[i*4+2]^state[i*4+3];

        for(i=0; i<4; i++)
        {
            tmp[i]=table2[state[i]^state[(i+1)%4]];
            tmp[i+4]=table2[state[i+4]^state[4+(i+1)%4]];
            tmp[i+8]=table2[state[i+8]^state[8+(i+1)%4]];
            tmp[i+12]=table2[state[i+12]^state[12+(i+1)%4]];

        }
        for( i=0; i<16; i++)
            state[i]=tmp[i]^state[i]^sum[i/4];


///Keyschedule
///sbox



        tmp[3]=SBOX[k[16*r+12]];
        tmp[0]=SBOX[k[16*r+12+1]]^Rcon[r];
        tmp[1]=SBOX[k[16*r+12+2]];
        tmp[2]=SBOX[k[16*r+12+3]];


        for( i=0; i<4; i++)
            k[16*(r+1)+i]=tmp[i]^k[16*(r)+i];


        for( i=0; i<12; i++)
            k[16*(r+1)+i+4]=k[16*(r+1)+i]^k[16*(r)+i+4];

        ///ADD Round key
        for( i=0; i<nb_word; i++)
        {
            state[i]=state[i]^k[i+16*(r+1)];
        }


    }

	    
	memcpy(tmp,state,16*sizeof(unsigned char));


	state[0]=SBOX[tmp[0]];
	state[1]=SBOX[tmp[5]];
	state[2]=SBOX[tmp[10]];
	state[3]=SBOX[tmp[15]];
	state[4]=SBOX[tmp[4]];
	state[5]=SBOX[tmp[9]];
	state[6]=SBOX[tmp[14]];
	state[7]=SBOX[tmp[3]];
	state[8]=SBOX[tmp[8]];
	state[9]=SBOX[tmp[13]];
	state[10]=SBOX[tmp[2]];
	state[11]=SBOX[tmp[7]];
	state[12]=SBOX[tmp[12]];
	state[13]=SBOX[tmp[1]];
	state[14]=SBOX[tmp[6]];
	state[15]=SBOX[tmp[11]];
///Keyschedule
///sbox



        tmp[3]=SBOX[k[16*r+12]];
        tmp[0]=SBOX[k[16*r+12+1]]^Rcon[r];
        tmp[1]=SBOX[k[16*r+12+2]];
        tmp[2]=SBOX[k[16*r+12+3]];


        for(i=0; i<4; i++)
            k[16*(r+1)+i]=tmp[i]^k[16*(r)+i];


        for(i=0; i<12; i++)
            k[16*(r+1)+i+4]=k[16*(r+1)+i]^k[16*(r)+i+4];
    ///ADD Round key
    for(i=0; i<nb_word; i++)
    {
        cipher[i]=state[i]^k[i+16*(r+1)];
    }

}

