#include <hls_math.h>

//stories15M.bin
#define P_DIM 288
#define P_HIDDEN_DIM 768
#define P_N_LAYERS 6
#define P_N_HEADS 6
#define P_VOCAB_SIZE 32000
#define P_SEQ_LEN 256
#define P_HEAD_SIZE 48
#define P_KV_DIM 288
#define INV_P_DIM 0.0034722222f
#define INV_SQRT_P_HEAD_SIZE 0.14433757f

void rmsnorm(float* o, float* x, float* w) {
	float W[P_DIM];
	#pragma HLS ARRAY_PARTITION variable=W cyclic factor=32 dim=1

	read_w:
	for (int i = 0; i < P_DIM; i++) {
		#pragma HLS UNROLL factor=8
		W[i] = w[i];
	}

    float partial[4] = {0};
    #pragma HLS ARRAY_PARTITION variable=partial complete
    
    calculate_ss:
    for (int i = 0; i < P_DIM; i++) {
    	#pragma HLS PIPELINE II=1
    	partial[i % 4] += x[i] * x[i];
    }
    
    float ss = partial[0] + partial[1] + partial[2] + partial[3];
    ss *= INV_P_DIM;
    ss += 1e-5f;
    ss = hls::rsqrt(ss);

	calculate_norm:
    for (int i = 0; i < P_DIM; i++) {
    	#pragma HLS UNROLL factor=8
        o[i] = W[i] * (ss * x[i]);
    }
}

void softmax(float* x, int size) {
	float max_val = x[0];
    
    find_max:
    for (int i = 1; i < size; i++) {
    	#pragma HLS LOOP_TRIPCOUNT min=1 max=257
    	#pragma HLS PIPELINE II=1
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    float partial[4] = {0};
    #pragma HLS ARRAY_PARTITION variable=partial complete
    
    calculate_sum:
    for (int i = 0; i < size; i++) {
    	#pragma HLS LOOP_TRIPCOUNT min=1 max=257
    	#pragma HLS PIPELINE II=1
        x[i] = hls::expf(x[i] - max_val);
        partial[i % 4] += x[i];
    }
    
    float sum = partial[0] + partial[1] + partial[2] + partial[3];
    float inv_sum = 1.0f / sum;
    
    calculate_norm:
    for (int i = 0; i < size; i++) {
    	#pragma HLS LOOP_TRIPCOUNT min=1 max=257
    	#pragma HLS UNROLL factor=32
        x[i] *= inv_sum;
    }
}

void matmul_dim_dim(float* o, float* x, float* w) {
    calculate_d:
    for (int i = 0; i < P_DIM; i++) {
        #pragma HLS loop_flatten off

    	float partial[4] = {0};
    	#pragma HLS ARRAY_PARTITION variable=partial complete
    	
    	calculate_n:
        for (int j = 0; j < P_DIM; j++) {
        	#pragma HLS PIPELINE II=1
            partial[j % 4] += w[i * P_DIM + j] * x[j];
        }
        
        o[i] = partial[0] + partial[1] + partial[2] + partial[3];
    }
}

void matmul_dim_kvdim(float* o, float* x, float* w) {
	calculate_d:
    for (int i = 0; i < P_DIM; i++) {
        #pragma HLS loop_flatten off

    	float partial[4] = {0};
    	#pragma HLS ARRAY_PARTITION variable=partial complete
    	
    	calculate_n:
        for (int j = 0; j < P_DIM; j++) {
        	#pragma HLS PIPELINE II=1
            partial[j % 4] += w[i * P_DIM + j] * x[j];
        }
        
        o[i] = partial[0] + partial[1] + partial[2] + partial[3];
    }
}

void matmul_dim_hiddendim(float* o, float* x, float* w) {
	calculate_d:
    for (int i = 0; i < P_HIDDEN_DIM; i++) {
        #pragma HLS loop_flatten off

    	float partial[4] = {0};
    	#pragma HLS ARRAY_PARTITION variable=partial complete
    	
    	calculate_n:
        for (int j = 0; j < P_DIM; j++) {
        	#pragma HLS PIPELINE II=1
            partial[j % 4] += w[i * P_DIM + j] * x[j];
        }
        
        o[i] = partial[0] + partial[1] + partial[2] + partial[3];
    }
}

void matmul_hiddendim_dim(float* o, float* x, float* w) {
    calculate_d:
    for (int i = 0; i < P_DIM; i++) {
        #pragma HLS loop_flatten off

    	float partial[4] = {0};
    	#pragma HLS ARRAY_PARTITION variable=partial complete
    	
    	calculate_n:
        for (int j = 0; j < P_HIDDEN_DIM; j++) {
        	#pragma HLS PIPELINE II=1
            partial[j % 4] += w[i * P_HIDDEN_DIM + j] * x[j];
        }
        
        o[i] = partial[0] + partial[1] + partial[2] + partial[3];
    }
}

void matmul_dim_vocabsize(float* o, float* x, float* w) {
	calculate_d:
    for (int i = 0; i < P_VOCAB_SIZE; i++) {
        #pragma HLS loop_flatten off

        float partial[4] = {0};
    	#pragma HLS ARRAY_PARTITION variable=partial complete
    	
    	calculate_n:
        for (int j = 0; j < P_DIM; j++) {
        	#pragma HLS PIPELINE II=1
            partial[j % 4] += w[i * P_DIM + j] * x[j];
        }
        
        o[i] = partial[0] + partial[1] + partial[2] + partial[3];
    }
}

void RoPE(float* TABLE, float* S_q, float* S_k) {
    float S_K[P_DIM];
    #pragma HLS ARRAY_PARTITION variable=S_K cyclic factor=32 dim=1
    
    read_sk:
    for (int i = 0; i < P_DIM; i++) {
        #pragma HLS UNROLL factor=8
        S_K[i] = S_k[i];
    }
    
    RoPE:
    for (int i = 0; i < P_DIM; i+=2) {
    	#pragma HLS UNROLL factor=16
        float fcr = TABLE[i];
        float fci = TABLE[i+1];
        
        if (i < P_KV_DIM) {
            float v0_q = S_q[i];
            float v1_q = S_q[i+1];
            S_q[i]   = v0_q * fcr - v1_q * fci;
            S_q[i+1] = v0_q * fci + v1_q * fcr;

            float v0_k = S_K[i];
            float v1_k = S_K[i+1];
            S_K[i]   = v0_k * fcr - v1_k * fci;
            S_K[i+1] = v0_k * fci + v1_k * fcr;
        } 
        else {
            float v0_q = S_q[i];
            float v1_q = S_q[i+1];
            S_q[i]   = v0_q * fcr - v1_q * fci;
            S_q[i+1] = v0_q * fci + v1_q * fcr;
        }
    }
    
    write_sk:
    for (int i = 0; i < P_DIM; i++) {
        #pragma HLS UNROLL factor=8
        S_k[i] = S_K[i];
    }
}

void attention(int loff, int pos, float* S_xb, float* S_q, float* S_att, float* S_key_cache, float* S_value_cache) {
    attention_heads:
    for (int h = 0; h < P_N_HEADS; h++) {
        float* q = S_q + h * P_HEAD_SIZE;
        float* att = S_att + h * P_SEQ_LEN;

        compute_scores:
        for (int t = 0; t <= pos; t++) {
            #pragma HLS loop_flatten off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=257
            
            float* k = S_key_cache + loff + h * P_HEAD_SIZE + t * P_KV_DIM;

            float partial[4] = {0};
            #pragma HLS ARRAY_PARTITION variable=partial complete
            
            dot_qk:
            for (int i = 0; i < P_HEAD_SIZE; i++) {
                #pragma HLS PIPELINE II=1
                partial[i % 4] += q[i] * k[i];
            }

            float score = partial[0] + partial[1] + partial[2] + partial[3];
            score *= INV_SQRT_P_HEAD_SIZE;
            att[t] = score;
        }

        softmax(att, pos + 1);

        float* xb = S_xb + h * P_HEAD_SIZE;
        
        float acc[P_HEAD_SIZE];
        #pragma HLS ARRAY_PARTITION variable=acc cyclic factor=32 dim=1
        
        init:
        for (int i = 0; i < P_HEAD_SIZE; i++) {
            #pragma HLS UNROLL factor=16
            xb[i] = 0.0f;
            acc[i] = 0.0f;
        }

        apply_attention:
        for (int t = 0; t <= pos; t++) {
        	#pragma HLS LOOP_TRIPCOUNT min=1 max=257
        	
            float* v = S_value_cache + loff + h * P_HEAD_SIZE + t * P_KV_DIM;

            float a = att[t];
            
            accumulate:
            for (int i = 0; i < P_HEAD_SIZE; i++) {
                #pragma HLS PIPELINE II=1
                acc[i] += a * v[i];
            }
        }
        
        write_xb:
        for (int i = 0; i < P_HEAD_SIZE; i++) {
            #pragma HLS UNROLL factor=16
            xb[i] = acc[i];
        }
    }
}


void residual(float* S_x, float* S_xb) {
    residual:
    for (int i = 0; i < P_DIM; i++) {
        #pragma HLS UNROLL factor=32
        S_x[i] += S_xb[i];
    }
}

void SwiGLU(float* S_hb, float* S_hb2) {
    SwiGLU:
    for (int i = 0; i < P_HIDDEN_DIM; i++) {
        #pragma HLS UNROLL factor=8
        S_hb[i] *= (1.0f / (1.0f + hls::expf(-S_hb[i])));
        S_hb[i] *= S_hb2[i];
    }
}

extern "C" {
void kernel_forward(int pos, int token, float* W_table, float* W_att, float* W_ffn, float* W_final, float* W_wcls, float* W_wq, float* W_wk, float* W_wv, 
    float* W_wo, float* W_w1, float* W_w2, float* W_w3, float* S_key_cache, float* S_value_cache, float* S_logits, float* table) {
    
    #pragma HLS INTERFACE m_axi port=W_table offset=slave bundle=gmem0 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port=W_att offset=slave bundle=gmem0 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port=W_ffn offset=slave bundle=gmem0 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port=W_final offset=slave bundle=gmem0 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port=W_wcls offset=slave bundle=gmem0 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port=W_wq offset=slave bundle=gmem1 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port=W_wk offset=slave bundle=gmem1 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port=W_wv offset=slave bundle=gmem1 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port=W_wo offset=slave bundle=gmem1 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port=W_w1 offset=slave bundle=gmem2 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port=W_w2 offset=slave bundle=gmem2 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port=W_w3 offset=slave bundle=gmem2 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port=S_key_cache offset=slave bundle=gmem3 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port=S_value_cache offset=slave bundle=gmem3 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port=S_logits offset=slave bundle=gmem3 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port=table offset=slave bundle=gmem1 max_read_burst_length=64
	
	#pragma HLS INTERFACE s_axilite port=W_table bundle=control
	#pragma HLS INTERFACE s_axilite port=W_att bundle=control
	#pragma HLS INTERFACE s_axilite port=W_ffn bundle=control
	#pragma HLS INTERFACE s_axilite port=W_final bundle=control
	#pragma HLS INTERFACE s_axilite port=W_wcls bundle=control
	#pragma HLS INTERFACE s_axilite port=W_wq bundle=control
	#pragma HLS INTERFACE s_axilite port=W_wk bundle=control
	#pragma HLS INTERFACE s_axilite port=W_wv bundle=control
	#pragma HLS INTERFACE s_axilite port=W_wo bundle=control
	#pragma HLS INTERFACE s_axilite port=W_w1 bundle=control
	#pragma HLS INTERFACE s_axilite port=W_w2 bundle=control
	#pragma HLS INTERFACE s_axilite port=W_w3 bundle=control
	#pragma HLS INTERFACE s_axilite port=S_key_cache bundle=control
	#pragma HLS INTERFACE s_axilite port=S_value_cache bundle=control
	#pragma HLS INTERFACE s_axilite port=S_logits bundle=control
	#pragma HLS INTERFACE s_axilite port=table bundle=control
	#pragma HLS INTERFACE s_axilite port=pos bundle=control
	#pragma HLS INTERFACE s_axilite port=token bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	
    float S_x[P_DIM], S_xb[P_DIM], S_xb2[P_DIM], S_hb[P_HIDDEN_DIM], S_hb2[P_HIDDEN_DIM], S_q[P_DIM], S_att[P_N_HEADS * P_SEQ_LEN];
	#pragma HLS ARRAY_PARTITION variable=S_x cyclic factor=32 dim=1
	#pragma HLS ARRAY_PARTITION variable=S_xb cyclic factor=32 dim=1
	#pragma HLS ARRAY_PARTITION variable=S_xb2 cyclic factor=32 dim=1
	#pragma HLS ARRAY_PARTITION variable=S_hb cyclic factor=32 dim=1
	#pragma HLS ARRAY_PARTITION variable=S_hb2 cyclic factor=32 dim=1
	#pragma HLS ARRAY_PARTITION variable=S_q cyclic factor=32 dim=1
    #pragma HLS ARRAY_PARTITION variable=S_att cyclic factor=32 dim=1
	
	token_embedding:
	for (int i = 0; i < P_DIM; i++) {
		#pragma HLS UNROLL factor=8
		S_x[i] = W_table[token * P_DIM + i];
	}
	
	float TABLE[P_DIM];
	#pragma HLS ARRAY_PARTITION variable=TABLE cyclic factor=32 dim=1
    
	read_table:
    for (int i = 0; i < P_DIM; i++) {
        #pragma HLS UNROLL factor=8
        TABLE[i] = table[i];
    }
    
    layer:
    for(int l = 0; l < P_N_LAYERS; l++) {
        int loff = l * P_SEQ_LEN * P_KV_DIM;
        float* S_k = S_key_cache + loff + pos * P_KV_DIM;
        float* S_v = S_value_cache + loff + pos * P_KV_DIM;

        rmsnorm(S_xb, S_x, W_att + l*P_DIM);
        matmul_dim_dim(S_q, S_xb, W_wq + l*P_DIM*P_DIM);
        matmul_dim_kvdim(S_k, S_xb, W_wk + l*P_DIM*P_KV_DIM);
        matmul_dim_kvdim(S_v, S_xb, W_wv + l*P_DIM*P_KV_DIM);
        
        RoPE(TABLE, S_q, S_k);
        attention(loff, pos, S_xb, S_q, S_att, S_key_cache, S_value_cache);
        matmul_dim_dim(S_xb2, S_xb, W_wo + l*P_DIM*P_DIM);
        residual(S_x, S_xb2);

        rmsnorm(S_xb, S_x, W_ffn + l*P_DIM);
        matmul_dim_hiddendim(S_hb, S_xb, W_w1 + l*P_DIM*P_HIDDEN_DIM);
        matmul_dim_hiddendim(S_hb2, S_xb, W_w3 + l*P_DIM*P_HIDDEN_DIM);
        
        SwiGLU(S_hb, S_hb2);
        matmul_hiddendim_dim(S_xb, S_hb, W_w2 + l*P_DIM*P_HIDDEN_DIM);
        residual(S_x, S_xb);
    }

    rmsnorm(S_x, S_x, W_final);
    matmul_dim_vocabsize(S_logits, S_x, W_wcls);
}
}