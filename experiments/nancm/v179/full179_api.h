#pragma once
// Stable C ABI record returned by parallel complete-context verification.
// The generated words and optional sample come solely from the original
// trained v176 model; no approximation or independent learned head exists.
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct FullResult179 {
 float avg_log_byte[5];
 int32_t indices[5];
 float word_probabilities[5];
 char words[5][48];
 uint32_t verifier_calls;
 uint32_t sampled_bytes;
 uint32_t sampled_accepted;
 uint32_t sampled_rejected;
 unsigned char sample[40];
} FullResult179;
#ifdef __cplusplus
}
#endif
