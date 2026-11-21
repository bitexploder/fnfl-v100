// draft_fmt: the format of the LM head rows (output.weight) the target head and the draft head copy read.
// q5_k  - default; ISTA-DASLab's IQ3_XXS file
// q8_0  - Unsloth's UD-Q4_K_XL file
// q6_k  - the Swift IQ3_S file (210 bytes a 256: dense::row_bytes)
#pragma once
#include <cstdint>

namespace fn { namespace head {

enum class draft_fmt : int32_t { q5_k, q8_0, q6_k };

} }
