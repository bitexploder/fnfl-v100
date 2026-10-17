#include "llama.h"

#include "llama-impl.h"
#include "llama-version.h"

#include "llama-chat.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// FNFL (FL2): what is left of llama.cpp's interface implementation. The model loader, contexts and decoding are gone;
// FNFL owns the engine (fn/fnfl.h).

const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:
            return "auto";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED:
            return "disabled";
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:
            return "enabled";
    }
    GGML_ABORT("fatal error");
}

#define LLAMA_FTYPE_PREFIX "(guessed) "

const char * llama_ftype_name(llama_ftype ftype) {
    static constexpr size_t guessed_prefix_len = sizeof(LLAMA_FTYPE_PREFIX) - 1;
    const char * name;
    switch ((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) {
        case LLAMA_FTYPE_ALL_F32:          name = LLAMA_FTYPE_PREFIX "all F32"; break;
        case LLAMA_FTYPE_MOSTLY_F16:       name = LLAMA_FTYPE_PREFIX "F16"; break;
        case LLAMA_FTYPE_MOSTLY_BF16:      name = LLAMA_FTYPE_PREFIX "BF16"; break;
        case LLAMA_FTYPE_MOSTLY_Q1_0:      name = LLAMA_FTYPE_PREFIX "Q1_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q2_0:      name = LLAMA_FTYPE_PREFIX "Q2_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_0:      name = LLAMA_FTYPE_PREFIX "Q4_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1:      name = LLAMA_FTYPE_PREFIX "Q4_1"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0:      name = LLAMA_FTYPE_PREFIX "Q5_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1:      name = LLAMA_FTYPE_PREFIX "Q5_1"; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0:      name = LLAMA_FTYPE_PREFIX "Q8_0"; break;
        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: name = LLAMA_FTYPE_PREFIX "MXFP4 MoE"; break;
        case LLAMA_FTYPE_MOSTLY_NVFP4:     name = LLAMA_FTYPE_PREFIX "NVFP4"; break;
        case LLAMA_FTYPE_MOSTLY_Q2_K:      name = LLAMA_FTYPE_PREFIX "Q2_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:    name = LLAMA_FTYPE_PREFIX "Q2_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:    name = LLAMA_FTYPE_PREFIX "Q3_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:    name = LLAMA_FTYPE_PREFIX "Q3_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:    name = LLAMA_FTYPE_PREFIX "Q3_K - Large"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:    name = LLAMA_FTYPE_PREFIX "Q4_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:    name = LLAMA_FTYPE_PREFIX "Q4_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:    name = LLAMA_FTYPE_PREFIX "Q5_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:    name = LLAMA_FTYPE_PREFIX "Q5_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:      name = LLAMA_FTYPE_PREFIX "Q6_K"; break;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:     name = LLAMA_FTYPE_PREFIX "TQ1_0 - 1.69 bpw ternary"; break;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:     name = LLAMA_FTYPE_PREFIX "TQ2_0 - 2.06 bpw ternary"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:   name = LLAMA_FTYPE_PREFIX "IQ2_XXS - 2.0625 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:    name = LLAMA_FTYPE_PREFIX "IQ2_XS - 2.3125 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:     name = LLAMA_FTYPE_PREFIX "IQ2_S - 2.5 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:     name = LLAMA_FTYPE_PREFIX "IQ2_M - 2.7 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:    name = LLAMA_FTYPE_PREFIX "IQ3_XS - 3.3 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:   name = LLAMA_FTYPE_PREFIX "IQ3_XXS - 3.0625 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:     name = LLAMA_FTYPE_PREFIX "IQ1_S - 1.5625 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_XS:    name = LLAMA_FTYPE_PREFIX "IQ1_XS - 1.4375 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_XXS:   name = LLAMA_FTYPE_PREFIX "IQ1_XXS - 1.3125 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_XXXS:  name = LLAMA_FTYPE_PREFIX "IQ1_XXXS - 1.1875 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:     name = LLAMA_FTYPE_PREFIX "IQ1_M - 1.75 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:    name = LLAMA_FTYPE_PREFIX "IQ4_NL - 4.5 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:    name = LLAMA_FTYPE_PREFIX "IQ4_XS - 4.25 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:     name = LLAMA_FTYPE_PREFIX "IQ3_S - 3.4375 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:     name = LLAMA_FTYPE_PREFIX "IQ3_S mix - 3.66 bpw"; break;
        default:                           name = LLAMA_FTYPE_PREFIX "unknown, may not work"; break;
    }
    return (ftype & LLAMA_FTYPE_GUESSED) ? name : name + guessed_prefix_len;
}

size_t llama_max_devices(void) {
    return 16;
}

size_t llama_max_tensor_buft_overrides() {
    return 4096;
}

bool llama_supports_gpu_offload(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

const char * llama_version(void) {
    return LLAMA_VERSION;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}
//
// chat templates
//

int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split
//

int32_t llama_split_path(
    char * split_path,
    size_t maxlen,
    const char * path_prefix,
    int32_t split_no,
    int32_t split_count) {

    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";

    const int written = snprintf(
        split_path,
        maxlen,
        SPLIT_PATH_FORMAT,
        path_prefix,
        split_no + 1,
        split_count
    );

    if (written < 0 || (size_t) written >= maxlen) {
        return 0;
    }

    return (int32_t) written;
}

int32_t llama_split_prefix(
    char * split_prefix,
    size_t maxlen,
    const char * split_path,
    int32_t split_no,
    int32_t split_count) {

    const std::string str_split_path(split_path);

    char postfix[32];
    snprintf(postfix, sizeof(postfix), "-%05d-of-%05d.gguf", split_no + 1, split_count);

    const std::string str_postfix(postfix);
    if (str_split_path.size() <= str_postfix.size()) {
        return 0;
    }

    const size_t size_prefix = str_split_path.size() - str_postfix.size();

    if (str_split_path.compare(size_prefix, std::string::npos, str_postfix) == 0) {
        const size_t copy_len = std::min(size_prefix + 1, maxlen);
        snprintf(split_prefix, copy_len, "%s", split_path);

        return (int32_t) size_prefix;
    }

    return 0;
}

const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}

